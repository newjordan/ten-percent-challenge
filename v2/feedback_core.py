"""Learning from annotated mistakes by switching off existing hidden units.

No weight value changes and nothing is added: the only state that moves is the on/off mask the artifact already has.
Inputs are the annotated items alone, plus a per-unit importance score computed at build time from unlabeled text.
Works in small rounds, re-measures after each one, and stops once the annotated mistakes are fixed."""
import torch
import fb_lib as L, tasklib as T, units as U


def flat_mask(blocks):
    return torch.cat([ex.unit_mask.flatten() for _, ex in blocks]) > 0


def _unflatten(blocks, indices):
    sizes = torch.tensor([ex.unit_mask.numel() for _, ex in blocks]).cumsum(0)
    out = []
    for i in indices.tolist():
        b = int(torch.searchsorted(sizes, i + 1))
        out.append((b, i - (0 if b == 0 else int(sizes[b - 1]))))
    return out


def grads(tok, m, blocks, items):
    """Per-item gradient of the mistake loss with respect to every unit switch. Returns (mean, variance, n)."""
    U.track(blocks, True)
    mean, sq, n = None, None, 0
    for it in items:
        for _, ex in blocks:
            ex.unit_mask.grad = None
        logits, tg = L.forced(tok, m, it, L.answer_text(it))
        loss, _ = L.mistake_loss(logits, tg)
        if loss is None:
            continue
        loss.backward()
        g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten()
                       for _, ex in blocks]).float()
        mean = g if mean is None else mean + g
        sq = g * g if sq is None else sq + g * g
        n += 1
    U.track(blocks, False)
    if n == 0:
        return None, None, 0
    return mean / n, (sq / n - (mean / n) ** 2).clamp_min(0), n


def fixed_count(tok, m, items):
    return sum(T.correct(i, o) for i, o in zip(items, L.predict(tok, m, items)))


def teach(tok, m, blocks, items, importance, lam=1.0, round_size=25, max_flips=400, target=0.8):
    """items carry the annotation in item['gold']. Returns flips and a per-round log."""
    flips, log = [], []
    goal = target * len(items)
    while True:
        now = fixed_count(tok, m, items) if items else 0
        log.append({"flips": len(flips), "annotated_fixed": now})
        if not items or now >= goal or len(flips) >= max_flips:
            break
        gm, gv, n = grads(tok, m, blocks, items)
        if gm is None:
            break
        t = gm / (gv / n).sqrt().clamp_min(1e-9)
        score = gm - lam * importance.to(gm.device)
        score = score.where((t > 2.0) & flat_mask(blocks), torch.tensor(-1e9, device=gm.device))
        idx = score.topk(round_size).indices
        idx = idx[score[idx] > -1e8]
        if len(idx) == 0:
            break
        picks = _unflatten(blocks, idx)
        U.apply_off(blocks, picks)
        flips += picks
    return flips, log
