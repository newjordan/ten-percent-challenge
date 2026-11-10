"""Iterative hidden-unit removal: re-measure importance after each cut, always against the base's own answers.
Open data only. Saves the alive mask at each step."""
import json, time
import torch
import runner as R, tasklib as T, units as U, fb_lib as L, lfm

tok, m, blocks, bs = lfm.load()
calib = R.items_from("calib", 16, 120, salt="v2-unit")
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
t0 = time.time()
own = [o.strip() for o in L.predict(tok, m, calib)]
base = L.predict(tok, m, dev)
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
print("RESULT", json.dumps({"keep": 1.0, "acc": round(sum(base_ok) / len(dev), 4), "n": len(dev), "secs": round(time.time() - t0)}), flush=True)


def importance():
    U.track(blocks, True)
    tot, n = None, 0
    for it, o in zip(calib, own):
        if not o:
            continue
        for _, ex in blocks:
            ex.unit_mask.grad = None
        logits, tg = L.forced(tok, m, it, o)
        L.margin_loss(logits, tg).backward()
        g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten() for _, ex in blocks]).abs().float()
        tot = g if tot is None else tot + g
        n += 1
    U.track(blocks, False)
    return tot / n


alive = torch.ones(sum(ex.unit_mask.numel() for _, ex in blocks), dtype=torch.bool, device="cuda")
for keep in (0.85, 0.7, 0.6, 0.5, 0.4, 0.3):
    t1 = time.time()
    imp = importance()
    imp[~alive] = -1
    n_keep = int(keep * alive.numel())
    th = imp.kthvalue(alive.numel() - n_keep + 1).values
    alive = imp >= th
    banned = lfm.apply_alive(blocks, bs, alive)
    torch.save(alive.cpu(), f"data/open/alive_{int(keep * 100)}.pt")
    o = L.predict(tok, m, dev)
    ok = [T.correct(i, x) for i, x in zip(dev, o)]
    print("RESULT", json.dumps({"keep": keep, "experts_banned": banned, "acc": round(sum(ok) / len(dev), 4),
          "faithful_on_base_correct": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
          "identical_output": round(sum(x == y for x, y in zip(o, base)) / len(dev), 4), "secs": round(time.time() - t1)}), flush=True)
print("ALLDONE", flush=True)
