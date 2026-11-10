"""Can annotated mistakes teach the model purely by switching off existing hidden units?
Open data only. usage: fb_spike.py MODEL base  |  fb_spike.py MODEL teach kind1,kind2"""
import json, random, sys, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, units as U, fb_lib as L

NAME, MODE = sys.argv[1], sys.argv[2]
SHORT = NAME.split("/")[1]
path = snapshot_download(NAME, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
for p in m.parameters():
    p.requires_grad_(False)
blocks = U.install(m)
dev = R.items_from("dev", 40, 200, salt="v2-fb")
BASEF = R.HERE / f"data/open/fb_base_{SHORT}.json"

if MODE == "base":
    o = L.predict(tok, m, dev)
    ok, rep = R.report(dev, o)
    json.dump({"ids": [x["id"] for x in dev], "ok": ok, "out": o, "rep": rep}, open(BASEF, "w"))
    print("RESULT", json.dumps(rep)); sys.exit(0)

saved = json.load(open(BASEF)); base_ok = dict(zip(saved["ids"], saved["ok"]))
pool_by_kind = {}
for it in T.pos_items("A", R.open_rows("pool"), T.spans_a):
    pool_by_kind.setdefault(it["kind"], []).append(it)
dev_all_by_kind = {}
for it in T.pos_items("A", R.open_rows("dev"), T.spans_a):
    dev_all_by_kind.setdefault(it["kind"], []).append(it)


def attribute(items, labels, kind_loss):
    """Per-unit gradient statistics over items. labels[i] is the answer text used for item i."""
    n, mean, sq = 0, None, None
    U.track(blocks, True)
    for it, ans in zip(items, labels):
        for _, ex in blocks:
            ex.unit_mask.grad = None
        logits, tg = L.forced(tok, m, it, ans)
        loss = kind_loss(logits, tg)
        if isinstance(loss, tuple):
            loss = loss[0]
        if loss is None:
            continue
        loss.backward()
        g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten() for _, ex in blocks]).float()
        mean = g if mean is None else mean + g
        sq = g * g if sq is None else sq + g * g
        n += 1
    U.track(blocks, False)
    return mean / n, (sq / n - (mean / n) ** 2).clamp_min(0), n


def unflatten(indices):
    sizes = [ex.unit_mask.numel() for _, ex in blocks]
    out, off, ib = [], 0, 0
    cuts = torch.tensor(sizes).cumsum(0)
    for i in indices.tolist():
        b = int(torch.searchsorted(cuts, i + 1))
        start = 0 if b == 0 else int(cuts[b - 1])
        e, u = divmod(i - start, blocks[b][1].intermediate_dim)
        out.append((b, e, u))
    return out


def evaluate(fb, te, keep):
    keep_o = L.predict(tok, m, keep)
    te_o = L.predict(tok, m, te)
    fb_o = L.predict(tok, m, fb)
    kc = [T.correct(i, o) for i, o in zip(keep, keep_o)]
    tc = [T.correct(i, o) for i, o in zip(te, te_o)]
    fc = [T.correct(i, o) for i, o in zip(fb, fb_o)]
    return {"fb_fixed": sum(fc), "fb_n": len(fc), "teach_acc": round(sum(tc) / len(tc), 3), "keep_retained": round(sum(kc) / len(kc), 3)}


kinds = sys.argv[3].split(",")
keep_all = [x for x in dev if x["type"] == "pos" and base_ok[x["id"]]]
rng = random.Random(7)
for kind in kinds:
    t0 = time.time()
    cand = sorted(pool_by_kind[kind], key=lambda x: T.sid("fb", x["id"]))[:200]
    cand_o = L.predict(tok, m, cand)
    wrong = [c for c, o in zip(cand, cand_o) if not T.correct(c, o)]
    fb = wrong[:25]
    te = sorted(dev_all_by_kind[kind], key=lambda x: T.sid("te", x["id"]))[:120]
    keep = [x for x in sorted(keep_all, key=lambda x: T.sid("kp", x["id"])) if x["kind"] != kind][:150]
    print(json.dumps({"kind": kind, "pool_acc": round(1 - len(wrong) / len(cand), 3), "n_fb": len(fb), "n_teach": len(te), "n_keep": len(keep)}), flush=True)
    base = evaluate(fb, te, keep)
    print("CELL", json.dumps({"kind": kind, "cell": "empty_feedback", **base}), flush=True)
    answers = [L.answer_text(x) for x in fb]
    shuf = answers[1:] + answers[:1]
    gm, gv, n = attribute(fb, answers, L.mistake_loss)
    keep_own = L.predict(tok, m, keep[:100])
    sm, _, _ = attribute(keep[:100], [o.strip() for o in keep_own], L.margin_loss)
    sens = sm.abs()
    gs, _, ns = attribute(fb, shuf, L.mistake_loss)
    t = gm / (gv / n).sqrt().clamp_min(1e-9)
    print(json.dumps({"attributed_items": n, "secs": round(time.time() - t0)}), flush=True)
    for lam in (1.0, 5.0):
        score = (gm - lam * sens).where(t > 2.0, torch.tensor(-1e9, device=gm.device))
        for N in (10, 100, 1000, 3000):
            idx = score.topk(N).indices
            U.reset(blocks); U.apply_off(blocks, unflatten(idx))
            print("CELL", json.dumps({"kind": kind, "cell": "real", "lam": lam, "N": N, "pred_gain": round(float(gm[idx].sum()), 2), **evaluate(fb, te, keep)}), flush=True)
    U.reset(blocks)
    for N in (100, 1000):
        U.reset(blocks); U.apply_off(blocks, unflatten(gs.topk(N).indices))
        print("CELL", json.dumps({"kind": kind, "cell": "shuffled_labels", "N": N, **evaluate(fb, te, keep)}), flush=True)
        U.reset(blocks); U.apply_off(blocks, unflatten(torch.randperm(gm.numel(), device=gm.device)[:N]))
        print("CELL", json.dumps({"kind": kind, "cell": "random_units", "N": N, **evaluate(fb, te, keep)}), flush=True)
    U.reset(blocks)
    print(json.dumps({"kind_done": kind, "secs": round(time.time() - t0)}), flush=True)
print("ALLDONE", flush=True)
