"""Structural cut at hidden-unit level. Importance of each unit = mean absolute effect of removing it on the model's
own answer margins, measured on unlabeled calibration documents. Full-precision weights; open data only."""
import json, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, units as U, fb_lib as L, moe

NAME = "LiquidAI/LFM2-8B-A1B"
path = snapshot_download(NAME, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
for p in m.parameters():
    p.requires_grad_(False)
blocks = U.install(m)
bs = moe.install_ban(m)
calib = R.items_from("calib", 8, 40, salt="v2-unit")
dev = R.items_from("dev", 10, 50, salt="v2-prune")
t0 = time.time()
own = L.predict(tok, m, calib)
base = L.predict(tok, m, dev)
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
print("RESULT", json.dumps({"k": "all", "acc": round(sum(base_ok) / len(dev), 4), "n": len(dev), "secs": round(time.time() - t0)}), flush=True)

U.track(blocks, True)
tot, n = None, 0
for it, o in zip(calib, own):
    if not o.strip():
        continue
    for _, ex in blocks:
        ex.unit_mask.grad = None
    logits, tg = L.forced(tok, m, it, o.strip())
    L.margin_loss(logits, tg).backward()
    g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten() for _, ex in blocks]).abs().float()
    tot = g if tot is None else tot + g
    n += 1
U.track(blocks, False)
imp = tot / n
torch.save(imp.cpu(), "data/open/unit_importance.pt")
sizes = [ex.unit_mask.numel() for _, ex in blocks]
nz = (imp > 0).float().mean()
print(json.dumps({"attributed_items": n, "secs": round(time.time() - t0), "units": imp.numel(), "units_never_used": round(float(1 - nz), 4)}), flush=True)
srt = imp.sort(descending=True).values
c = srt.cumsum(0) / srt.sum()
print(json.dumps({f"attribution_share_top_{p}pct": round(float(c[int(imp.numel() * p / 100) - 1]), 4) for p in (10, 20, 30, 40, 50, 60, 70, 80)}), flush=True)
exp_share = 0.9294
for keep in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    t1 = time.time()
    th = srt[int(imp.numel() * keep) - 1]
    alive = imp >= th
    off, ban_n = 0, 0
    for (name, ex), (_, _, router, _), a in zip(blocks, bs, torch.split(alive, sizes)):
        am = a.view(ex.num_experts, ex.intermediate_dim)
        ex.unit_mask.data.copy_(am.float())
        router.banned.copy_(am.float().mean(1) < 0.05)
        ban_n += int(router.banned.sum())
    o = L.predict(tok, m, dev)
    ok = [T.correct(i, x) for i, x in zip(dev, o)]
    print("RESULT", json.dumps({"units_kept": keep, "experts_banned": ban_n, "acc": round(sum(ok) / len(dev), 4),
          "faithful_on_base_correct": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
          "identical_output": round(sum(x == y for x, y in zip(o, base)) / len(dev), 4),
          "params_left_share": round(1 - exp_share * (1 - keep), 4), "secs": round(time.time() - t1)}), flush=True)
print("ALLDONE", flush=True)
