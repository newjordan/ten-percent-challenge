"""Build the shipped artifact from the base checkpoint. Open data only. usage: build.py OUT_DIR"""
import gzip, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import runner as R, tasklib as T, fb_lib as L, lfm, compact as C, artifact as A, units as U

OUT = Path(sys.argv[1])
KEEP, FRAC3, LIN_BITS, EMB_BITS = 50, 0.45, 6, 6
HDIR = R.HERE / f"data/open/hess_{KEEP}"
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
calib = R.items_from("calib", 16, 120, salt="v2-unit")
cached = json.load(open(R.HERE / "data/open/base_outputs.json")); own = cached["own"]; base = json.load(open(R.HERE / "data/open/plain_small.json"))
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
tok, m, _, bs = lfm.load()
FP16 = 2 * sum(p.numel() for p in m.parameters())
alive = torch.load(f"data/open/alive_{KEEP}.pt").to("cuda")
imp = torch.load("data/open/unit_importance.pt").to("cuda")
cb = C.compactify(m, alive)
print(json.dumps({"fp16_bytes": FP16, "budget_bytes": FP16 // 10, "alive_units": int(alive.sum())}), flush=True)

lin_h, hooks = {}, []
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and "lm_head" not in name and min(mod.weight.shape) >= 64:
        def hook(md, args, n=name):
            x = args[0].reshape(-1, args[0].shape[-1]).float().detach()
            lin_h[n] = lin_h.get(n, 0) + x.T @ x
        hooks.append(mod.register_forward_pre_hook(hook))
with torch.no_grad():
    for it, o in zip(calib, own):
        if o:
            L.forced(tok, m, it, o)
for h in hooks:
    h.remove()

units_per_block = cb[0][1].num_experts * cb[0][1].intermediate_dim
ranked = []
for b, (_, ex) in enumerate(cb):
    bi = imp[b * units_per_block:(b + 1) * units_per_block].view(ex.num_experts, ex.intermediate_dim)
    for e in ex.alive_experts():
        ranked.append((float(bi[e][ex.keep[e].to(bi.device)].mean()), len(ex.keep[e]), b, e))
ranked.sort()
tot = sum(r[1] for r in ranked); three, acc = set(), 0
for _, u, b, e in ranked:
    if acc / tot >= FRAC3:
        break
    three.add((b, e)); acc += u

if OUT.exists():
    shutil.rmtree(OUT)
w = A.Writer(OUT)
keep_index = {}
with torch.no_grad():
    for b, (name, ex) in enumerate(cb):
        hb = torch.load(HDIR / f"b{b}.pt", map_location="cuda")
        keep_index[name] = []
        for e in range(ex.num_experts):
            if len(ex.keep[e]) == 0:
                keep_index[name].append(""); continue
            bits_arr = np.zeros(ex.intermediate_dim, dtype=np.uint8); bits_arr[ex.keep[e].numpy()] = 1
            keep_index[name].append(np.packbits(bits_arr).tobytes().hex())
            bits = 3 if (b, e) in three else 4
            for nm, H in ((f"gu{e}", hb["g"].get(e)), (f"d{e}", hb["d"].get(e))):
                W = getattr(ex, nm)
                H = H if H is not None else torch.eye(W.shape[1], device=W.device)
                Q, codes, s = C.gptq(W, H, bits)
                W.copy_(Q)
                w.quantized(f"{name}.{nm}", codes, s, bits)
        del hb
    for n, p in m.named_parameters():
        mod_name = n.rsplit(".", 1)[0]
        if p.ndim >= 2 and mod_name in lin_h:
            Q, codes, s = C.gptq(p, lin_h[mod_name], LIN_BITS)
            p.copy_(Q.to(p.dtype)); w.quantized(n, codes, s, LIN_BITS)
        elif p.ndim >= 2:
            bits = EMB_BITS if "embed_tokens" in n else 8
            W = p.float(); s = (W.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 0.5)).to(torch.bfloat16).float()
            q = (W / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            p.copy_((q * s).to(p.dtype)); w.quantized(n, q.to(torch.int16), s.flatten(), bits)
        else:
            w.raw(n, p)
    for n, buf in m.named_buffers():
        if ".gu" in n or (".d" in n and n.rsplit(".", 1)[1].startswith("d") and n.rsplit(".", 1)[1][1:].isdigit()):
            continue
        w.raw(n, buf)

# build-time importance of every surviving unit on unlabeled calibration text (used only to protect units during feedback)
U.track(cb, True)
tot_g, n = None, 0
for it, o in zip(calib, own):
    if not o:
        continue
    for _, ex in cb:
        ex.unit_mask.grad = None
    logits, tg = L.forced(tok, m, it, o)
    L.margin_loss(logits, tg).backward()
    g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten() for _, ex in cb]).abs().float()
    tot_g = g if tot_g is None else tot_g + g
    n += 1
U.track(cb, False)
imp_final = tot_g / n
w.close({"keep": keep_index, "fp16_bytes": FP16, "config": {"keep_pct": KEEP, "frac3": FRAC3, "lin_bits": LIN_BITS, "emb_bits": EMB_BITS},
         "importance_scale": float(imp_final.max())})
np.save(OUT / "importance.npy", imp_final.cpu().numpy().astype(np.float16))
(OUT / "tok").mkdir()
src = Path(lfm.snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS))
for f in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "chat_template.jinja"):
    if (src / f).exists():
        with open(src / f, "rb") as a, gzip.open(OUT / "tok" / (f + ".gz"), "wb", compresslevel=9) as b:
            shutil.copyfileobj(a, b)
for f in ("artifact.py", "compact.py", "moe.py", "fb_lib.py", "feedback_core.py", "units.py", "tasklib.py", "answer.py", "feedback.py"):
    if (R.HERE / f).exists():
        shutil.copy(R.HERE / f, OUT / f)
o = L.predict(tok, m, dev)
ok = [T.correct(i, x) for i, x in zip(dev, o)]
json.dump(o, open(R.HERE / "data/open/inmem_dev.json", "w"))
size = A.artifact_size(OUT)
print("RESULT", json.dumps({"stage": "in_memory", "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4), "acc": round(sum(ok) / len(dev), 4),
      "artifact_bytes": size, "pct_of_fp16": round(100 * size / FP16, 3)}), flush=True)
print("ALLDONE", flush=True)
