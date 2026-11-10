"""Final allocation comparison on a large open sample. usage: gptq3.py KEEP "frac3,frac5" ...  Open data only."""
import gc, json, sys, time
import torch
from torch import nn
import runner as R, tasklib as T, fb_lib as L, lfm, compact as C

dev = R.items_from("dev", 80, 400, salt="v2-big")
calib = R.items_from("calib", 16, 120, salt="v2-unit")
cached = json.load(open(R.HERE / "data/open/base_outputs.json")); own = cached["own"]
big = json.load(open(R.HERE / "data/open/plain_big.json")); assert big["ids"] == [x["id"] for x in dev]; base = big["out"]
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
keep = int(sys.argv[1]); cells = sys.argv[2:]
hdir = R.HERE / f"data/open/hess_{keep}"
tok, m, blocks, bs = lfm.load()
budget = 2 * sum(p.numel() for p in m.parameters()) * 0.1 / 1e6
alive = torch.load(f"data/open/alive_{keep}.pt").to("cuda")
imp = torch.load("data/open/unit_importance.pt").to("cuda")
cb = C.compactify(m, alive)

lin_h = {}; hooks = []
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and "lm_head" not in name and min(mod.weight.shape) >= 64:
        def hook(md, args, n=name):
            x = args[0].reshape(-1, args[0].shape[-1]).float().detach()
            lin_h[n] = lin_h.get(n, 0) + x.T @ x
        hooks.append(mod.register_forward_pre_hook(hook))
t0 = time.time()
with torch.no_grad():
    for it, o in zip(calib, own):
        if o:
            L.forced(tok, m, it, o)
for h in hooks:
    h.remove()
print(json.dumps({"linear_hessians": len(lin_h), "secs": round(time.time() - t0), "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)

# expert importance (mean over surviving units) and volume, for bit allocation
units_per_block = cb[0][1].num_experts * cb[0][1].intermediate_dim
ranked = []
for b, (_, ex) in enumerate(cb):
    block_imp = imp[b * units_per_block:(b + 1) * units_per_block].view(ex.num_experts, ex.intermediate_dim)
    for e in ex.alive_experts():
        ranked.append((float(block_imp[e][ex.keep[e].to(block_imp.device)].mean()), len(ex.keep[e]), b, e))
ranked.sort()
tot_units = sum(r[1] for r in ranked)
snap = {f"{b}:{k}": v.cpu().clone() for b, (_, ex) in enumerate(cb) for k, v in ex.named_buffers()}
other_snap = {n: p.detach().cpu().clone() for n, p in m.named_parameters()}

for spec in cells:
    frac3, frac5 = [float(x) for x in spec.split(",")]; lin_bits, emb_bits = 6, 6
    t1 = time.time()
    three, acc = set(), 0
    for _, u, b, e in ranked:
        if acc / tot_units >= frac3:
            break
        three.add((b, e)); acc += u
    five, acc = set(), 0
    for _, u, b, e in reversed(ranked):
        if acc / tot_units >= frac5:
            break
        five.add((b, e)); acc += u
    mb_e, mb_o = 0.0, 0.0
    with torch.no_grad():
        for b, (_, ex) in enumerate(cb):
            for k, v in ex.named_buffers():
                v.copy_(snap[f"{b}:{k}"].to(v.device))
            hb = torch.load(hdir / f"b{b}.pt", map_location="cuda")
            for e in ex.alive_experts():
                bits = 3 if (b, e) in three else (5 if (b, e) in five else 4)
                for nm, H in ((f"gu{e}", hb["g"].get(e)), (f"d{e}", hb["d"].get(e))):
                    W = getattr(ex, nm)
                    H = H if H is not None else torch.eye(W.shape[1], device=W.device)
                    Q, codes, _ = C.gptq(W, H, bits)
                    W.copy_(Q)
                    mb_e += C.entropy_bits(codes, bits) / 8e6
            del hb
        for n, p in m.named_parameters():
            p.copy_(other_snap[n].to(p.device))
            if p.ndim < 2 or not torch.is_floating_point(p):
                continue
            mod_name = n.rsplit(".", 1)[0]
            if mod_name in lin_h:
                Q, codes, _ = C.gptq(p, lin_h[mod_name], lin_bits)
                p.copy_(Q.to(p.dtype)); mb_o += C.entropy_bits(codes, lin_bits) / 8e6
            else:
                bits = emb_bits if "embed_tokens" in n else 8
                W = p.float(); s = W.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 0.5)
                q = (W / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                p.copy_((q * s).to(p.dtype)); mb_o += C.entropy_bits(q.to(torch.int16), bits) / 8e6
    o = L.predict(tok, m, dev)
    ok = [T.correct(i, x) for i, x in zip(dev, o)]
    per = {}
    for it, bo, co in zip(dev, base_ok, ok):
        k = it["kind"] if it["type"] == "pos" else "<absent>"
        if bo:
            per.setdefault(k, [0, 0]); per[k][0] += co; per[k][1] += 1
    worst = sorted(((round(a_ / n_, 3), k_) for k_, (a_, n_) in per.items()))[:6]
    print("WORST", json.dumps(worst), "below_95", sum(a_ / n_ < 0.95 for a_, n_ in per.values()), "of", len(per), flush=True)
    print("RESULT", json.dumps({"keep": keep, "frac_at_3bit": frac3, "frac_at_5bit": frac5, "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
          "acc": round(sum(ok) / len(dev), 4), "identical": round(sum(x == y for x, y in zip(o, base)) / len(dev), 4), "mb_experts": round(mb_e, 1), "mb_other": round(mb_o, 1),
          "pct_of_fp16": round(100 * (mb_e + mb_o) / (budget * 10), 2), "secs": round(time.time() - t1)}), flush=True)
    gc.collect(); torch.cuda.empty_cache()
print("ALLDONE", flush=True)
