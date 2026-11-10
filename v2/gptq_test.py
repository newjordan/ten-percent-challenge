"""Hessian-aware rounding on the compact model. usage: gptq_test.py KEEP:BITS[,BITS] ...  Open data only."""
import json, os, sys, time
import torch
import runner as R, tasklib as T, fb_lib as L, lfm, compact as C

dev = R.items_from("dev", 20, 100, salt="v2-prune2")
calib = R.items_from("calib", 16, 120, salt="v2-unit")
CACHE = R.HERE / "data/open/base_outputs.json"
OTHER_BITS = 8


def quant_other(m, expert_buffers, bits):
    total, n = 0.0, 0
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim < 2 or not torch.is_floating_point(p):
                continue
            W = p.float()
            s = W.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 0.5)
            q = (W / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            p.copy_((q * s).to(p.dtype))
            total += C.entropy_bits(q.to(torch.int16), bits); n += q.numel()
    return total / 8e6, n


for spec in sys.argv[1:]:
    keep, bits_list = spec.split(":")
    keep = int(keep); bits_list = [int(b) for b in bits_list.split(",")]
    tok, m, blocks, bs = lfm.load()
    budget = 2 * sum(p.numel() for p in m.parameters()) * 0.1 / 1e6
    if CACHE.exists():
        cached = json.load(open(CACHE)); base, own = cached["base"], cached["own"]
    else:
        base = L.predict(tok, m, dev); own = [o.strip() for o in L.predict(tok, m, calib)]
        json.dump({"base": base, "own": own}, open(CACHE, "w"))
    base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
    alive = torch.load(f"data/open/alive_{keep}.pt").to("cuda")
    cb = C.compactify(m, alive)
    print(json.dumps({"keep": keep, "gpu_gb_after_compact": round(torch.cuda.memory_allocated() / 1e9, 2), "alive_units": int(alive.sum())}), flush=True)
    t0 = time.time()
    for _, ex in cb:
        ex.collect = True
    with torch.no_grad():
        for it, o in zip(calib, own):
            if o:
                L.forced(tok, m, it, o)
    for _, ex in cb:
        ex.collect = False
    print(json.dumps({"hessian_secs": round(time.time() - t0)}), flush=True)
    hdir = R.HERE / f"data/open/hess_{keep}"; hdir.mkdir(exist_ok=True)
    for b, (_, ex) in enumerate(cb):
        torch.save({"g": {e: h.cpu() for e, h in ex.Hg.items()}, "d": {e: h.cpu() for e, h in ex.Hd.items()}}, hdir / f"b{b}.pt")
        ex.Hg, ex.Hd = {}, {}
    torch.cuda.empty_cache()
    snap = {f"{b}:{k}": v.cpu().clone() for b, (_, ex) in enumerate(cb) for k, v in ex.named_buffers()}
    other_snap = {n: p.detach().cpu().clone() for n, p in m.named_parameters()}
    for bits in bits_list:
        t1 = time.time()
        mb_e, ne = 0.0, 0
        with torch.no_grad():
            for b, (_, ex) in enumerate(cb):
                for k, v in ex.named_buffers():
                    v.copy_(snap[f"{b}:{k}"].to(v.device))
                hb = torch.load(hdir / f"b{b}.pt", map_location="cuda")
                for e in ex.alive_experts():
                    for nm, H in ((f"gu{e}", hb["g"].get(e)), (f"d{e}", hb["d"].get(e))):
                        W = getattr(ex, nm)
                        if H is None:
                            H = torch.eye(W.shape[1], device=W.device)
                        Q, codes, _ = C.gptq(W, H, bits)
                        W.copy_(Q)
                        mb_e += C.entropy_bits(codes, bits) / 8e6
                        ne += codes.numel()
            for n, p in m.named_parameters():
                p.copy_(other_snap[n].to(p.device))
        mb_o, no = quant_other(m, None, OTHER_BITS)
        q_s = time.time() - t1
        o = L.predict(tok, m, dev)
        ok = [T.correct(i, x) for i, x in zip(dev, o)]
        print("RESULT", json.dumps({"keep": keep, "exp_bits": bits, "other_bits": OTHER_BITS, "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
              "acc": round(sum(ok) / len(dev), 4), "identical": round(sum(x == y for x, y in zip(o, base)) / len(dev), 4), "expert_params_b": round(ne / 1e9, 3),
              "entropy_mb_experts": round(mb_e, 1), "entropy_mb_other": round(mb_o, 1), "bits_per_expert_weight": round(mb_e * 8e6 / ne, 3),
              "pct_of_fp16": round(100 * (mb_e + mb_o) / (budget * 10), 2), "quant_secs": round(q_s), "total_secs": round(time.time() - t1)}), flush=True)
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(other_snap[n].to(p.device))
    del m, cb, blocks, bs
    import gc; gc.collect(); torch.cuda.empty_cache()
print("ALLDONE", flush=True)
