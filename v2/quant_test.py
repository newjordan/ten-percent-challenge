"""How few bits can the remaining weights tolerate? Rounds in place, reports faithfulness and estimated entropy-coded size.
usage: quant_test.py KEEP_PERCENT. Open data only."""
import json, math, sys, time
import torch
import runner as R, tasklib as T, fb_lib as L, lfm

KEEP = int(sys.argv[1])
tok, m, blocks, bs = lfm.load()
alive = torch.load(f"data/open/alive_{KEEP}.pt").to("cuda")
sizes = [ex.unit_mask.numel() for _, ex in blocks]
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
base = L.predict(tok, m, dev)
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
lfm.apply_alive(blocks, bs, alive)
pruned = L.predict(tok, m, dev)
pruned_ok = [T.correct(i, o) for i, o in zip(dev, pruned)]
keep_snapshot = {n: p.detach().clone().cpu() for n, p in m.named_parameters()}
print("RESULT", json.dumps({"cell": "pruned_full_precision", "keep": KEEP, "faithful": round(sum(a and b for a, b in zip(base_ok, pruned_ok)) / sum(base_ok), 4)}), flush=True)


def entropy(hist):
    p = hist[hist > 0].float() / hist.sum()
    return float(-(p * p.log2()).sum())


def quant(W, bits, dim, clip):
    """Scalar rounding with one step per group along dim. Returns reconstruction and integer codes."""
    W = W.float()
    if clip is None:
        step = W.abs().amax(dim=dim, keepdim=True) / (2 ** (bits - 1) - 0.5)
    else:
        step = 2 * clip * W.std(dim=dim, keepdim=True) / (2 ** bits)
    step = step.clamp_min(1e-12)
    q = (W / step).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (q * step).to(torch.bfloat16), q.to(torch.int16)


def apply(exp_bits, other_bits, clip):
    with torch.no_grad():
        n_exp, n_other, hist_bits, ids = 0, 0, 0.0, {id(ex.gate_up_proj): ex for _, ex in blocks} | {id(ex.down_proj): ex for _, ex in blocks}
        total_bits = 0.0
        for (b, (_, ex)), a in zip(enumerate(blocks), torch.split(alive, sizes)):
            am = a.view(ex.num_experts, ex.intermediate_dim)
            for e in range(ex.num_experts):
                keep = am[e].nonzero().flatten()
                if len(keep) == 0 or am[e].float().mean() < 0.05:
                    continue
                gu = ex.gate_up_proj[e]; d = ex.down_proj[e]
                idx2 = torch.cat([keep, keep + ex.intermediate_dim])
                ga, qa = quant(gu[idx2], exp_bits, 1, clip)
                da, qd = quant(d[:, keep], exp_bits, 0, clip)
                gu[idx2] = ga; d[:, keep] = da
                for q in (qa, qd):
                    h = torch.bincount((q + 2 ** (exp_bits - 1)).flatten().long(), minlength=2 ** exp_bits)
                    total_bits += entropy(h) * q.numel()
                    n_exp += q.numel()
        expert_ids = {id(ex.gate_up_proj) for _, ex in blocks} | {id(ex.down_proj) for _, ex in blocks}
        other_coded = 0.0
        for n, p in m.named_parameters():
            if id(p) in expert_ids or p.ndim < 2 or not torch.is_floating_point(p):
                continue
            a, q = quant(p, other_bits, -1, None)
            p.copy_(a)
            h = torch.bincount((q + 2 ** (other_bits - 1)).flatten().long(), minlength=2 ** other_bits)
            other_coded += entropy(h) * q.numel()
            n_other += q.numel()
    return n_exp, n_other, total_bits / 8e6, other_coded / 8e6


def restore():
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(keep_snapshot[n].to(p.device))


budget = 2 * sum(p.numel() for p in m.parameters()) * 0.1 / 1e6
print(json.dumps({"budget_mb": round(budget, 1)}), flush=True)
for exp_bits, other_bits, clip in [(4, 8, None), (4, 8, 3.0), (4, 6, 3.0), (3, 6, 3.0), (3, 6, None), (5, 6, 3.0), (2, 6, 3.0)]:
    t = time.time()
    restore(); ne, no, mb_e, mb_o = apply(exp_bits, other_bits, clip)
    o = L.predict(tok, m, dev)
    ok = [T.correct(i, x) for i, x in zip(dev, o)]
    print("RESULT", json.dumps({"exp_bits": exp_bits, "other_bits": other_bits, "clip_sigma": clip, "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
          "acc": round(sum(ok) / len(dev), 4), "expert_params_b": round(ne / 1e9, 3), "entropy_mb_experts": round(mb_e, 1), "entropy_mb_other": round(mb_o, 1),
          "entropy_bits_per_expert_weight": round(mb_e * 8e6 / max(1, ne), 3), "pct_of_fp16": round(100 * (mb_e + mb_o) / (budget * 10), 2), "secs": round(time.time() - t)}), flush=True)
print("ALLDONE", flush=True)
