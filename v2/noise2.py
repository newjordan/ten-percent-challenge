"""With nothing pruned or rounded, does the compact model now reproduce the untouched base? Open data only."""
import json, torch
import runner as R, tasklib as T, fb_lib as L, lfm, compact as C
tok, m, _, bs = lfm.load()
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
plain = json.load(open(R.HERE / "data/open/plain_small.json"))
alive = torch.ones(sum(ex.num_experts * ex.intermediate_dim for *_, ex in bs), dtype=torch.bool, device="cuda")
cb = C.compactify(m, alive)
o = L.predict(tok, m, dev)
oa, ob = [T.correct(i, x) for i, x in zip(dev, plain)], [T.correct(i, x) for i, x in zip(dev, o)]
print("RESULT", json.dumps({"compact_full_precision_vs_default": True, "identical": round(sum(a == b for a, b in zip(plain, o)) / len(o), 4),
      "faithful": round(sum(a and b for a, b in zip(oa, ob)) / sum(oa), 4), "acc": round(sum(ob) / len(ob), 4), "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)
