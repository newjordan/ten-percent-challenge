import json, torch
import runner as R, tasklib as T
items = R.items_from("dev", 80, 400)
tok, m = R.load()
print(json.dumps({"items": len(items), "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)
outs = R.predict(tok, m, items, log=5)
ok, rep = R.report(items, outs)
print(json.dumps(rep, indent=0))
json.dump([{"id": it["id"], "ok": c, "out": o} for it, c, o in zip(items, ok, outs)], open("data/open/base_dev.json", "w"))
print(json.dumps({"peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))
for it, c, o in list(zip(items, ok, outs))[:6]:
    print(json.dumps({"kind": it["kind"], "gold": it["gold"], "out": o[:120], "ok": c}, ensure_ascii=False))
