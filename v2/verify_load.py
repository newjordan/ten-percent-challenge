"""Fresh process: rebuild the model from artifact files alone and compare answers to the in-memory build."""
import json, sys, torch
import runner as R, tasklib as T, fb_lib as L, artifact as A
tok, m, cb = A.load(sys.argv[1])
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
base = json.load(open(R.HERE / "data/open/plain_small.json"))
base_ok = [T.correct(i, o) for i, o in zip(dev, base)]
o = L.predict(tok, m, dev)
ok = [T.correct(i, x) for i, x in zip(dev, o)]
mem = json.load(open(R.HERE / "data/open/inmem_dev.json"))
print("RESULT", json.dumps({"stage": "loaded_from_files", "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2), "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4),
      "acc": round(sum(ok) / len(dev), 4), "identical_to_in_memory": round(sum(a == b for a, b in zip(o, mem)) / len(o), 4), "artifact_bytes": A.artifact_size(sys.argv[1])}), flush=True)
