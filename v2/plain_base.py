"""The untouched model: no patches, nothing installed. Saves its answers on a large open sample and checks they match the patched baseline."""
import json, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, fb_lib as L, lfm

path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
big = R.items_from("dev", 80, 400, salt="v2-big")
t = time.time(); o = L.predict(tok, m, big); ok = [T.correct(i, x) for i, x in zip(big, o)]
json.dump({"ids": [x["id"] for x in big], "out": o}, open(R.HERE / "data/open/plain_big.json", "w"))
print("RESULT", json.dumps({"n": len(big), "acc": round(sum(ok) / len(big), 4), "secs": round(time.time() - t)}), flush=True)
small = R.items_from("dev", 20, 100, salt="v2-prune2")
s = L.predict(tok, m, small)
cached = json.load(open(R.HERE / "data/open/base_outputs.json"))["base"]
print("RESULT", json.dumps({"plain_vs_patched_identical": round(sum(a == b for a, b in zip(s, cached)) / len(s), 4), "n": len(s)}), flush=True)
print("ALLDONE", flush=True)
