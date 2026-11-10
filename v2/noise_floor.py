"""Identical weights, different code path: how much do correct answers move? Open data only."""
import gc, json, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, fb_lib as L, lfm

path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
ok = lambda outs: [T.correct(i, o) for i, o in zip(dev, outs)]
def stats(a, b, name):
    oa, ob = ok(a), ok(b)
    return {"pair": name, "identical": round(sum(x == y for x, y in zip(a, b)) / len(a), 4), "faithful_b_on_a_correct": round(sum(p and q for p, q in zip(oa, ob)) / sum(oa), 4), "acc_a": round(sum(oa) / len(oa), 4), "acc_b": round(sum(ob) / len(ob), 4)}
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
print(json.dumps({"default_experts_implementation": m.config._experts_implementation}), flush=True)
plain = L.predict(tok, m, dev)
json.dump(plain, open(R.HERE / "data/open/plain_small.json", "w"))
del m; gc.collect(); torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True, experts_implementation="eager").to("cuda").eval()
print(json.dumps({"second_path": m.config._experts_implementation}), flush=True)
eager = L.predict(tok, m, dev)
print("RESULT", json.dumps(stats(plain, eager, "default_vs_eager")), flush=True)
patched = json.load(open(R.HERE / "data/open/base_outputs.json"))["base"]
print("RESULT", json.dumps(stats(plain, patched, "default_vs_my_patched_path")), flush=True)
mem = json.load(open(R.HERE / "data/open/inmem_dev.json"))
print("RESULT", json.dumps(stats(plain, mem, "default_vs_artifact")), flush=True)
print("ALLDONE", flush=True)
