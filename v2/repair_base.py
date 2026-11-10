import json, time, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, fb_lib as L, lfm, repair_set as S
its = S.items()
print(len(its), flush=True)
path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
t = time.time(); o = L.predict(tok, m, its)
json.dump({"ids": [x["id"] for x in its], "out": o}, open(R.HERE / "data/open/repair_base.json", "w"))
print("DONE", len(o), round(time.time() - t), flush=True)
