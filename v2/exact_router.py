"""Tensor-level check of the routing patch, then a staged search for what still differs."""
import json, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, fb_lib as L, lfm, moe

path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
caps, hooks, bl = {}, [], moe.blocks(m)
for i, (name, owner, router, ex) in enumerate(bl):
    def post(mod, args, out, i=i):
        caps.setdefault(i, ([a.detach().clone() if a is not None else None for a in args], [o.detach().clone() for o in out]))
    hooks.append(router.register_forward_hook(post))
enc = tok([L.chat(tok, it) for it in dev[:8]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
enc.pop("token_type_ids", None)
with torch.no_grad():
    m(**enc, use_cache=False)
for h in hooks:
    h.remove()
bs = moe.install_ban(m)
rows = []
for i in (0, 5, 12, 21):
    args, out = caps[i]
    with torch.no_grad():
        o2 = bs[i][2](*args)
    rows.append({"block": i, "logits_equal": bool(torch.equal(out[0], o2[0])), "weights_equal": bool(torch.equal(out[1], o2[1])), "index_equal": bool(torch.equal(out[2], o2[2])),
                 "dtypes": [str(out[1].dtype), str(o2[1].dtype)]})
print("RESULT", json.dumps(rows), flush=True)
o = L.predict(tok, m, dev)
first = json.load(open(R.HERE / "data/open/plain_small.json"))
print("RESULT", json.dumps({"plain_plus_routing_patch_identical": round(sum(a == b for a, b in zip(first, o)) / len(o), 4)}), flush=True)
