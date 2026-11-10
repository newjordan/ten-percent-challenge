"""Tensor-level check of the compact expert path against the library's default path, plus run-to-run determinism."""
import json, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, fb_lib as L, lfm, compact as C, moe

path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
# capture real inputs to each expert block
caps = {}
hooks = []
bl = moe.blocks(m)
for i, (name, owner, router, ex) in enumerate(bl):
    def pre(mod, args, i=i):
        caps.setdefault(i, [a.detach().clone() for a in args])
    def post(mod, args, out, i=i):
        caps[i].append(out.detach().clone()) if len(caps[i]) == 3 else None
    hooks.append(ex.register_forward_pre_hook(pre)); hooks.append(ex.register_forward_hook(post))
enc = tok([L.chat(tok, it) for it in dev[:8]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
enc.pop("token_type_ids", None)
with torch.no_grad():
    m(**enc, use_cache=False)
for h in hooks:
    h.remove()
rows = []
for i in (0, 5, 12, 21):
    x, idx, w, out = caps[i]
    ex = bl[i][3]
    alive = torch.ones(ex.num_experts, ex.intermediate_dim, dtype=torch.bool, device="cuda")
    c = C.CompactExperts(ex, alive).to("cuda")
    with torch.no_grad():
        o2 = c(x, idx, w)
    rows.append({"block": i, "tensor_equal": bool(torch.equal(out, o2)), "frac_elements_equal": round(float((out == o2).float().mean()), 5),
                 "max_abs_diff": float((out.float() - o2.float()).abs().max()), "scale": float(out.float().abs().mean())})
    del c
print("RESULT", json.dumps(rows), flush=True)
again = L.predict(tok, m, dev)
first = json.load(open(R.HERE / "data/open/plain_small.json"))
print("RESULT", json.dumps({"plain_run1_vs_run2_identical": round(sum(a == b for a, b in zip(first, again)) / len(first), 4)}), flush=True)
