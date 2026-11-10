"""Do compact experts match the default path during cached generation steps (few tokens per expert)?"""
import json, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, fb_lib as L, lfm, compact as C, moe

path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
dev = R.items_from("dev", 20, 100, salt="v2-prune2")
bl = moe.blocks(m)
caps, hooks = {}, []
for i, (_, _, _, ex) in enumerate(bl):
    def post(mod, args, out, i=i):
        if args[0].shape[0] <= 64:
            caps.setdefault(i, []).append(([a.detach().clone() for a in args], out.detach().clone()))
    hooks.append(ex.register_forward_hook(post))
enc = tok([L.chat(tok, it) for it in dev[:16]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
enc.pop("token_type_ids", None)
with torch.no_grad():
    m.generate(**enc, max_new_tokens=6, do_sample=False, temperature=None, top_p=None, top_k=None)
for h in hooks:
    h.remove()
tot, eq, frac, worst = 0, 0, [], 0.0
for i in (3, 9, 15, 21):
    ex = bl[i][3]
    c = C.CompactExperts(ex, torch.ones(ex.num_experts, ex.intermediate_dim, dtype=torch.bool, device="cuda")).to("cuda")
    for args, out in caps.get(i, []):
        with torch.no_grad():
            o2 = c(*args)
        tot += 1; eq += bool(torch.equal(out, o2)); frac.append(float((out == o2).float().mean())); worst = max(worst, float((out.float() - o2.float()).abs().max()))
    del c
print("RESULT", json.dumps({"steps_checked": tot, "steps_exactly_equal": eq, "mean_frac_elements_equal": round(sum(frac) / max(1, len(frac)), 5), "max_abs_diff": worst,
      "tokens_per_step": sorted({a[0].shape[0] for i in (3, 9) for a, _ in caps.get(i, [])})[:6]}), flush=True)
