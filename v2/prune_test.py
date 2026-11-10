"""How many experts can go before answers change? Full-precision weights; open data only."""
import gc, json, sys, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T, moe

SPEC = {"LiquidAI/LFM2-8B-A1B": (8, 12, 16, 24)}
calib = R.items_from("calib", 10, 50)
dev = R.items_from("dev", 10, 50, salt="v2-prune")

@torch.inference_mode()
def run(tok, m, items, bs=24, mx=64):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["doc"]))
    out = [None] * len(items)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        p = [tok.apply_chat_template([{"role": "user", "content": T.prompt_of(items[j])}], add_generation_prompt=True, tokenize=False) for j in idx]
        enc = tok(p, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        enc.pop("token_type_ids", None)
        gen = m.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None)
        for j, s in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = s
    return out

for name, ks in SPEC.items():
    path = snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    bs = moe.install_ban(m)
    nparam = sum(p.numel() for p in m.parameters())
    exp_param = sum(p.numel() for *_, ex in bs for p in ex.parameters())
    print(json.dumps({"model": name, "blocks": len(bs), "experts_per_block": bs[0][3].num_experts, "expert_share": round(exp_param / nparam, 4), "params_b": round(nparam / 1e9, 3)}), flush=True)
    t0 = time.time()
    mass, stop = moe.census(bs)
    run(tok, m, calib)
    stop()
    print(json.dumps({"census_s": round(time.time() - t0)}), flush=True)
    for frac in (0.25, 0.375, 0.5, 0.75):
        k = int(round(frac * bs[0][3].num_experts))
        cov = sum(float(mm.topk(k).values.sum() / mm.sum()) for mm in mass) / len(mass)
        print(json.dumps({"k": k, "mean_routing_mass_covered": round(cov, 4)}), flush=True)
    base = run(tok, m, dev)
    base_ok = [T.correct(it, o) for it, o in zip(dev, base)]
    print("RESULT", json.dumps({"model": name, "k": "all", "acc": round(sum(base_ok) / len(dev), 4), "n": len(dev)}), flush=True)
    rows = []
    for k in ks:
        t1 = time.time()
        moe.keep_top(bs, mass, k)
        o = run(tok, m, dev)
        ok = [T.correct(it, x) for it, x in zip(dev, o)]
        kept = sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok)
        same = sum(x == y for x, y in zip(o, base)) / len(dev)
        rows.append({"model": name, "k": k, "of": bs[0][3].num_experts, "acc": round(sum(ok) / len(dev), 4), "faithful_on_base_correct": round(kept, 4),
                     "identical_output": round(same, 4), "params_left_share": round(1 - exp_param / nparam * (1 - k / bs[0][3].num_experts), 4), "secs": round(time.time() - t1)})
        print("RESULT", json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"data/open/prune_{name.split('/')[1]}.json", "w"), indent=1)
    del m; gc.collect(); torch.cuda.empty_cache()
print("ALLDONE", flush=True)
