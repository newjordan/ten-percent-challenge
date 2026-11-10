"""Compare direct-answering expert-routed bases on the same open development subset."""
import gc, json, os, time, traceback
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import runner as R, tasklib as T

items = R.items_from("dev", 80, 400)
sub = sorted(items, key=lambda x: T.sid("proto", x["id"]))[:240]
CANDS = ["ibm-granite/granite-4.0-h-tiny", "allenai/OLMoE-1B-7B-0125-Instruct", "LiquidAI/LFM2-8B-A1B"]

@torch.inference_mode()
def run(tok, m, bs=24, mx=64):
    order = sorted(range(len(sub)), key=lambda i: len(sub[i]["doc"]))
    out, lens = [None] * len(sub), []
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        prompts = [tok.apply_chat_template([{"role": "user", "content": T.prompt_of(sub[j])}], add_generation_prompt=True,
                                           tokenize=False) for j in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        enc.pop("token_type_ids", None)
        gen = m.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None)
        new = gen[:, enc["input_ids"].shape[1]:]
        lens += (new != tok.pad_token_id).sum(1).tolist()
        for j, s in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
            out[j] = s
    return out, lens

for name in CANDS:
    try:
        t0 = time.time()
        path = snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
        print(json.dumps({"model": name, "download_s": round(time.time() - t0)}), flush=True)
        tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        c = m.config.to_dict()
        print(json.dumps({"model": name, "cfg": {k: c.get(k) for k in ("hidden_size", "num_hidden_layers", "num_experts", "num_experts_per_tok", "num_local_experts",
              "moe_intermediate_size", "intermediate_size", "vocab_size", "layer_types")}, "params_b": round(sum(p.numel() for p in m.parameters()) / 1e9, 3),
              "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})[:700], flush=True)
        t1 = time.time(); outs, lens = run(tok, m); dt = time.time() - t1
        ok = [T.correct(it, o) for it, o in zip(sub, outs)]
        pos = [c for c, it in zip(ok, sub) if it["type"] == "pos"]; neg = [c for c, it in zip(ok, sub) if it["type"] == "neg"]
        print("RESULT", json.dumps({"model": name, "acc": round(sum(ok) / len(ok), 3), "pos": round(sum(pos) / len(pos), 3), "neg": round(sum(neg) / len(neg), 3),
              "secs": round(dt), "mean_new_toks": round(sum(lens) / len(lens), 1), "max_new": max(lens),
              "sample": [o[:70] for o in outs[:3]]}, ensure_ascii=False), flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print("FAILED", name, repr(e)[:300], flush=True); traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
print("ALLDONE", flush=True)
