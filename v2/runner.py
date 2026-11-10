"""Model loading, batched greedy generation, and development-item construction. Reads open data only."""
import glob, gzip, json, time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tasklib as T

HERE = Path(__file__).parent
MODEL_DIR = sorted(glob.glob("/work/cache/hf/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/*"))[-1]
DEVICE = "cuda"
NO_THINK = "<think>\n\n</think>\n\n"

def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, low_cpu_mem_usage=True).to(DEVICE).eval()
    return tok, m

def text_of(tok, item, answer_prefix=""):
    s = tok.apply_chat_template([{"role": "user", "content": T.prompt_of(item)}],
                                add_generation_prompt=True, tokenize=False)
    # One fixed protocol for every model: direct answer, no reasoning preamble.
    return s + NO_THINK + answer_prefix

@torch.inference_mode()
def predict(tok, m, items, bs=24, max_new=64, log=None):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["doc"]))
    out, t0 = [None] * len(items), time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([text_of(tok, items[j]) for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        gen = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, s in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
            out[j] = s
        if log and (b // bs) % log == 0:
            print(json.dumps({"done": b + len(idx), "of": len(order), "secs": round(time.time() - t0)}), flush=True)
    return out

def open_rows(name):
    assert name in ("dev", "calib", "pool"), "open buckets only"
    return [json.loads(l) for l in gzip.open(HERE / f"data/open/{name}.jsonl.gz", "rt")]

def items_from(name, per_kind, neg, salt="v2-open"):
    rows = open_rows(name)
    by = defaultdict(list)
    for it in T.pos_items("A", rows, T.spans_a):
        by[it["kind"]].append(it)
    out = []
    for k, v in sorted(by.items()):
        out += sorted(v, key=lambda x: T.sid(salt, x["id"]))[:per_kind]
    out += sorted(T.neg_items("A", rows, T.spans_a, by), key=lambda x: T.sid(salt, x["id"]))[:neg]
    return out

def report(items, outs):
    ok = [T.correct(it, o) for it, o in zip(items, outs)]
    per = defaultdict(lambda: [0, 0])
    for it, c in zip(items, ok):
        k = it["kind"] if it["type"] == "pos" else "<absent>"
        per[k][0] += c; per[k][1] += 1
    return ok, {"acc": round(sum(ok) / len(ok), 4), "n": len(ok),
                "per_kind": {k: round(a / n, 3) for k, (a, n) in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])}}
