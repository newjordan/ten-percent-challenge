import json, time, torch
import runner as R, tasklib as T
items = R.items_from("dev", 80, 400)
sub = sorted(items, key=lambda x: T.sid("proto", x["id"]))[:240]
tok, m = R.load()
BASE = R.NO_THINK
VARIANTS = {"empty_think": ("<think>\n\n</think>\n\n", 64, None), "tight_think": ("<think>\n</think>\n", 64, None),
            "canned_think": ("<think>\nThe values are in the document, so I will copy them out exactly.\n</think>\n\n", 64, None),
            "system_direct": ("", 64, "Answer directly with only the requested output. Do not reason, do not explain."),
            "natural_think": ("", 400, None)}
def text(item, pre, sys_msg):
    msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": T.prompt_of(item)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + pre
@torch.inference_mode()
def run(pre, mx, sys_msg, bs=24):
    order = sorted(range(len(sub)), key=lambda i: len(sub[i]["doc"])); out = [None] * len(sub); lens = []
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([text(sub[j], pre, sys_msg) for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = m.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None)
        new = gen[:, enc["input_ids"].shape[1]:]
        lens += (new != tok.pad_token_id).sum(1).tolist()
        for j, s in zip(idx, tok.batch_decode(new, skip_special_tokens=False)): out[j] = s
    return out, lens
def clean(o, natural):
    o = o.replace("<|im_end|>", "\x00").split("\x00")[0].replace("<|pad|>", "")
    if natural:
        return o.split("</think>")[-1] if "</think>" in o else ""
    return o.split("</think>")[0]
res = {}
for name, (pre, mx, sm) in VARIANTS.items():
    t = time.time(); outs, lens = run(pre, mx, sm); dt = time.time() - t
    nat = name == "natural_think"
    a = [T.correct(it, clean(o, nat)) for it, o in zip(sub, outs)]
    b = [T.correct(it, clean(o, nat).strip().split("\n\n")[0]) for it, o in zip(sub, outs)]
    pos = [i for i, it in enumerate(sub) if it["type"] == "pos"]; neg = [i for i, it in enumerate(sub) if it["type"] == "neg"]
    res[name] = {"acc_strict": round(sum(a) / len(a), 3), "acc_first_block": round(sum(b) / len(b), 3), "pos_block": round(sum(b[i] for i in pos) / len(pos), 3),
                 "neg_block": round(sum(b[i] for i in neg) / max(1, len(neg)), 3), "n_neg": len(neg), "secs": round(dt), "mean_new_toks": round(sum(lens) / len(lens), 1), "max_new": max(lens)}
    print(name, json.dumps(res[name]), flush=True)
    print("  sample", json.dumps([clean(o, nat)[:90] for o in outs[:3]], ensure_ascii=False), flush=True)
json.dump(res, open("data/open/proto.json", "w"), indent=1)
print("DONE", flush=True)
