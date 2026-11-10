import json, time, torch
t0 = time.time()
import runner as R, tasklib as T
items = R.items_from("dev", 80, 400)
print("items", len(items), round(time.time() - t0, 1), flush=True)
t1 = time.time(); tok, m = R.load()
print("load_s", round(time.time() - t1, 1), "gpu_gb", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
for bs in (8, 24):
    sub = items[::len(items) // bs][:bs]
    t2 = time.time(); outs = R.predict(tok, m, sub, bs=bs)
    print("bs", bs, "gen_s", round(time.time() - t2, 1), "max_gpu", round(torch.cuda.max_memory_allocated() / 1e9, 2),
          "new_chars", sum(len(o) for o in outs), flush=True)
print(json.dumps([{"kind": it["kind"], "gold": it["gold"][:3], "out": o[:100], "ok": T.correct(it, o)} for it, o in zip(sub, outs)][:8], ensure_ascii=False))
print(tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False), m.generation_config.eos_token_id)
