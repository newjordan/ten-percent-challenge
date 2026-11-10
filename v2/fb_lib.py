"""Attribution of annotated mistakes to existing hidden units, and evaluation helpers."""
import torch
import tasklib as T


def chat(tok, item):
    return tok.apply_chat_template([{"role": "user", "content": T.prompt_of(item)}], add_generation_prompt=True, tokenize=False)


def answer_text(item):
    """Canonical correct answer: values in document order, one per line; NONE when empty."""
    if not item["gold"]:
        return "NONE"
    low = item["doc"].casefold()
    return "\n".join(sorted(item["gold"], key=lambda g: low.find(g.casefold())))


def eos_id(tok, model):
    e = model.generation_config.eos_token_id
    return e[0] if isinstance(e, (list, tuple)) else e


def forced(tok, model, item, answer):
    """Teacher-forced forward over prompt+answer. Returns (logits at answer positions, target ids)."""
    p = tok(chat(tok, item), add_special_tokens=False, return_tensors="pt")["input_ids"]
    a = tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = torch.tensor([[eos_id(tok, model)]])
    ids = torch.cat([p, a, eos], dim=1).to("cuda")
    logits = model(input_ids=ids, use_cache=False).logits[0]
    start = p.shape[1] - 1
    return logits[start:-1], ids[0, start + 1:]


def mistake_loss(logits, targets):
    """Sum over wrong positions of (logit of predicted wrong token - logit of correct token). None if all right."""
    pred = logits.argmax(-1)
    wrong = (pred != targets).nonzero().flatten()
    if len(wrong) == 0:
        return None, 0
    idx = wrong
    return (logits[idx, pred[idx]].float() - logits[idx, targets[idx]].float()).sum(), len(wrong)


def margin_loss(logits, targets):
    """Sum over positions of (logit of the model's own token - best other)."""
    t = logits.float()
    own = t.gather(1, targets[:, None]).squeeze(1)
    other = t.scatter(1, targets[:, None], -1e9).max(-1).values
    return (own - other).sum()


@torch.inference_mode()
def predict(tok, m, items, bs=24, mx=64):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["doc"]))
    out = [None] * len(items)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([chat(tok, items[j]) for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        enc.pop("token_type_ids", None)
        gen = m.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None)
        for j, s in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = s
    return out
