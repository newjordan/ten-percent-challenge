"""Expert-routed block utilities shared across model families: usage census and expert removal.
Removal uses the router alone: a removed expert can never be selected, so the next-best takes its place."""
import types
import torch
import torch.nn.functional as F
from torch import nn


def blocks(model):
    """[(name, owner, router, experts)] for each routed block."""
    out = []
    for name, mod in model.named_modules():
        ex = getattr(mod, "experts", None)
        if isinstance(ex, nn.Module):
            router = getattr(mod, "gate", None) or getattr(mod, "router", None)
            out.append((name, mod, router, ex))
    return out


def _lfm_router(self, hidden_states, expert_bias=None):
    logits = F.linear(hidden_states, self.weight)
    w = logits.sigmoid()
    scores = (w + expert_bias) if self.use_expert_bias and expert_bias is not None else w
    scores = scores.masked_fill(self.banned, -1e9)
    _, sel = torch.topk(scores, k=self.top_k, dim=-1)
    w = torch.gather(w, 1, sel).type_as(logits)
    if self.norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
    return logits, w * self.routed_scaling_factor, sel


def _granite_router(self, hidden_states):
    logits = F.linear(hidden_states, self.weight).float().masked_fill(self.banned, -1e9)
    top, idx = logits.topk(self.top_k, dim=-1)
    return logits, torch.softmax(top, dim=-1).type_as(hidden_states), idx


def install_ban(model):
    bs = blocks(model)
    for _, owner, router, ex in bs:
        router.banned = torch.zeros(ex.num_experts, dtype=torch.bool, device=ex.down_proj.device)
        fn = _lfm_router if hasattr(router, "routed_scaling_factor") else _granite_router
        router.forward = types.MethodType(fn, router)
    return bs


def census(bs):
    """Accumulate routing mass per expert while the model runs. Returns (mass list, remove-hooks fn)."""
    mass = [torch.zeros(ex.num_experts, dtype=torch.float64, device=ex.down_proj.device) for *_, ex in bs]
    hooks = []
    for m, (_, _, _, ex) in zip(mass, bs):
        def hook(mod, args, m=m):
            idx, w = args[1], args[2]
            m.index_add_(0, idx.reshape(-1), w.reshape(-1).to(m.dtype))
        hooks.append(ex.register_forward_pre_hook(hook))
    return mass, (lambda: [h.remove() for h in hooks])


def keep_top(bs, mass, k):
    for m, (_, _, router, ex) in zip(mass, bs):
        router.banned.fill_(True)
        router.banned[m.topk(k).indices] = False


def clear(bs):
    for _, _, router, _ in bs:
        router.banned.fill_(False)
