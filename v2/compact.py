"""The model in its real, smaller form: each expert stores only its surviving hidden units."""
import gc
import torch
import torch.nn.functional as F
from torch import nn
import moe


class CompactExperts(nn.Module):
    def __init__(self, ex, alive, min_frac=0.05):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = ex.num_experts, ex.hidden_dim, ex.intermediate_dim
        self.act_fn = ex.act_fn
        self.keep, self.offset = {}, [0]
        for e in range(ex.num_experts):
            k = alive[e].nonzero().flatten() if alive[e].float().mean() >= min_frac else torch.empty(0, dtype=torch.long, device=alive.device)
            self.keep[e] = k.cpu()
            if len(k):
                idx2 = torch.cat([k, k + ex.intermediate_dim])
                self.register_buffer(f"gu{e}", ex.gate_up_proj[e][idx2].clone())
                self.register_buffer(f"d{e}", ex.down_proj[e][:, k].clone())
            self.offset.append(self.offset[-1] + len(k))
        self.unit_mask = torch.ones(self.offset[-1], device=alive.device, dtype=torch.float32)
        self.collect = False
        self.Hg, self.Hd = {}, {}

    @classmethod
    def empty(cls, num_experts, hidden_dim, intermediate_dim, act_fn, keep_lists, device):
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.num_experts, self.hidden_dim, self.intermediate_dim = num_experts, hidden_dim, intermediate_dim
        self.act_fn = act_fn
        self.keep, self.offset = {}, [0]
        for e in range(num_experts):
            self.keep[e] = torch.as_tensor(keep_lists[e], dtype=torch.long)
            self.offset.append(self.offset[-1] + len(self.keep[e]))
        self.unit_mask = torch.ones(self.offset[-1], device=device, dtype=torch.float32)
        self.collect, self.Hg, self.Hd = False, {}, {}
        return self

    @property
    def down_proj(self):
        return next(self.buffers())

    def alive_experts(self):
        return [e for e in range(self.num_experts) if len(self.keep[e])]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        n_tok, top_k, dim = hidden_states.size(0), top_k_index.size(-1), hidden_states.size(-1)
        slots = torch.zeros(n_tok * top_k, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for eh in hit:
            e = int(eh[0])
            if e == self.num_experts or len(self.keep[e]) == 0:
                continue
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            gate, up = F.linear(x, getattr(self, f"gu{e}")).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            if self.collect:
                with torch.no_grad():
                    xf, hf = x.float(), h.float()
                    self.Hg[e] = self.Hg.get(e, 0) + xf.T @ xf
                    self.Hd[e] = self.Hd.get(e, 0) + hf.T @ hf
            h = h * self.unit_mask[self.offset[e]:self.offset[e + 1]].to(h.dtype)
            y = F.linear(h, getattr(self, f"d{e}")) * top_k_weights[tok, pos, None]
            slots[tok * top_k + pos] = y.to(slots.dtype)
        # Same reduction as the library's default path: sum in 32-bit, then round once.
        return slots.view(n_tok, top_k, dim).sum(dim=1).to(hidden_states.dtype)


def compactify(model, alive_flat):
    """Replace every expert block with its compact form. Returns [(name, compact_experts)]."""
    out, off = [], 0
    for name, owner, router, ex in moe.blocks(model):
        n = ex.num_experts * ex.intermediate_dim
        am = alive_flat[off:off + n].view(ex.num_experts, ex.intermediate_dim)
        off += n
        c = CompactExperts(ex, am).to(ex.down_proj.device)
        router.banned.copy_(torch.tensor([len(c.keep[e]) == 0 for e in range(c.num_experts)], device=router.banned.device))
        owner.experts = c
        ex.gate_up_proj = None
        ex.down_proj = None
        torch.cuda.empty_cache()
        out.append((name, c))
    gc.collect(); torch.cuda.empty_cache()
    return out


@torch.no_grad()
def gptq(W, H, bits, block=128, damp=0.01):
    """Closed-form Hessian-aware rounding with one step per row. Returns (reconstruction, integer codes, step per row)."""
    W = W.float().clone()
    rows, cols = W.shape
    s = (W.abs().amax(1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 0.5)).squeeze(1).to(torch.bfloat16).float()
    H = H.double() + (H.diag().mean().double() * damp) * torch.eye(cols, device=H.device, dtype=torch.float64)
    Hinv = torch.linalg.inv(H).float()
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]
            q = (w / s).round().clamp(lo, hi) * s
            Q1[:, j] = q
            err = (w - q) / Hi[j, j].clamp_min(1e-10)
            if j + 1 < i2 - i1:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hi[j, j + 1:].unsqueeze(0)
            Err[:, j] = err
        W[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    codes = (W / s.unsqueeze(1)).round().to(torch.int16)
    return (codes.float() * s.unsqueeze(1)).to(torch.bfloat16), codes, s.to(torch.bfloat16)


def entropy_bits(codes, bits):
    h = torch.bincount((codes + 2 ** (bits - 1)).flatten().long(), minlength=2 ** bits)
    p = h[h > 0].float() / h.sum()
    return float(-(p * p.log2()).sum()) * codes.numel()
