"""On/off switches over hidden units. Weight values are never touched.
install() is for the full-size research model; the compact model carries its own 1-D mask."""
import types
import torch
import torch.nn.functional as F
import moe


def _forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit:
        e = e[0]
        if e == self.num_experts:
            continue
        pos, tok = torch.where(mask[e])
        gate, up = F.linear(hidden_states[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
        h = self.act_fn(gate) * up * self.unit_mask[e].to(gate.dtype)
        y = F.linear(h, self.down_proj[e]) * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, y.to(final.dtype))
    return final


def install(model):
    out = []
    for name, _, _, ex in moe.blocks(model):
        ex.unit_mask = torch.ones(ex.num_experts, ex.intermediate_dim, device=ex.down_proj.device, dtype=torch.float32)
        ex.forward = types.MethodType(_forward, ex)
        out.append((name, ex))
    return out


def track(blocks, on):
    for _, ex in blocks:
        ex.unit_mask.requires_grad_(on)
        ex.unit_mask.grad = None


def reset(blocks):
    for _, ex in blocks:
        ex.unit_mask.data.fill_(1.0)


def apply_off(blocks, picks):
    """picks: (block, flat_index) for compact masks, or (block, expert, unit) for full-size masks."""
    for p in picks:
        if len(p) == 2:
            blocks[p[0]][1].unit_mask.data[p[1]] = 0.0
        else:
            blocks[p[0]][1].unit_mask.data[p[1], p[2]] = 0.0
