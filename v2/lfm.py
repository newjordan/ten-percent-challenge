"""Load the chosen base with hidden-unit switches and expert removal installed."""
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import moe, units as U

NAME = "LiquidAI/LFM2-8B-A1B"
PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*"]


def load():
    path = snapshot_download(NAME, allow_patterns=PATTERNS)
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    blocks = U.install(m)
    bs = moe.install_ban(m)
    return tok, m, blocks, bs


def apply_alive(blocks, bs, alive, min_frac=0.05):
    """alive: flat bool tensor over all units. Experts with almost nothing left are removed from routing."""
    sizes = [ex.unit_mask.numel() for _, ex in blocks]
    banned = 0
    for (_, ex), (_, _, router, _), a in zip(blocks, bs, torch.split(alive, sizes)):
        am = a.view(ex.num_experts, ex.intermediate_dim)
        ex.unit_mask.data.copy_(am.float())
        router.banned.copy_(am.float().mean(1) < min_frac)
        banned += int(router.banned.sum())
    return banned
