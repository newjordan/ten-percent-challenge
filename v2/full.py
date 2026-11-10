"""Expand the artifact into the library's own expert class and run its untouched default forward path.
Removed units become zero columns. Only the router's expert ban is patched. Shipped bytes are the same artifact."""
import gzip, os, shutil, tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import artifact as A, moe


def load_full(directory, device="cuda"):
    d = Path(directory)
    tmp = Path(tempfile.mkdtemp(prefix="artifact-tok-"))
    for f in os.listdir(d / "tok"):
        with gzip.open(d / "tok" / f, "rb") as a, open(tmp / f.removesuffix(".gz"), "wb") as b:
            shutil.copyfileobj(a, b)
    tok = AutoTokenizer.from_pretrained(tmp); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(tmp)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    r = A.Reader(d)
    keep = r.index["keep"]
    state = d / "feedback_state.bin" if (d / "feedback_state.bin").exists() else d / "repair_state.bin"
    bits = np.unpackbits(np.fromfile(state, dtype=np.uint8)) if state.exists() else None
    off, banned_lists = 0, []
    for name, owner, router, ex in moe.blocks(m):
        n, hdim, idim = ex.num_experts, ex.hidden_dim, ex.intermediate_dim
        gu = torch.zeros(n, 2 * idim, hdim, dtype=torch.bfloat16, device=device)
        dn = torch.zeros(n, hdim, idim, dtype=torch.bfloat16, device=device)
        ban = []
        for e in range(n):
            if not keep[name][e]:
                ban.append(True); continue
            k = np.flatnonzero(np.unpackbits(np.frombuffer(bytes.fromhex(keep[name][e]), dtype=np.uint8))[:idim])
            ban.append(False)
            kt = torch.from_numpy(k).to(device)
            gu[e, torch.cat([kt, kt + idim])] = r.tensor(f"{name}.gu{e}", device)
            dd = r.tensor(f"{name}.d{e}", device)
            if bits is not None:
                dd = dd * torch.from_numpy(bits[off:off + len(k)].astype(np.float32)).to(device).to(dd.dtype)
            dn[e, :, kt] = dd
            off += len(k)
        ex.gate_up_proj = nn.Parameter(gu, requires_grad=False)
        ex.down_proj = nn.Parameter(dn, requires_grad=False)
        banned_lists.append(ban)
    for pname, _ in list(m.named_parameters()) + list(m.named_buffers()):
        if pname in r.index["tensors"]:
            set_module_tensor_to_device(m, pname, device, value=r.tensor(pname, device, torch.bfloat16))
    m.tie_weights()
    m.generation_config = GenerationConfig.from_pretrained(tmp)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    bs = moe.install_ban(m)
    for (_, _, router, _), ban in zip(bs, banned_lists):
        router.banned.copy_(torch.tensor(ban, device=device))
    return tok, m
