"""On-disk format and standalone loader. Loading needs only the files in the artifact directory."""
import gzip, json, os, shutil, tempfile
from pathlib import Path
import constriction
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import compact as C, moe


class Writer:
    def __init__(self, directory):
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "weights.bin", "wb")
        self.index = {"tensors": {}}

    def _blob(self, arr):
        off = self.f.tell(); self.f.write(arr.tobytes()); return off, arr.size

    def quantized(self, name, codes, step, bits):
        arr = (codes.long() + 2 ** (bits - 1)).flatten().cpu().numpy().astype(np.int32)
        counts = np.bincount(arr, minlength=2 ** bits).astype(np.uint32)
        enc = constriction.stream.queue.RangeEncoder()
        enc.encode(arr, constriction.stream.model.Categorical((counts + 1e-30) / (counts + 1e-30).sum(), perfect=False))
        c = enc.get_compressed()
        co, cn = self._blob(np.asarray(c, dtype=np.uint32))
        so, sn = self._blob(step.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy())
        self.index["tensors"][name] = {"kind": "q", "shape": list(codes.shape), "bits": bits, "counts": counts.tolist(), "c_off": co, "c_n": cn, "s_off": so, "s_n": sn}

    def raw(self, name, t):
        a = t.detach().float().cpu().numpy().astype(np.float32)
        off, n = self._blob(a.ravel())
        self.index["tensors"][name] = {"kind": "raw", "shape": list(t.shape), "off": off, "n": n, "dtype": str(t.dtype)}

    def close(self, extra):
        self.f.close()
        self.index.update(extra)
        with gzip.open(self.dir / "index.json.gz", "wt", encoding="utf-8", compresslevel=9) as h:
            json.dump(self.index, h, separators=(",", ":"))


class Reader:
    def __init__(self, directory):
        self.dir = Path(directory)
        with gzip.open(self.dir / "index.json.gz", "rt", encoding="utf-8") as h:
            self.index = json.load(h)
        self.mm = np.memmap(self.dir / "weights.bin", mode="r", dtype=np.uint8)

    def _u32(self, off, n):
        return np.frombuffer(self.mm, dtype=np.uint32, count=n, offset=off)

    def tensor(self, name, device, dtype=torch.bfloat16):
        m = self.index["tensors"][name]
        if m["kind"] == "raw":
            a = np.frombuffer(self.mm, dtype=np.float32, count=m["n"], offset=m["off"]).reshape(m["shape"])
            return torch.from_numpy(a.copy()).to(device).to(dtype if m["dtype"] != "torch.float32" else torch.float32)
        counts = np.asarray(m["counts"], dtype=np.float64)
        dec = constriction.stream.queue.RangeDecoder(self._u32(m["c_off"], m["c_n"]).copy())
        arr = dec.decode(constriction.stream.model.Categorical((counts + 1e-30) / (counts + 1e-30).sum(), perfect=False), int(np.prod(m["shape"])))
        s = torch.from_numpy(np.frombuffer(self.mm, dtype=np.uint16, count=m["s_n"], offset=m["s_off"]).copy()).view(torch.bfloat16).float().to(device)
        codes = torch.from_numpy(arr.astype(np.int16)).to(device).reshape(m["shape"]) - 2 ** (m["bits"] - 1)
        return (codes.float() * s.reshape(-1, *[1] * (codes.ndim - 1))).to(dtype)


def load(directory, device="cuda"):
    """Rebuild the model from the artifact alone. Returns (tokenizer, model, [(name, CompactExperts)])."""
    d = Path(directory)
    tmp = Path(tempfile.mkdtemp(prefix="artifact-tok-"))
    for f in os.listdir(d / "tok"):
        target = tmp / f.removesuffix(".gz")
        if not target.exists():
            with gzip.open(d / "tok" / f, "rb") as a, open(target, "wb") as b:
                shutil.copyfileobj(a, b)
    tok = AutoTokenizer.from_pretrained(tmp); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(tmp)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    r = Reader(d)
    keep = r.index["keep"]
    out, name_map = [], {}
    for name, owner, router, ex in moe.blocks(m):
        n_exp, hdim, idim = ex.num_experts, ex.hidden_dim, ex.intermediate_dim
        lists = [np.flatnonzero(np.unpackbits(np.frombuffer(bytes.fromhex(keep[name][e]), dtype=np.uint8))[:idim]).tolist() if keep[name][e] else [] for e in range(n_exp)]
        c = C.CompactExperts.empty(n_exp, hdim, idim, ex.act_fn, lists, device)
        for e in range(n_exp):
            if lists[e]:
                c.register_buffer(f"gu{e}", r.tensor(f"{name}.gu{e}", device))
                c.register_buffer(f"d{e}", r.tensor(f"{name}.d{e}", device))
        owner.experts = c
        out.append((name, c))
    for pname, _ in list(m.named_parameters()) + list(m.named_buffers()):
        base = pname
        if base in r.index["tensors"]:
            t = r.tensor(base, device, torch.bfloat16)
            set_module_tensor_to_device(m, pname, device, value=t)
    m.tie_weights()
    m.generation_config = GenerationConfig.from_pretrained(tmp)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    bs = moe.install_ban(m)
    for (nm, _, router, _), (_, c) in zip(bs, out):
        router.banned.copy_(torch.tensor([len(c.keep[e]) == 0 for e in range(c.num_experts)], device=device))
    state = d / "feedback_state.bin" if (d / "feedback_state.bin").exists() else d / "repair_state.bin"
    if state.exists():
        bits = np.unpackbits(np.fromfile(state, dtype=np.uint8))
        off = 0
        for _, c in out:
            n = c.unit_mask.numel()
            c.unit_mask.data.copy_(torch.from_numpy(bits[off:off + n].astype(np.float32)).to(device)); off += n
    return tok, m, out


def artifact_size(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
