#!/usr/bin/env python3
"""Teach from annotated mistakes. usage: feedback.py ARTIFACT_DIR ANNOTATIONS.jsonl
Lines: {"doc": "...", "kind": "email", "values": ["correct", "values"]}; at most 25 lines.
Changes only the on/off state of hidden units already in the artifact. No weight value changes; nothing is added."""
import json, sys
from pathlib import Path
import numpy as np, torch
import artifact as A, feedback_core as FC

d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
assert len(rows) <= 25, "at most 25 annotated items"
items = [{"doc": r["doc"], "kind": r["kind"], "gold": list(r["values"]), "id": str(i), "type": "pos" if r["values"] else "neg", "src": "user"} for i, r in enumerate(rows)]
tok, m, cb = A.load(d)
imp = torch.from_numpy(np.load(d / "importance.npy").astype(np.float32)).to("cuda")
flips, log = FC.teach(tok, m, cb, items, imp)
bits = np.concatenate([ex.unit_mask.detach().cpu().numpy() > 0 for _, ex in cb]).astype(np.uint8)
np.packbits(bits).tofile(d / "feedback_state.bin")
print(json.dumps({"annotated": len(items), "units_switched_off": len(flips), "rounds": log}))
