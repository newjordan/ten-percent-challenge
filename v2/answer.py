#!/usr/bin/env python3
"""Answer extraction requests. usage: answer.py ARTIFACT_DIR INPUT.jsonl OUTPUT.jsonl
Input lines: {"doc": "...", "kind": "email"}. Output lines: {"values": [...], "text": "raw answer"}."""
import json, sys
import artifact as A, fb_lib as L, tasklib as T

d, inp, outp = sys.argv[1:4]
tok, m, _ = A.load(d)
items = [{"doc": r["doc"], "kind": r["kind"], "gold": [], "id": str(i), "type": "pos", "src": "user"} for i, r in enumerate(map(json.loads, open(inp)))]
outs = L.predict(tok, m, items)
with open(outp, "w") as h:
    for o in outs:
        h.write(json.dumps({"values": sorted(T.parse(o)), "text": o}, ensure_ascii=False) + "\n")
