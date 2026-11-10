"""Freeze the exam. Selection depends only on salted hashes and fixed counts, never on any model.
Prints counts and hashes only. Refuses to overwrite an existing lock."""
import gzip, hashlib, json, time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import tasklib as T

ROOT = Path(__file__).parent / "data"
SALT = "ten-percent-v2-exam-2026-11-09"
PER_KIND_A, NEG_A, POS_B, NEG_B = 300, 1500, 1000, 200
LOCKF = Path(__file__).parent / "LOCK.json"
assert not LOCKF.exists(), "already locked"

def order(items):
    return sorted(items, key=lambda it: T.sid(SALT, it["id"]))

rows_a = [json.loads(l) for l in gzip.open(ROOT / "locked/exam_pool_a.jsonl.gz", "rt")]
rows_b = []
for f in sorted((ROOT / "raw").glob("B_*.parquet")):
    rows_b += pq.read_table(f, columns=["text", "entities"]).to_pylist()

by_kind = {}
for it in T.pos_items("A", rows_a, T.spans_a):
    by_kind.setdefault(it["kind"], []).append(it)
kept, dropped, exam = {}, {}, []
for k, its in sorted(by_kind.items()):
    if len(its) < PER_KIND_A:
        dropped[k] = len(its)
        continue
    chosen = order(its)[:PER_KIND_A]
    exam += chosen
    kept[k] = len(chosen)
exam += order(list(T.neg_items("A", rows_a, T.spans_a, kept)))[:NEG_A]
pos_b = order(list(T.pos_items("B", rows_b, T.spans_b)))[:POS_B]
kinds_b = sorted({it["kind"] for it in pos_b})
neg_b = order(list(T.neg_items("B", rows_b, T.spans_b, kinds_b)))[:NEG_B]
exam += pos_b + neg_b
ids = [it["id"] for it in exam]
assert len(ids) == len(set(ids))

path = ROOT / "locked/exam.jsonl"
with open(path, "w", encoding="utf-8") as h:
    for it in sorted(exam, key=lambda x: x["id"]):
        h.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = {"created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "note": "Selected before any model was run on this task. Depends on hashes and fixed counts only.",
        "exam_sha256": sha(path), "exam_items": len(exam),
        "teach_pool_a_sha256": sha(ROOT / "locked/teach_pool_a.jsonl.gz"),
        "exam_pool_a_sha256": sha(ROOT / "locked/exam_pool_a.jsonl.gz"),
        "raw_sha256": {p.name: sha(p) for p in sorted((ROOT / "raw").glob("*.parquet"))},
        "rules": {"salt": SALT, "max_chars": T.MAX_CHARS, "per_kind_a": PER_KIND_A, "neg_a": NEG_A,
                  "pos_b": POS_B, "neg_b": NEG_B},
        "code_sha256": {f: sha(Path(__file__).parent / f) for f in ("split_a.py", "lock.py", "tasklib.py")}}
LOCKF.write_text(json.dumps(lock, indent=1))
print(json.dumps({"exam_items": len(exam), "by_src": Counter(it["src"] for it in exam),
                  "by_type": Counter(it["type"] for it in exam), "kinds_a": len(kept),
                  "kinds_dropped_lt300": dropped, "kinds_b": len(kinds_b),
                  "exam_sha256": lock["exam_sha256"][:24]}, indent=0))
