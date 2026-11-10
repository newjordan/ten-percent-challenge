"""Step 1: assign every source-A record to a bucket by salted hash of its text alone.
Only OPEN buckets may ever be inspected. Locked buckets are written without printing."""
import gzip, hashlib, json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

SALT = b"ten-percent-v2-split-2026-11-09"
ROOT = Path(__file__).parent / "data"
(ROOT / "open").mkdir(parents=True, exist_ok=True)
(ROOT / "locked").mkdir(parents=True, exist_ok=True)

def bucket(text: str) -> int:
    return int.from_bytes(hashlib.sha256(SALT + text.encode()).digest()[:8], "big") % 100

def name_of(b: int) -> tuple[str, str]:
    if b < 20: return "locked", "exam_pool_a"
    if b < 25: return "locked", "teach_pool_a"
    if b < 35: return "open", "dev"
    if b < 45: return "open", "calib"
    return "open", "pool"

out = {}
def sink(zone, nm):
    k = (zone, nm)
    if k not in out:
        out[k] = gzip.open(ROOT / zone / f"{nm}.jsonl.gz", "wt", encoding="utf-8", compresslevel=6)
    return out[k]

counts, seen = Counter(), set()
for f in sorted((ROOT / "raw").glob("A_*.parquet")):
    t = pq.read_table(f, columns=["uid", "domain", "document_type", "locale", "text", "spans"]).to_pylist()
    for r in t:
        if r["text"] in seen:
            counts["dup_dropped"] += 1
            continue
        seen.add(r["text"])
        zone, nm = name_of(bucket(r["text"]))
        sink(zone, nm).write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[nm] += 1
for h in out.values():
    h.close()
print(json.dumps(counts, indent=0))
