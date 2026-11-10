"""Freeze a second, fresh exam from documents the first exam did not use, plus a third source nothing has touched.
Selection depends only on salted hashes and fixed counts. Prints counts and hashes only."""
import ast, gzip, hashlib, json, time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import tasklib as T

HERE = Path(__file__).parent; ROOT = HERE / "data"; OUT = ROOT / "locked2"; OUT.mkdir(exist_ok=True)
LOCKF = HERE / "LOCK2.json"; assert not LOCKF.exists(), "already locked"
SALT = "ten-percent-v2-exam2-2026-11-09"
PER_KIND, NEG_A, POS_B, NEG_B, POS_C, NEG_C = 300, 1500, 1000, 200, 1000, 200
order = lambda items: sorted(items, key=lambda it: T.sid(SALT, it["id"]))

used = {json.loads(l)["doc"] for l in open(ROOT / "locked/exam.jsonl", encoding="utf-8")}
rows_a = [r for r in map(json.loads, gzip.open(ROOT / "locked/exam_pool_a.jsonl.gz", "rt", encoding="utf-8")) if r["text"] not in used]
is_teach = lambda text: int(T.sid(SALT, "split", text), 16) % 5 == 0
teach_rows = [r for r in rows_a if is_teach(r["text"])]
exam_rows = [r for r in rows_a if not is_teach(r["text"])]
with gzip.open(OUT / "teach2_rows.jsonl.gz", "wt", encoding="utf-8") as h:
    for r in teach_rows:
        h.write(json.dumps(r, ensure_ascii=False) + "\n")

by_kind = {}
for it in T.pos_items("A", exam_rows, T.spans_a):
    by_kind.setdefault(it["kind"], []).append(it)
kept, dropped, exam = {}, {}, []
for k, v in sorted(by_kind.items()):
    if len(v) < PER_KIND:
        dropped[k] = len(v); continue
    exam += order(v)[:PER_KIND]; kept[k] = PER_KIND
exam += order(list(T.neg_items("A", exam_rows, T.spans_a, kept)))[:NEG_A]

rows_b = []
for f in sorted((ROOT / "raw").glob("B_*.parquet")):
    rows_b += [r for r in pq.read_table(f, columns=["text", "entities"]).to_pylist() if r["text"] not in used]
pb = order(list(T.pos_items("B", rows_b, T.spans_b)))[:POS_B]
exam += pb + order(list(T.neg_items("B", rows_b, T.spans_b, sorted({x["kind"] for x in pb}))))[:NEG_B]


def spans_c(row):
    out, low = {}, row["text"].casefold()
    for s in ast.literal_eval(row["pii_spans"]):
        v = row["text"][s["start"]:s["end"]]
        if T.norm(v) and v.casefold() in low:
            out.setdefault(s["label"], set()).add(v)
    return out


rows_c = []
for f in sorted((ROOT / "raw").glob("C_*.parquet")):
    rows_c += [{"text": r["generated_text"], "pii_spans": r["pii_spans"]} for r in pq.read_table(f, columns=["generated_text", "pii_spans"]).to_pylist()]
pc = order(list(T.pos_items("C", rows_c, spans_c)))[:POS_C]
exam += pc + order(list(T.neg_items("C", rows_c, spans_c, sorted({x["kind"] for x in pc}))))[:NEG_C]
ids = [x["id"] for x in exam]; assert len(ids) == len(set(ids))
assert not ({x["doc"] for x in exam} & used), "overlap with first exam"
with open(OUT / "exam.jsonl", "w", encoding="utf-8") as h:
    for it in sorted(exam, key=lambda x: x["id"]):
        h.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
LOCKF.write_text(json.dumps({"created_utc": time.strftime("%FT%TZ", time.gmtime()), "exam_sha256": sha(OUT / "exam.jsonl"), "exam_items": len(exam), "teach2_sha256": sha(OUT / "teach2_rows.jsonl.gz"),
                             "raw_c_sha256": {p.name: sha(p) for p in sorted((ROOT / "raw").glob("C_*.parquet"))}, "rules": {"salt": SALT, "per_kind": PER_KIND, "neg_a": NEG_A, "pos_b": POS_B, "neg_b": NEG_B, "pos_c": POS_C, "neg_c": NEG_C},
                             "code_sha256": {f: sha(HERE / f) for f in ("lock2.py", "tasklib.py")}}, indent=1))
print(json.dumps({"exam_items": len(exam), "by_src": Counter(x["src"] for x in exam), "by_type": Counter(x["type"] for x in exam), "kinds_a": len(kept), "dropped_lt300": dropped, "teach_docs": len(teach_rows), "exam_docs_a": len(exam_rows)}))
