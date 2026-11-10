"""Open items used for build-time repair. Excludes the documents reserved for feedback annotations."""
import json
import runner as R, tasklib as T

KINDS = ["time", "date_time", "coordinate", "occupation"]


def reserved_docs():
    by = {}
    for it in T.pos_items("A", R.open_rows("pool"), T.spans_a):
        if it["kind"] in KINDS:
            by.setdefault(it["kind"], []).append(it)
    return {x["doc"] for k in KINDS for x in sorted(by[k], key=lambda x: T.sid("fb", x["id"]))[:300]}


def items(per_kind=400, neg=2000):
    res = reserved_docs()
    rows = [r for r in R.open_rows("calib") + R.open_rows("pool") if r["text"] not in res]
    by = {}
    for it in T.pos_items("A", rows, T.spans_a):
        by.setdefault(it["kind"], []).append(it)
    out = []
    for k, v in sorted(by.items()):
        out += sorted(v, key=lambda x: T.sid("repair", x["id"]))[:per_kind]
    out += sorted(T.neg_items("A", rows, T.spans_a, by), key=lambda x: T.sid("repair", x["id"]))[:neg]
    return out
