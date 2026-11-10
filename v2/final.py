"""Single-use evaluation of the frozen system on the locked exam.
Stages (run in order, each its own process): base | cand | teach | report. Results are aggregates only.
A stage may be resumed after a crash, but never with a different artifact."""
import gc, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
import tasklib as T

HERE = Path(__file__).parent
LOCKED = HERE / "data/locked"; OUTDIR = LOCKED / "final"; OUTDIR.mkdir(exist_ok=True)
LOCK = json.load(open(HERE / "LOCK.json"))
ARTIFACT = Path(os.environ.get("ARTIFACT", HERE / "data/artifact_final"))
KINDS = ["time", "date_time", "coordinate", "occupation"]
TEACH_N = 300
STAGE = sys.argv[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def artifact_id():
    return hashlib.sha256("".join(sha(p) for p in sorted(ARTIFACT.rglob("*")) if p.is_file() and p.name != "feedback_state.bin").encode()).hexdigest()


def claim():
    """Records the first opening. A later run must name the same artifact."""
    c = LOCKED / "CLAIM.json"
    aid = artifact_id()
    assert sha(LOCKED / "exam.jsonl") == LOCK["exam_sha256"], "exam file changed"
    if c.exists():
        assert json.load(open(c))["artifact"] == aid, "the exam was already opened for a different artifact"
    else:
        with open(c, "x") as h:
            json.dump({"artifact": aid, "first_opened_utc": time.strftime("%FT%TZ", time.gmtime())}, h)


def exam():
    return [json.loads(l) for l in open(LOCKED / "exam.jsonl", encoding="utf-8")]


def done(name):
    return (OUTDIR / f"{name}.json").exists()


def save(name, outs):
    json.dump(outs, open(OUTDIR / f"{name}.json", "w"))


if STAGE == "base":
    claim()
    if not done("base"):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import fb_lib as L, lfm
        path = snapshot_download(lfm.NAME, allow_patterns=lfm.PATTERNS)
        tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
        save("base", L.predict(tok, m, exam()))
    print("base done", flush=True)

elif STAGE == "cand":
    claim()
    if not done("cand"):
        import artifact as A, fb_lib as L
        tok, m, _ = A.load(ARTIFACT)
        save("cand", L.predict(tok, m, exam()))
    print("cand done", flush=True)

elif STAGE == "teach":
    claim()
    import numpy as np
    import artifact as A, feedback_core as FC, fb_lib as L, runner as R, units as U
    tok, m, cb = A.load(ARTIFACT)
    imp = torch.from_numpy(np.load(ARTIFACT / "importance.npy").astype(np.float32)).to("cuda")
    pool_rows = R.open_rows("pool"); by_doc = {r["text"]: r for r in pool_rows}
    pool_by = {}
    for it in T.pos_items("A", pool_rows, T.spans_a):
        pool_by.setdefault(it["kind"], []).append(it)
    teach_rows = [json.loads(l) for l in gzip.open(LOCKED / "teach_pool_a.jsonl.gz", "rt", encoding="utf-8")]
    teach_by = {}
    for it in T.pos_items("A", teach_rows, T.spans_a):
        teach_by.setdefault(it["kind"], []).append(it)
    ex = exam()
    for kind in KINDS:
        if done(f"teach_{kind}"):
            continue
        U.reset(cb)
        cand = sorted(pool_by[kind], key=lambda x: T.sid("fb", x["id"]))[:300]
        co = L.predict(tok, m, cand)
        fb = [c for c, o in zip(cand, co) if not T.correct(c, o)][:25]
        te = sorted(teach_by[kind], key=lambda x: T.sid("teach-final", x["id"]))[:TEACH_N]
        res = {"kind": kind, "n_annotated": len(fb), "n_teach": len(te)}
        run = lambda items: [bool(T.correct(i, o)) for i, o in zip(items, L.predict(tok, m, items))]
        res["empty_teach"] = run(te)
        golds = [x["gold"] for x in fb]
        wv = []
        for x in fb:
            alts = sorted((v for k, v in T.spans_a(by_doc[x["doc"]]).items() if k != kind), key=str)
            wv.append(sorted(alts[0]) if alts else x["gold"])
        for name, labels in (("shuffled", golds[1:] + golds[:1]), ("wrong_value", wv)):
            U.reset(cb)
            flips, _ = FC.teach(tok, m, cb, [{**x, "gold": g} for x, g in zip(fb, labels)], imp)
            res[f"{name}_flips"] = len(flips); res[f"{name}_teach"] = run(te)
        U.reset(cb)
        flips, log = FC.teach(tok, m, cb, [{**x, "gold": g} for x, g in zip(fb, golds)], imp)
        res["real_flips"] = len(flips); res["real_log"] = log; res["real_teach"] = run(te)
        res["exam_after_feedback"] = L.predict(tok, m, ex)
        save(f"teach_{kind}", res)
        print("teach done", kind, flush=True)

elif STAGE == "report":
    ex = exam()
    base = json.load(open(OUTDIR / "base.json")); cand = json.load(open(OUTDIR / "cand.json"))
    b_ok = [T.correct(i, o) for i, o in zip(ex, base)]
    c_ok = [T.correct(i, o) for i, o in zip(ex, cand)]

    def group(it):
        return it["kind"] if it["type"] == "pos" and it["src"] == "A" else ("<absent>" if it["type"] == "neg" else f"<source-{it['src']}>")

    def faith(ok):
        groups = {}
        for it, b, c in zip(ex, b_ok, ok):
            if b:
                g = groups.setdefault(group(it), [0, 0]); g[0] += c; g[1] += 1
        tot = sum(g[0] for g in groups.values()) / sum(g[1] for g in groups.values())
        rates = {k: g[0] / g[1] for k, g in groups.items()}
        return tot, rates, {k: g[1] for k, g in groups.items()}

    tot, rates, ns = faith(c_ok)
    out = {"artifact_bytes": sum(p.stat().st_size for p in ARTIFACT.rglob("*") if p.is_file()), "n": len(ex),
           "base_acc": sum(b_ok) / len(ex), "cand_acc": sum(c_ok) / len(ex), "faithful_overall": tot,
           "kinds_reported": len(rates), "worst_kinds": sorted((round(v, 4), k, ns[k]) for k, v in rates.items())[:8],
           "kinds_below_95": sorted(k for k, v in rates.items() if v < 0.95), "stages": {}}
    for kind in KINDS:
        if not done(f"teach_{kind}"):
            continue
        r = json.load(open(OUTDIR / f"teach_{kind}.json"))
        a = [T.correct(i, o) for i, o in zip(ex, r["exam_after_feedback"])]
        t2, rt, _ = faith(a)
        out["stages"][kind] = {"n_annotated": r["n_annotated"], "n_teach": r["n_teach"], "units_off": r["real_flips"],
                               "teach_empty": sum(r["empty_teach"]) / r["n_teach"], "teach_real": sum(r["real_teach"]) / r["n_teach"],
                               "teach_shuffled": sum(r["shuffled_teach"]) / r["n_teach"], "teach_wrong_value": sum(r["wrong_value_teach"]) / r["n_teach"],
                               "exam_acc_after": sum(a) / len(ex), "faithful_after": t2, "kinds_below_95_after": sorted(k for k, v in rt.items() if v < 0.95)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(HERE / "data/open/final_report.json", "w"), indent=1)
