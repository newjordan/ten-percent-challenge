"""Single-use evaluation of the frozen system on the second locked exam.
Stages (run in order, each its own process): base | cand | full | teach | report. Results are aggregates only.
A stage may be resumed after a crash, but never with a different artifact."""
import gzip, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
import tasklib as T

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
LOCKED = HERE / "data/locked2"; OUTDIR = LOCKED / "final"; OUTDIR.mkdir(exist_ok=True)
LOCK = json.load(open(HERE / "LOCK2.json"))
ARTIFACT = Path(os.environ.get("ARTIFACT", HERE / "data/artifact_r1"))
KINDS = ["time", "date_time", "coordinate", "occupation"]
TEACH_N, MIN_TEACH = 300, 25
STAGE = sys.argv[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def artifact_files():
    return [p for p in sorted(ARTIFACT.rglob("*")) if p.is_file() and p.name != "feedback_state.bin" and "__pycache__" not in p.parts]


def artifact_id():
    return hashlib.sha256("".join(sha(p) for p in artifact_files()).encode()).hexdigest()


def claim():
    """Records the first opening. A later run must name the same artifact."""
    c = LOCKED / "CLAIM.json"
    aid = artifact_id()
    assert sha(LOCKED / "exam.jsonl") == LOCK["exam_sha256"], "exam file changed"
    assert aid == json.load(open(HERE / "PROTOCOL2_LOCK.json"))["artifact_sha256"], "artifact is not the frozen one"
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


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - w) / d, 4), round((c + w) / d, 4)]


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

elif STAGE == "full":
    claim()
    if not done("full"):
        import full as F, fb_lib as L
        tok, m = F.load_full(ARTIFACT)
        save("full", L.predict(tok, m, exam()))
    print("full done", flush=True)

elif STAGE == "teach":
    claim()
    import artifact as A, feedback_core as FC, fb_lib as L, runner as R
    tok, m, cb = A.load(ARTIFACT)
    start = [ex.unit_mask.detach().clone() for _, ex in cb]

    def restore():
        for (_, ex), s in zip(cb, start):
            ex.unit_mask.data.copy_(s)

    imp = torch.from_numpy(np.load(ARTIFACT / "importance.npy").astype(np.float32)).to("cuda")
    pool_rows = R.open_rows("pool"); by_doc = {r["text"]: r for r in pool_rows}
    pool_by = {}
    for it in T.pos_items("A", pool_rows, T.spans_a):
        pool_by.setdefault(it["kind"], []).append(it)
    assert sha(LOCKED / "teach2_rows.jsonl.gz") == LOCK["teach2_sha256"], "teaching file changed"
    teach_rows = [json.loads(l) for l in gzip.open(LOCKED / "teach2_rows.jsonl.gz", "rt", encoding="utf-8")]
    teach_by = {}
    for it in T.pos_items("A", teach_rows, T.spans_a):
        teach_by.setdefault(it["kind"], []).append(it)
    ex = exam()
    for kind in KINDS:
        if done(f"teach_{kind}"):
            continue
        restore()
        te = sorted(teach_by.get(kind, []), key=lambda x: T.sid("teach-final2", x["id"]))[:TEACH_N]
        if len(te) < MIN_TEACH:
            save(f"teach_{kind}", {"kind": kind, "n_teach": len(te), "skipped": True})
            print("teach skipped", kind, flush=True)
            continue
        cand = sorted(pool_by[kind], key=lambda x: T.sid("fb", x["id"]))[:300]
        co = L.predict(tok, m, cand)
        fb = [c for c, o in zip(cand, co) if not T.correct(c, o)][:25]
        res = {"kind": kind, "n_annotated": len(fb), "n_teach": len(te)}
        run = lambda items: [bool(T.correct(i, o)) for i, o in zip(items, L.predict(tok, m, items))]
        res["empty_teach"] = run(te)
        golds = [x["gold"] for x in fb]
        wv = []
        for x in fb:
            alts = sorted((v for k, v in T.spans_a(by_doc[x["doc"]]).items() if k != kind), key=str)
            wv.append(sorted(alts[0]) if alts else x["gold"])
        for name, labels in (("shuffled", golds[1:] + golds[:1]), ("wrong_value", wv)):
            restore()
            flips, _ = FC.teach(tok, m, cb, [{**x, "gold": g} for x, g in zip(fb, labels)], imp)
            res[f"{name}_flips"] = len(flips); res[f"{name}_teach"] = run(te)
        restore()
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
        return "<absent>" if it["type"] == "neg" else (it["kind"] if it["src"] == "A" else f"<source-{it['src']}>")

    def faith(ok, ref=b_ok):
        groups = {}
        for it, b, c in zip(ex, ref, ok):
            if b:
                g = groups.setdefault(group(it), [0, 0]); g[0] += c; g[1] += 1
        kept = sum(g[0] for g in groups.values()); n = sum(g[1] for g in groups.values())
        return kept / n, {k: g[0] / g[1] for k, g in groups.items()}, {k: g[1] for k, g in groups.items()}, kept, n

    def summary(ok):
        tot, rates, ns, kept, n = faith(ok)
        by_src = {}
        for it, b, c in zip(ex, b_ok, ok):
            if b:
                s = by_src.setdefault(it["src"], [0, 0]); s[0] += c; s[1] += 1
        return {"accuracy": sum(ok) / len(ex), "faithful_overall": tot, "kept": kept, "base_correct": n, "faithful_ci95": wilson(kept, n),
                "faithful_by_source": {k: {"rate": v[0] / v[1], "n": v[1], "ci95": wilson(*v)} for k, v in sorted(by_src.items())},
                "groups_reported": len(rates), "worst_groups": sorted((round(v, 4), k, ns[k]) for k, v in rates.items())[:10],
                "groups_below_95": sorted(k for k, v in rates.items() if v < 0.95), "wins_over_base": sum(c and not b for b, c in zip(b_ok, ok))}

    out = {"artifact_bytes": sum(p.stat().st_size for p in artifact_files()), "fp16_bytes": 16679859712, "n": len(ex), "base_acc": sum(b_ok) / len(ex), "compact": summary(c_ok), "stages": {}}
    out["size_fraction"] = out["artifact_bytes"] / out["fp16_bytes"]
    if done("full"):
        full = json.load(open(OUTDIR / "full.json"))
        out["library_path"] = summary([T.correct(i, o) for i, o in zip(ex, full)])
    for kind in KINDS:
        if not done(f"teach_{kind}"):
            continue
        r = json.load(open(OUTDIR / f"teach_{kind}.json"))
        if r.get("skipped"):
            out["stages"][kind] = r
            continue
        a = [T.correct(i, o) for i, o in zip(ex, r["exam_after_feedback"])]
        t2, rt, _, _, _ = faith(a)
        kept_cand = sum(x and y for x, y in zip(c_ok, a)) / max(1, sum(c_ok))
        n = r["n_teach"]
        out["stages"][kind] = {"n_annotated": r["n_annotated"], "n_teach": n, "units_off": r["real_flips"],
                               "teach_empty": sum(r["empty_teach"]) / n, "teach_real": sum(r["real_teach"]) / n, "teach_real_ci95": wilson(sum(r["real_teach"]), n),
                               "teach_shuffled": sum(r["shuffled_teach"]) / n, "teach_wrong_value": sum(r["wrong_value_teach"]) / n,
                               "shuffled_units_off": r["shuffled_flips"], "wrong_value_units_off": r["wrong_value_flips"],
                               "exam_acc_after": sum(a) / len(ex), "faithful_after": t2, "kept_right_vs_before_feedback": kept_cand, "groups_below_95_after": sorted(k for k, v in rt.items() if v < 0.95)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(HERE / "data/final2_report.json", "w"), indent=1)
