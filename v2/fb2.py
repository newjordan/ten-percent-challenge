"""Learning test on the real artifact, with controls. Fixed protocol: lam=1, 25 units per round, at most 400 units, stop at 80% fixed.
Open data only. usage: fb2.py ARTIFACT kind1,kind2,..."""
import json, sys, time
import numpy as np, torch
import runner as R, tasklib as T, fb_lib as L, artifact as A, feedback_core as FC, units as U

d = sys.argv[1]; kinds = sys.argv[2].split(",")
tok, m, cb = A.load(d)
imp = torch.from_numpy(np.load(f"{d}/importance.npy").astype(np.float32)).to("cuda")
pool_rows = R.open_rows("pool"); by_doc = {r["text"]: r for r in pool_rows}
pool_by, dev_by = {}, {}
for it in T.pos_items("A", pool_rows, T.spans_a):
    pool_by.setdefault(it["kind"], []).append(it)
for it in T.pos_items("A", R.open_rows("dev"), T.spans_a):
    dev_by.setdefault(it["kind"], []).append(it)
gen = R.items_from("dev", 20, 100, salt="v2-prune2")
pre = L.predict(tok, m, gen)
pre_ok = [T.correct(i, o) for i, o in zip(gen, pre)]
print(json.dumps({"artifact_dev_acc": round(sum(pre_ok) / len(gen), 4)}), flush=True)


def scores(fb, te, keep, other_wrong):
    te1 = [x for x in te if len(x["gold"]) == 1]; te2 = [x for x in te if len(x["gold"]) >= 2]
    all_items = te + keep + other_wrong + fb
    o = L.predict(tok, m, all_items)
    ok = [T.correct(i, x) for i, x in zip(all_items, o)]
    a, b, c = len(te), len(keep), len(other_wrong)
    multi = [c_ for it, c_ in zip(te, ok[:a]) if len(it["gold"]) >= 2]
    return {"teach_acc": round(sum(ok[:a]) / a, 3), "teach_multi_acc": round(sum(multi) / max(1, len(multi)), 3), "n_multi": len(multi),
            "keep_retained": round(sum(ok[a:a + b]) / b, 3), "other_wrong_fixed": round(sum(ok[a + b:a + b + c]) / max(1, c), 3), "fb_true_fixed": sum(ok[a + b + c:])}


def relabel(fb, golds):
    return [{**x, "gold": g} for x, g in zip(fb, golds)]


for kind in kinds:
    t0 = time.time()
    cand = sorted(pool_by[kind], key=lambda x: T.sid("fb", x["id"]))[:300]
    co = L.predict(tok, m, cand)
    wrong = [c for c, o in zip(cand, co) if not T.correct(c, o)]
    fb = wrong[:25]
    te = sorted(dev_by[kind], key=lambda x: T.sid("te", x["id"]))[:200]
    good = [x for x, c in zip(gen, pre_ok) if c and x["kind"] != kind]
    bad = [x for x, c in zip(gen, pre_ok) if not c and x["kind"] != kind]
    keep = good[:300]; other_wrong = bad[:150]
    print(json.dumps({"kind": kind, "pool_acc": round(1 - len(wrong) / len(cand), 3), "n_fb": len(fb), "n_teach": len(te), "n_keep": len(keep)}), flush=True)
    U.reset(cb)
    print("CELL", json.dumps({"kind": kind, "cell": "empty", "flips": 0, **scores(fb, te, keep, other_wrong)}), flush=True)
    golds = [x["gold"] for x in fb]
    shuf = golds[1:] + golds[:1]
    wv = []
    for x in fb:
        sp = T.spans_a(by_doc[x["doc"]]); alts = sorted((v for k, v in sp.items() if k != kind), key=str)
        wv.append(sorted(alts[0]) if alts else x["gold"])
    n_real = None
    for name, labels in (("real", golds), ("shuffled", shuf), ("wrong_value", wv)):
        U.reset(cb); t1 = time.time()
        flips, log = FC.teach(tok, m, cb, relabel(fb, labels), imp)
        if name == "real":
            n_real = len(flips)
        print("CELL", json.dumps({"kind": kind, "cell": name, "flips": len(flips), "self_fixed_end": log[-1]["annotated_fixed"], "teach_s": round(time.time() - t1), **scores(fb, te, keep, other_wrong)}), flush=True)
    U.reset(cb)
    n_all = sum(ex.unit_mask.numel() for _, ex in cb); idx = torch.randperm(n_all)[:max(1, n_real or 1)]
    U.apply_off(cb, FC._unflatten(cb, idx))
    print("CELL", json.dumps({"kind": kind, "cell": "random_units", "flips": int(n_real or 1), **scores(fb, te, keep, other_wrong)}), flush=True)
    U.reset(cb)
    print(json.dumps({"kind_done": kind, "secs": round(time.time() - t0)}), flush=True)
print("ALLDONE", flush=True)
