"""Build-time repair: use the same on/off switches to pull the compressed model back toward the base's own correct answers.
Targets are the base's answers on open data. No weight value changes. usage: repair.py ARTIFACT OUT_MASK"""
import json, random, sys, time
import numpy as np, torch
import runner as R, tasklib as T, fb_lib as L, artifact as A, feedback_core as FC, units as U, repair_set as S

d, out_mask = sys.argv[1], sys.argv[2]
ROUNDS, PER_GROUP_UNITS, MAX_ITEMS = int(sys.argv[3]) if len(sys.argv) > 3 else 3, 50, 60
tok, m, cb = A.load(d)
imp = torch.from_numpy(np.load(f"{d}/importance.npy").astype(np.float32)).to("cuda")
its = S.items()
b = json.load(open(R.HERE / "data/open/repair_base.json")); assert b["ids"] == [x["id"] for x in its]
good = [dict(x, base_text=o) for x, o in zip(its, b["out"]) if T.correct(x, o)]
group = lambda it: it["kind"] if it["type"] == "pos" else "<absent>"
rng = random.Random(11); rng.shuffle(good)
guard, pool = good[:1500], good[1500:]
print(json.dumps({"base_correct": len(good), "guard": len(guard), "pool": len(pool)}), flush=True)


def state():
    return [ex.unit_mask.detach().clone() for _, ex in cb]


def restore(st):
    for (_, ex), s in zip(cb, st):
        ex.unit_mask.data.copy_(s)


def rate(items):
    o = L.predict(tok, m, items)
    return [T.correct(i, x) for i, x in zip(items, o)]


def save():
    np.packbits(np.concatenate([ex.unit_mask.detach().cpu().numpy() > 0 for _, ex in cb]).astype(np.uint8)).tofile(out_mask)


g0 = sum(rate(guard)) / len(guard)
print(json.dumps({"guard_start": round(g0, 4)}), flush=True)
best = g0
total_flips = 0
for rnd in range(ROUNDS):
    t0 = time.time()
    ok = rate(pool)
    wrong = [x for x, c in zip(pool, ok) if not c]
    per = {}
    for x in wrong:
        per.setdefault(group(x), []).append(x)
    print(json.dumps({"round": rnd, "pool_faithful": round(sum(ok) / len(ok), 4), "wrong": len(wrong), "groups": len(per), "secs": round(time.time() - t0)}), flush=True)
    for g, xs in sorted(per.items(), key=lambda kv: -len(kv[1])):
        if len(xs) < 5:
            continue
        st = state()
        batch = xs[:MAX_ITEMS]
        flips, log = FC.teach(tok, m, cb, batch, imp, round_size=25, max_flips=PER_GROUP_UNITS, target=0.75)
        gr = sum(rate(guard)) / len(guard)
        if gr < best - 0.0015 or log[-1]["annotated_fixed"] <= log[0]["annotated_fixed"]:
            restore(st)
            print(json.dumps({"group": g, "reverted": True, "guard": round(gr, 4), "log": [log[0], log[-1]]}), flush=True)
        else:
            best = max(best, gr); total_flips += len(flips); save()
            print(json.dumps({"group": g, "n": len(batch), "flips": len(flips), "fixed": [log[0]["annotated_fixed"], log[-1]["annotated_fixed"]], "guard": round(gr, 4), "total_flips": total_flips}), flush=True)
    print(json.dumps({"round_done": rnd, "secs": round(time.time() - t0), "total_flips": total_flips}), flush=True)
save()
print("ALLDONE", flush=True)
