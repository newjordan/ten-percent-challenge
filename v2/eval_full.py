"""Same check as eval_big.py, but through the library's own expert code. usage: eval_full.py ARTIFACT [SAVE]"""
import json, sys
import runner as R, tasklib as T, fb_lib as L, full as F
tok, m = F.load_full(sys.argv[1])
print("impl", m.config._experts_implementation, type(m.model.layers[1].mlp.experts if hasattr(m.model.layers[1], "mlp") else None).__name__, flush=True)
big = R.items_from("dev", 80, 400, salt="v2-big")
p = json.load(open(R.HERE / "data/open/plain_big.json")); assert p["ids"] == [x["id"] for x in big]; base = p["out"]
o = L.predict(tok, m, big)
if len(sys.argv) > 2:
    json.dump(o, open(sys.argv[2], "w"))
bo = [T.correct(i, x) for i, x in zip(big, base)]; co = [T.correct(i, x) for i, x in zip(big, o)]
per = {}
for it, b, c in zip(big, bo, co):
    k = it["kind"] if it["type"] == "pos" else "<absent>"
    if b:
        per.setdefault(k, [0, 0]); per[k][0] += c; per[k][1] += 1
rates = {k: a / n for k, (a, n) in per.items()}
print("RESULT", json.dumps({"n": len(big), "base_acc": round(sum(bo) / len(big), 4), "artifact_acc": round(sum(co) / len(big), 4), "faithful": round(sum(a and b for a, b in zip(bo, co)) / sum(bo), 4),
      "identical": round(sum(a == b for a, b in zip(base, o)) / len(o), 4), "kinds_below_95": sum(v < .95 for v in rates.values()), "of": len(rates),
      "worst": sorted((round(v, 3), k, per[k][1]) for k, v in rates.items())[:10]}), flush=True)
