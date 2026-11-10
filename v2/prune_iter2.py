"""Second ranking attempt: importance from base-correct calibration items only, weighted toward close decisions.
Compared with the current mask on 4,800 open items against the untouched base. Full precision. Open data only."""
import json, time
import torch
import runner as R, tasklib as T, units as U, fb_lib as L, lfm

tok, m, blocks, bs = lfm.load()
calib = R.items_from("calib", 30, 150, salt="v2-unit2")
big = R.items_from("dev", 80, 400, salt="v2-big")
p = json.load(open(R.HERE / "data/open/plain_big.json")); assert p["ids"] == [x["id"] for x in big]; base = p["out"]
base_ok = [T.correct(i, o) for i, o in zip(big, base)]
t0 = time.time()
own = L.predict(tok, m, calib)
good = [(i, o.strip()) for i, o in zip(calib, own) if T.correct(i, o) and o.strip()]
print(json.dumps({"calib_items": len(calib), "base_correct_used": len(good), "secs": round(time.time() - t0)}), flush=True)


def weighted_margin(logits, targets):
    t = logits.float()
    own_l = t.gather(1, targets[:, None]).squeeze(1)
    other = t.scatter(1, targets[:, None], -1e9).max(-1).values
    mg = own_l - other
    return (mg / (1.0 + mg.detach().clamp_min(0))).sum()


def importance():
    U.track(blocks, True)
    tot, n = None, 0
    for it, o in good:
        for _, ex in blocks:
            ex.unit_mask.grad = None
        logits, tg = L.forced(tok, m, it, o)
        weighted_margin(logits, tg).backward()
        g = torch.cat([(ex.unit_mask.grad if ex.unit_mask.grad is not None else torch.zeros_like(ex.unit_mask)).flatten() for _, ex in blocks]).abs().float()
        tot = g if tot is None else tot + g
        n += 1
    U.track(blocks, False)
    return tot / n


def score(name):
    t1 = time.time()
    o = L.predict(tok, m, big)
    ok = [T.correct(i, x) for i, x in zip(big, o)]
    print("RESULT", json.dumps({"mask": name, "faithful": round(sum(a and b for a, b in zip(base_ok, ok)) / sum(base_ok), 4), "acc": round(sum(ok) / len(big), 4),
          "identical": round(sum(x == y for x, y in zip(o, base)) / len(big), 4), "secs": round(time.time() - t1)}), flush=True)


old = torch.load("data/open/alive_50.pt").to("cuda")
lfm.apply_alive(blocks, bs, old)
score("current_50pct")
U.reset(blocks)
for _, _, router, _ in bs:
    router.banned.fill_(False)
alive = torch.ones_like(old)
for keep in (0.85, 0.7, 0.6, 0.5):
    t1 = time.time()
    imp = importance()
    imp[~alive] = -1
    th = imp.kthvalue(alive.numel() - int(keep * alive.numel()) + 1).values
    alive = imp >= th
    lfm.apply_alive(blocks, bs, alive)
    torch.save(alive.cpu(), f"data/open/alive_b{int(keep * 100)}.pt")
    print(json.dumps({"keep": keep, "round_secs": round(time.time() - t1)}), flush=True)
torch.save(imp.cpu(), "data/open/unit_importance_b.pt")
score("reranked_50pct")
print("ALLDONE", flush=True)
