# Handoff: 10% challenge, v2

State at 2026-11-10 00:52 UTC. **Exam 2 is still running, so there is no pass or fail result yet.**

## Where it stands

Exam 2 was opened once, at 2026-11-09 23:04:07 UTC. Its stages run in this order:

| Stage | What it does | State |
|---|---|---|
| `base` | untouched base on the exam | done |
| `cand` | artifact, compact form (primary result) | done |
| `full` | same artifact through the library's own expert code (secondary result) | running |
| `teach` | feedback sessions and controls on four kinds | not started |
| `report` | aggregates, written to `data/final2_report.json` | not started |

Logs are `data/final2_<stage>.log`. Nobody has looked at any exam 2 result yet.

To check whether it is alive: `pgrep -af '[f]inal2.py'`.

If it died, resume it from `v2/`. Finished stages and finished kinds are kept, and the runner refuses a different artifact.

```
export HF_HOME=/work/cache/hf PYTHONDONTWRITEBYTECODE=1
for s in base cand full teach report; do .venv/bin/python final2.py $s > data/final2_$s.log 2>&1 || break; done
```

## Do not

- **Do not run `feedback.py` on `data/artifact_r1`.** It writes `feedback_state.bin`. The loader prefers that file over `repair_state.bin`, and the artifact hash ignores it, so the exam would change with no error.
- Do not tune anything now that exam 2 is open. Exam 1 is also burned.
- Do not read `data/locked2/` by hand, including `final/*.json`. Use the `report` stage, which gives aggregates only. Never read `LOCKED_EXAM_TASKS` in `src/ten_percent/tasks.py`.
- Use the local RTX 3090, one model process at a time.
- Commit, push or post only when asked. Commits are authored by the repo owner only, with no tool attribution lines.

## What is left

1. Wait for `report`, then read `data/final2_report.json`.
2. Score it against every rule, and against the prediction recorded in `PROTOCOL2.md`.
3. Update `RESULTS.md`. It still describes only exam 1, and line 56 overstates the noise floor (see below).
4. Give the final report, including every miss.

Not done, and not doable here: the maintainer's rerun on their own data.

Not re-tested on the frozen artifact: the `feedback.py` command line. `answer.py` was tested offline on it. The `teach` stage runs the same teaching code but not that script. Test it on a copy of the artifact, never on `data/artifact_r1`.

## The frozen system

`data/artifact_r1`, 1,636,798,804 bytes, 9.813% of fp16 (limit 10%). Hash `de7a13a3f71d19d9...`, full value in `PROTOCOL2_LOCK.json`.

Base is `LiquidAI/LFM2-8B-A1B`, 8.34B parameters. The recipe:

- Remove 50% of expert hidden units, ranked by attribution.
- Round with a closed-form Hessian-aware method: 3 or 4 bits on experts, 6 on other layers and the embedding, 8 on the rest.
- Entropy code the result.
- Ship a repair mask with 1,150 more units off.
- Feedback only switches units off. No weight value changes.

## Evidence so far

Faithfulness on 4,800 open items, measured against the untouched base:

| Version | Keeps base-correct answers | Groups below 95% |
|---|---|---|
| Before repair (`artifact_final`) | 96.2% | not re-measured here |
| **r1, 3 repair rounds, frozen** | **97.95%** | 8 of 56 |
| r2, 4 more rounds | 97.84% | 9 of 56 |
| r1 through library code (`full.py`) | 98.01% | 8 of 56 |

Exam 1, before repair: 96.09% faithful, 14 of 57 groups below 95%, source B 94.57%. Real feedback beat shuffled, wrong-value and empty on all four kinds. Details are in `RESULTS.md`.

The recorded prediction for exam 2 is a miss on the per-kind rule, and at best a marginal result on the 98% overall rule.

## What was learned

- **Repair has hit its limit.** r2 rose on its own pool (98.25 to 98.36) but fell on independent items. More rounds overfit.
- **The noise floor does not explain the loss.** Running through the library's own code gained 0.06 points, inside the standard error of about 0.24. The 98.65% identical-weights figure measured a real numeric effect, but removing it recovers almost nothing, so the remaining loss comes from the compression.
- **The remaining losses are mostly not fixable with masks.** Of 69 lost answers on open data (library path):
  - 36 returned a different value, either a corrupted copy like `dismessed` or wider scope like `Honolulu, HI`
  - 19 invented a value where the right answer was NONE
  - 7 added extra values
  - 4 said NONE wrongly
  - 3 dropped values
- **The per-kind 95% rule is the harder one.** County, city, absent-kind and occupation sit near 90 to 94% on open data.
- **Untried, with small expected gain.** About 31 MB of budget is unused, worth roughly 1% more units. Moving bits away from the non-expert layers is also possible. Each trial costs hours and is close to measurement noise.

## Caveats for the write-up

- Repair selects masks using labeled open data at build time. A reader may count that as fitting. It is disclosed in `PROTOCOL2.md`.
- The 4,800 open items also chose r1 over r2, so 97.95% is slightly optimistic.
- All three data sources are synthetic.
- The library-path variant needs full-size memory at run time.
- `/critic` and `/benchguard` are not installed here, so audits have been by hand.

## Where things are

| Path | Purpose |
|---|---|
| `final2.py`, `PROTOCOL2.md`, `PROTOCOL2_LOCK.json`, `LOCK2.json` | exam 2 runner, protocol and hashes |
| `final.py`, `PROTOCOL.md`, `LOCK.json`, `RESULTS.md` | exam 1 |
| `build.py`, `repair.py`, `repair_set.py`, `eval_big.py`, `eval_full.py` | build, repair, open evaluation |
| `artifact.py`, `compact.py`, `full.py`, `feedback_core.py` | runtime |
| `data/repair.log`, `data/repair2.log`, `data/eval_r1.log`, `data/eval_r2.log`, `data/eval_full_r1.log` | logs |

Use the `v2/.venv` environment with `HF_HOME=/work/cache/hf`.

The code and documents in `v2/` are pushed as one commit on branch `v2-compression`, based on `origin/main`. Weights, datasets, logs and locked exams are not in the repo. In the main checkout `v2/` is still untracked, and local `main` has 110 older commits that have not been pushed.
