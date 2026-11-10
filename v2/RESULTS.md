# The 10% challenge: how I would do it, and what happened

**Outcome: not a complete pass.** The system meets the size limit, beats the base's score, and learns from a few
annotations with no weight value changing. It misses the faithfulness rule: it keeps 96.09% of the answers the base
got right, and the rule is 98%. The maintainer's rerun on independent data has not been done.

## The approach

1. **Compress by removing hidden units, not whole experts.** Every routed expert is a set of hidden units. Each unit is
   ranked by attribution of the base's own answers on calibration text, re-measured after each cut. Half the units go.
   Whole-expert removal is far worse: at about 42% of parameters it kept 71%, where unit removal kept 96%.
2. **Closed-form rounding.** Hessian-aware (GPTQ-style) rounding, with 4 bits on important experts, 3 bits on the least
   important 45% of expert volume, 6 bits on other layers and the embedding. Entropy coded. No gradient step on any weight.
3. **Feedback only switches off units already in the artifact.** Annotated mistakes are attributed to units in rounds of 25;
   the process stops when 80% of the annotations are fixed, or at 400 units. No weight value changes, nothing is added to the
   prompt, and there is no new head or lookup memory. The state is one bit per surviving unit (78,543 bytes).

## Results on the locked exam (opened once)

Base: `LiquidAI/LFM2-8B-A1B`, 8.34B parameters, untouched checkpoint on the library's default path.
Task: extract a requested kind of field from a document, output the values or NONE, exact match.
Exam: 19,200 items, selected by hash before any model ran. 55 kinds at 300 each, 1,500 absent-kind items, 1,200 items from a source nothing was built on.

| Requirement | Result | Status |
|---|---|---|
| At most 10% of fp16, every file | 9.813% (1,636,717,157 of 16,679,859,712 bytes) | Pass |
| Above 0.5B parameters | 8.34B | Pass |
| No weight training | attribution, closed-form rounding, on/off masks | Pass, on my reading |
| Keep at least 98% of base-correct answers | **96.09%** (95% CI 95.76 to 96.40), 541 of 13,850 lost | **Miss** |
| At least 95% within every kind | **14 of 57 groups below 95%** | **Miss** |
| Exam of 2,000+, 300+ per kind, 500+ unseen source | 19,200 / 300 / 1,200 | Pass |
| Maintainer's rerun on independent data | not done | Unknown |

Accuracy: base 72.14%, artifact 75.95%. It also got 1,273 items right that the base got wrong.
Untouched source only: base 66.0%, artifact 68.4%, faithfulness 94.57% (95% CI 92.8 to 95.9).

### Feedback (25 annotated mistakes per kind, fixed in advance; 300 unseen locked items per kind, 279 for `coordinate`)

| Kind | Before | Real labels | Shuffled | Wrong-value | Units off | Taught kind on exam |
|---|---|---|---|---|---|---|
| `time` | 17.3% | **75.7%** | 41.3% | 12.3% | 175 | 27.3% to 79.0% |
| `date_time` | 52.3% | **83.3%** | 71.0% | 39.3% | 125 | 57.3% to 85.7% |
| `coordinate` | 52.0% | **90.3%** | 60.6% | 25.4% | 50 | 59.0% to 91.0% |
| `occupation` | 33.3% | **50.7%** | 32.0% | 15.3% | 400 | 35.0% to 55.7% |

Exam answers that were right before feedback and stayed right: 99.21%, 99.04%, 99.45%, 98.87%.
Exam accuracy after feedback: 79.3%, 78.5%, 78.8%, 79.8%. Faithfulness to the base after feedback: 96.8% to 97.0%.

## Limits that matter

- **The learning is partly generic.** Most taught mistakes are over-listing: the model finds the value and does not stop.
  Shuffled labels still show "one value, then stop", so they help on `time` and `date_time`. Real labels win by 12 to 34 points.
  Wrong-value labels make the model worse, so the change does follow the labels.
- **`occupation` learns weakly and hits the unit cap.** This mistake is a judgment about content, not a behaviour.
- **Masks chosen from labels may count as fitted computation.** No weight value moves, and it is the same operation as the compression, but that is a judgment the maintainer may read differently.
- **98% sits close to a numerical noise floor.** On 1,200 open items, identical weights run through a compact code path kept 98.65% of base-correct answers. The library's rounding depends on how many tokens are processed per call.
- **Both data sources are synthetic.**

## Where the remaining loss is

Measured on 4,800 open items: unit removal alone 97.2%, the full artifact 96.2%, against the noise floor above. Roughly 1.4 points from removal and 1.0 from rounding. Re-ranking units on base-correct items only made it worse (96.1% against 97.2%).

## Reproduce

```
python data/artifact_final/answer.py   data/artifact_final input.jsonl output.jsonl
python data/artifact_final/feedback.py data/artifact_final annotations.jsonl
```

Frozen protocol and hashes: `PROTOCOL.md`, `PROTOCOL_LOCK.json`, `LOCK.json`. Build, in order: `split_a.py`, `lock.py`, `unit_prune.py`, `prune_iter.py`, `gptq_test.py`, `build.py`. Exam run: `final.py`.
