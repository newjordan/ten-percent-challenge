# Frozen protocol (written before the exam was opened)

**Base.** `LiquidAI/LFM2-8B-A1B`, 8,339,929,856 parameters, the unmodified checkpoint on the library's default code path.
Greedy decoding, default chat template, at most 64 new tokens.

**Task.** Field extraction. One item is a document plus a requested kind. The output is the values, copied from the document,
one per line, or NONE. An item is correct when the set of normalized values equals the gold set.

**Exam.** `data/locked/exam.jsonl`, 19,200 items selected by salted hash before any model ran on this task.
55 kinds at 300 items each from source A, 1,500 absent-kind items, and 1,200 items from source B, which nothing was built or tuned on.

**System under test.** `data/artifact_final`. Hidden-unit removal by attribution on unlabeled calibration text, closed-form
Hessian-aware rounding, entropy coding. No gradient step on any weight.

**Thresholds, as published by the challenge.**
- Size: at most 10% of fp16, counting every shipped file.
- Faithfulness: at least 98% of the items the base gets right stay right; at least 95% within each kind.
- Feedback: at most 25 annotated real mistakes; at least 25 unseen items on what is taught; real feedback must clearly beat shuffled labels and empty feedback; nearby exam items must not break.

**Feedback protocol.** Four kinds fixed in advance as the four lowest on development data: `time`, `date_time`, `coordinate`, `occupation`.
Each session starts from the clean artifact. Annotations: the first 25 items, by hash, from the open pool that the artifact gets wrong.
Teaching set: 300 unseen items per kind from the locked teaching pool, by hash. Method: rounds of 25 units, lam 1, at most 400 units, stop at 80% of annotations fixed.
Controls on the teaching set: empty, shuffled labels, wrong-value labels. The full exam is re-run after each real feedback session.

**Prediction recorded before opening.** On 4,800 open items the artifact keeps 96.2% of base-correct answers, so it is expected to MISS the 98% threshold,
and several kinds are expected to fall below 95%. Noise floor measured on open data: identical weights on a different code path keep 98.65%.
The exam is used once. Nothing is tuned after it opens.
