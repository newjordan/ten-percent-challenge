# Frozen protocol for the second exam (written before it was opened)

**Why a second exam.** The first exam was opened once and showed a miss. Everything after it, including build-time repair, was developed on open data only and must be tested on data it has never touched.

**Base.** `LiquidAI/LFM2-8B-A1B`, 8,339,929,856 parameters, the unmodified checkpoint on the library's default code path. Greedy decoding, default chat template, at most 64 new tokens.

**Task.** Field extraction. One item is a document plus a requested kind. The output is the values copied from the document, one per line, or NONE. An item is correct when the set of normalized values equals the gold set.

**Exam.** `data/locked2/exam.jsonl`, 13,200 items selected by salted hash before any model ran on them.
31 kinds at 300 items each from unused source A documents, 1,500 absent-kind items from source A, 1,200 items from unused source B documents, and 1,200 items from source C, which nothing has touched. Groups are the 31 kinds, `<absent>`, `<source-B>` and `<source-C>`.

**System under test.** `data/artifact_r1`, used through `artifact.py`, the compact form. This is the primary result.
Hidden-unit removal by attribution on unlabeled calibration text, closed-form Hessian-aware rounding, entropy coding, no gradient step on any weight.

**Disclosure: build-time repair.** The artifact also ships a one-bit-per-unit mask. It was chosen with the same on/off mechanism as feedback, targeting answers the base got right on open data, with a separate guard set and revert on damage. 1,150 surviving units are off. No weight value changes. This uses labeled open data at build time, and a reader may count that as fitting.

**Secondary result, fixed in advance.** The same artifact expanded into the library's own expert class (`full.py`), removed units as zero columns, only the router ban patched. It uses full-size memory at run time. It exists to show whether the code path changes the result. It does not replace the primary result.

**Thresholds, as published by the challenge.** At most 10% of fp16 counting every shipped file. At least 98% of base-correct items stay right, and at least 95% within each kind. At most 25 annotated real mistakes, at least 25 unseen items on what is taught, real feedback clearly beating shuffled and empty, nearby items not broken.

**Feedback protocol.** The same four kinds as the first exam: `time`, `date_time`, `coordinate`, `occupation`. Each session starts from the shipped artifact state. Annotations: the first 25 items the artifact gets wrong among the first 300 by hash from the open pool. These documents were excluded from repair. Teaching set: up to 300 unseen items per kind from the locked teaching rows, by hash. A kind with fewer than 25 is reported as not tested. Method: rounds of 25 units, lam 1, at most 400 units, stop at 80% of annotations fixed. Controls: empty, shuffled labels, wrong-value labels.

**Prediction recorded before opening.** On 4,800 independent open items the artifact keeps 97.95% of base-correct answers, 8 of 56 groups are below 95%. The repair was fit on source A open data. I expect overall faithfulness near or below 98%, source B and C lower than source A, and several groups below 95%, so I expect a MISS of the per-kind rule and at best a marginal result on the overall rule. A pass of both would falsify this.

**The exam is used once.** Nothing is tuned after it opens.
