# Hardening and reproduction record — 22 October 2026

This round corrected serving and artifact-validation defects without retraining
models, changing the fixed candidate grid, or selecting settings from reserve
results. The active generation remains `v3-1b898ab1a5394f47`.

## Corrections

- A hash-consistent neural checkpoint could previously replace the catalog's
  genre feature buffer. Loading now requires that buffer to match the verified
  catalog exactly, and checks the finiteness of model state and item encodings.
- Non-object model configuration, ambiguous numeric blend settings, and
  nonnumeric ranker arrays now fail with explicit validation errors before
  inference. Format versions reject booleans, strings, and floating-point
  coercions.
- Finite ALS factors can overflow their normal equations. The solver validates
  the constructed system and target before invoking LAPACK, and rejects invalid
  results. Fold-in also checks history indexes and the cached Gram matrix.
- Catalog dimensions and genre counts are bounded before dense feature
  allocation. Movie display strings must be nonempty and bounded.
- The local HTTP application caps streamed request bytes before JSON parsing,
  limits server concurrency, and attaches non-sensitive request trace metadata.

## Frozen-result reproduction

The [machine-readable audit](serving_audit.json) records the dataset, frozen
report, and every model artifact digest. At histories one/five/ten, it reproduces
all twelve candidates' NDCG@10, Recall@10, long-tail share, and catalog coverage
with absolute tolerance `1e-10` for reported means. All **5,970** rankings across
ten serving modes match an independent stable sort of evaluation scores,
including history exclusion and the rounded scores returned by the service.
This is reproduction on already inspected users, not a new accuracy estimate.

```bash
OMP_NUM_THREADS=2 neural-recommender audit-serving
ruff check .
ruff format --check .
pytest -q
```

The completed local verification passed **66 tests**, including malformed
artifact cases, finite-solver checks, concurrent inference, checkpoint round-trip,
and streamed HTTP boundary, JSON nesting, and unexpected-error privacy tests.
Ruff lint and formatting checks passed. The
only test warning is the installed Starlette test client's upstream `httpx`
deprecation notice.

## What the evidence supports

The model chosen on validation improves over popularity on the reserved
five-like task by `0.0487` NDCG@10, with paired 95% interval
`[0.0232, 0.0771]`. It does not establish superiority over kNN at five likes;
the paired interval crosses zero. At ten likes the selected blend improves over
kNN by `0.0294`, interval `[0.0133, 0.0473]`. At one like, selection correctly
retains popularity. All intervals remain descriptive and unadjusted for multiple
comparisons. See [the frozen benchmark](refined_benchmark.json).

These measured results, reproducible artifacts, explicit negative comparisons,
and serving/evaluation parity make a reviewable portfolio project. They do not
establish current user satisfaction, item cold-start performance, global
chronological deployment accuracy, or internet-facing production readiness.

## Focused follow-up: publication races and cached downloads

A deterministic regression reproduced a source-generation race: replacing a
baseline file after its initial verification caused the publisher to hash the
replacement and activate it under the preceding source metadata. Staged baseline
files now have to match the **original verified source hashes**, and inherited
model identity fields must match the source manifest. Refinement also uses the
same generation path that its in-memory service loaded.

Publication and rollback now share an exclusive writer lock. A conflicting
operation fails before pointer mutation; ordinary success and failure release
the lock. A process termination leaves a fail-closed lock that requires an
operator to confirm all writers have stopped before removal. The two pointer
files and later report write are still not a cross-file transaction.

The cached-download path previously skipped the current byte budget, ignored the
receipt's recorded byte count, and loaded the whole cached archive while hashing.
Malformed receipts also leaked `TypeError` or `KeyError`. Reproduced regressions
now verify size-budget enforcement, strict receipt structure/counts, and streamed
hashing for cached and newly downloaded files. Actual read counts also enforce the
limit if a cached file grows after its initial size check; deeply nested receipts
fail with a controlled validation error.

After these changes and the streamed-growth regressions, the complete suite passed **77 tests** and Ruff lint and
format checks passed. The [new independent reproduction audit](serving_audit_hardening.json)
again matched **5,970 serving rankings** and every frozen candidate metric, with
the same model, dataset, and report hashes. No models, grids, reserve users, or
measured ranking results changed.

## Explicit user-controlled genre diversity

The new optional genre MMR layer changes the final order within a bounded pool
of 100 unseen relevance candidates. It leaves model weights, model selection,
original score values, and the default `relevance_weight=1` path intact. The
browser, API, and CLI expose the same numeric preference, and label the
relevance/genre-variety tradeoff explicitly.

The fixed five-weight protocol was saved before its descriptive evaluation on
the already inspected 199 five-like users. At weight `0.75`, genre list diversity
increased from `0.7524` to `0.8156`, while NDCG@10 changed from `0.4012` to
`0.4004`; its paired NDCG interval crosses zero. Lower relevance weights `0.5`
and `0` show clear archived-rating relevance losses in this sample. All outcomes
are retained in [the tradeoff report](diversity_tradeoffs.json); no default or
winning weight was selected from these users.

The [new default audit](serving_audit_diversity.json) reproduced all 5,970 original
rankings and every frozen candidate metric, with identical model/report hashes.
A numeric overflow safeguard in MMR normalization was separately checked against
the initial formula on all 995 user/weight combinations; no ranking changed.
Regression tests cover exact default parity, deterministic ties, the pure
diversity endpoint, history exclusion, strict/nonfinite weight validation,
constant/extreme scores, and a 10,000-item catalog whose pairwise genre work is
verified to remain at most 100×100. The API integration test verifies metadata
and a tiny end-to-end descriptive-grid reproduction. No training or download
was required.
