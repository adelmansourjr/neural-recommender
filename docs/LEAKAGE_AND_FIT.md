# Leakage and fit audit

The recorded models pass the checks in [the executable audit](../reports/leakage_audit.json).
This establishes isolation for the recorded **new-user preference completion** task. It does
not establish global chronological forecasting, absence of all overfitting, or optimal
neural convergence. Existing predictions, measured benchmarks, protocols, and model files
were preserved; that recorded isolation audit performs no training or hyperparameter selection.
The subsequent [validation-only convergence diagnostic](CONVERGENCE.md) is a
separate, predeclared fitting experiment that discards its models and never scores
reserve users.

## What the task permits

A new user supplies 1, 5, or 10 liked films. The model uses those supplied likes and
models fitted to other users to rank the full catalog, excluding supplied films. The
remaining positive ratings are evaluation labels and are unavailable to the ranker.
Non-positive observed ratings are not added to the query history. Genre features come
from the entire available catalog; this is an existing-item task, not item cold start.

User sequences use rating timestamps, with catalog index breaking ties. The benchmark
does **not** require training interactions to predate each new user's query. Training
users' full observed histories also exclude negative candidates, including observations
later than a training prefix. That is valid under the disclosed offline user split; it
must not be presented as time-causal next-event training or a historical deployment
simulation. A globally chronological experiment would need a separately frozen protocol,
time-available catalog, time-causal negative sampling and fresh evaluation data.

## Concrete checks and corrections

The cached upgrade protocol previously checked only dataset and configuration equality.
That left a gap where an edited membership list or tuning grid could be accepted. The
loader now reconstructs **all deterministic user lists and candidate grids** on every
load. Membership checks reject wrong-split users, duplicates, coerced IDs and JSON
booleans. Refinement validates the inherited upgrade contract before using its users.
Regression tests deliberately inject overlap, reuse inspected users, reorder membership,
duplicate IDs, and alter the grid; each corrupted protocol fails before fitting.

The read-only audit checks the actual preserved dataset and serving generation:

- **2,500 training users, 200 validation users, and three separate 200-user test
  samples** have the recorded deterministic membership. Each test generation excludes
  previously inspected users. These samples are now inspected and cannot be reused as
  a fresh confirmatory test for future model choices.
- Poisoning the positive histories and observed-item sets of **all 800 accessed
  validation/test users** leaves training prefixes, targets, owners, negative pools and
  the fitting interaction matrix unchanged. The check uses the exact production input
  builder for each of the three fixed neural seeds, without running an optimizer.
- Active popularity scores equal the train-only interaction counts exactly. EASE, kNN
  and ALS receive that train-only matrix or its deterministic Gram matrix in production.
  The audit binds the active artifact hashes to the frozen experiment; it does not
  independently attest to a historical training process.
- Training targets never occur in their input prefixes. Training negative pools exclude
  every observed item of that training owner, including positives and rated dislikes.
  Unobserved items remain unlabeled rather than known dislikes.
- Evaluation prefix items and remaining positive targets are disjoint. Selectors and
  blends match the recorded **validation** argmax; ALS selection matches the recorded
  validation optimum. The active bundle and protocol hashes match those receipts.

The timestamp audit found a material limit: prefix/target boundaries share a timestamp
for **90/200** one-like users, **86/199** five-like users and **75/198** ten-like users.
The deterministic tie breaker does not reveal within-session ordering. These users were
not dropped after inspecting results; the original metrics remain intact with this
disclosure. A future strict temporal benchmark must freeze a rule that keeps tied groups
together before any fitting or evaluation.

## Overfitting and underfitting evidence

Each neural seed selects its checkpoint using validation NDCG@10. Training loss declines
from approximately 1.68–1.71 to 0.78–0.79, while validation NDCG improves for all three
seeds. This does not show a late validation collapse in the available five epochs.
However, **all three best validation epochs are the final epoch**. The curves do not
establish convergence: underfitting or a too-short epoch budget remains possible. The
sampled classification loss and full-catalog ranking metric have different objectives,
so their magnitudes cannot be subtracted to estimate a generalization gap.

The follow-up [20-epoch diagnostic](CONVERGENCE.md) reproduces every original
first-five loss/validation value exactly and improves best validation NDCG@10
for all three seeds by 0.0138–0.0185. Under its frozen delta/patience rule, seeds
7 and 19 show a terminal practical plateau while seed 42 remains budget limited.
The existing validation set is reused development data; these are not independent
test improvements, and the active five-epoch models remain unchanged.

Validation was reused for epoch selection, baseline hyperparameters, blend weights and
history-conditional selection across generations. That creates selection-overfitting
risk despite user disjointness. Three neural seeds share a split and do not measure
split uncertainty. Reserve-user bootstrap intervals condition on the fitted models and
are descriptive, with no multiplicity adjustment. Neither user disjointness nor good
baseline gains is proof of an overfitting-free model.

Future model changes should use a new development/validation design, a predeclared
bounded epoch/capacity comparison and a genuinely fresh final sample. The current
reserve results must not determine new stopping rules, architectures or blend weights.
The optional diversity grid is descriptive on already inspected users; it did not
select a new default. No retraining is justified solely to improve published reserve
numbers.

## Reproduce

From this project with the existing local dataset and model artifacts:

```bash
neural-recommender audit-leakage --output reports/leakage_audit_rerun.json
python -m pytest -q
```

The audit refuses an existing output path. It binds source hashes, dataset, protocols,
selection receipts and model hashes, and raises before producing a successful report
when a check fails. The tests use tiny synthetic fixtures, including a real fitted
fixture bundle, so they also run without downloading MovieLens or using local weights.

The distinction between fitting and held-out data follows the
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
Timestamp semantics and dataset limitations come from the
[GroupLens MovieLens-1M README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt).
The tie counts, isolation results and fit diagnostics above are measured by this
repository's audit rather than inferred from those external sources.
