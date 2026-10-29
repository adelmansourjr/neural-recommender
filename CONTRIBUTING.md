# Contributing

Use Python 3.11 or 3.12 in an activated virtual environment. Run the following
commands from this repository root:

```bash
python -m pip install -r requirements-tested.txt -e '.[dev]'
ruff format --check src tests
ruff check src tests
pytest -q
```

The tests synthesize tiny licensed-free fixtures and require no network, GPU,
external model server, or installed MovieLens archive. New tests should cover a
real failure boundary or mathematical invariant, not restate implementation lines.
Examples include malformed candidate weights, future/held-out data affecting
training statistics, nonfinite solver inputs, or an incomplete model generation.

## Changing a method

1. Read the [architecture](docs/ARCHITECTURE.md) and existing protocol/report.
2. Define one question, comparison, primary metric, candidate grid, compute limit,
   and validation selection rule **before** examining new test results.
3. Keep all fitted statistics inside training users. Tune on validation users;
   never use reserved users for fitting, blend weights, confidence thresholds,
   epoch counts, or model selection.
4. Preserve prior protocols/reports. Previously inspected users must be disclosed
   and excluded when a genuinely new reserved sample is claimed. Reproducing an
   existing protocol is useful but does not restore its test set to uninspected.
5. Record null/negative results, confidence-interval assumptions, sample counts,
   data hash, selected settings, code revision when available, and local runtime.

The current `refine` command reuses the preceding upgrade's fixed models and
training/validation membership. It is not a general unbounded hyperparameter
optimizer. A larger search requires a new explicitly budgeted protocol and fresh
evaluation plan, rather than silently editing the checked-in grid.

## Model publication

Use a new artifact directory for unrelated experiments. `train`/`optimize` retain
historical formats; `refine` publishes a complete format 3 generation with an
atomic pointer. Do not edit deployed `.npz` or `.pt` files in place or rewrite
hashes to hide incompatibilities. Add a documented migration and regression
tests when changing artifact schemas. Test prediction parity across save/load
and verify each supported API mode against the same scoring implementation.

`neural-recommender rollback --artifacts artifacts` validates and restores the
previous generation. Restart the local server to load the activated generation.
Do not delete a prior generation needed to reproduce a report. Hard-linked
unchanged matrices keep retained generations inexpensive; an independent copy
on another filesystem may require their full storage size.

## Code and documentation

Keep arithmetic, ranking, selection, and I/O responsibilities separate. Document
why a split, solver assumption, tie policy, or fallback exists. Avoid comments
that merely narrate a line of Python. Public methods should validate their inputs
before unchecked low-level numerical operations. Never suppress numeric warnings
to obtain a passing benchmark; diagnose the data or computation instead.

When behavior changes, update CLI/API help, artifact-format notes, README commands,
and tests together. Preserve backward compatibility deliberately or fail with an
actionable message; do not silently substitute an untrained/default model.

Keep raw MovieLens data and model artifacts out of Git. The repository's MIT code
license does not grant commercial or redistribution rights to the upstream data;
see [DATA_LICENSE.md](DATA_LICENSE.md). CI runs offline tests/lint; executing the
full real-data research pipeline remains an explicit local action.
