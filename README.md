# dispatch-eval

Input-uncertainty-aware evaluation of ride-hailing dispatch policies.

Published comparisons of dispatch policies typically report a single
point estimate — "policy X beats policy Y by 3%" — without asking whether
that ranking survives the uncertainty in the input models the comparison
was built from. This project builds a small, honest digital twin (an
event-driven dispatch simulator) plus a bootstrap-over-input-models
evaluation harness, and tests whether a standard ladder of five dispatch
mechanisms' rankings actually hold up once that uncertainty is propagated
— reporting the *indifference set*, not a forced winner, when they don't.

**For the full methodology, every measured result, and how to reproduce
all of it, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).** This file is
the short version.

## Headline results

- **No indifference between structurally different dispatch mechanisms**
  (greedy vs. batched-optimal vs. value-corrected vs. two repositioning
  heuristics) across every configuration tested — one mechanism wins
  100% of bootstrap draws every time, robust to forecast misspecification
  and the abandonment-hazard specification. Indifference *does* show up
  — inside one mechanism's own tuning dial, where four settings spanning
  a 3x range are statistically indistinguishable.
- Re-running that close, indistinguishable pair on progressively
  coarsened data (matching what real published open mobility data
  actually offers — ward-level zones, no request/pickup timestamp split)
  is the one place in this whole project where a ranking **actually
  flips** — and where the bootstrap-majority winner disagrees with the
  single point-estimate winner at every coarser resolution tested.
- The simulator does **not** currently reproduce real New York City
  wait-time data at a pre-registered tolerance (Kolmogorov-Smirnov
  distance 0.74 against a 0.10 threshold), while its arrival-timing shape
  does (cosine similarity 0.99). Reported as a finding, not adjusted
  after seeing it — see the technical report for the root-cause
  follow-up.
- Structural uncertainty about a latent behavioral parameter (how
  quickly riders abandon an unmatched request) turns out to matter
  roughly **25x more** than sampling-driven input-model estimation error
  — the opposite of where a naive two-term variance decomposition would
  point.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quickstart

```bash
uv run pytest -q     # 158 tests, synthetic-data-only, no network needed
uv run ruff check .  # lint
```

Run the confirmatory study end to end (input-model calibration → policy
tuning → the bootstrap-CRN comparison):

```bash
uv run python run_study.py --n-bootstrap 40 --n-replications 4   # ~4.5 min
```

Every other analysis behind the headline results above is an independently
runnable script under `analysis/` — see the
[analysis scripts table](TECHNICAL_REPORT.md#analysis-scripts) in the
technical report for what each one produces and how to run it.

## What's in this repo

```
src/dispatch_eval/   the simulator, dispatch policies (B0-B5), input-model
                      calibration, and the bootstrap-CRN evaluation harness
tests/                158 tests, synthetic-data-only
analysis/             independently runnable scripts, one per measured result
run_study.py          single entrypoint: calibration + tuning + the full
                      bootstrap-CRN confirmatory study
```

Full module-by-module architecture: [TECHNICAL_REPORT.md §3](TECHNICAL_REPORT.md#3-system-architecture).

## Data

No trip data is bundled with this repository — `analysis/cache/` and
`results/` are gitignored, and every script that needs real data fetches
and caches it itself from a public source on first run (no credentials
needed for NYC TLC or Census/BBMP data). See
[TECHNICAL_REPORT.md §8](TECHNICAL_REPORT.md#fetching-real-data) for exact
sources and URLs. Everything else in this repo — the full test suite and
every ladder/mechanism-comparison result — runs against a synthetic trip
generator and needs no external data at all.

## Status

The confirmatory synthetic-data study (§5.1-5.4, 5.6-5.7 in the technical
report) is complete. Real-data validation against NYC (§5.5) is complete
and reported as a partial failure — the arrival-timing shape validates,
the wait-time distribution does not, root-caused to a genuine scale gap
in the reactive baseline's dispatch mechanism rather than a fixable
calibration artifact. See the technical report's Limitations section for
the full, current list of what remains open.
