# dispatch-eval

Input-uncertainty-aware evaluation of ride-hailing dispatch policies.

Full thesis, methodology, and phase plan: [dispatch-evaluation-project-plan.md](dispatch-evaluation-project-plan.md).
Data-quantity-by-quantity observability audit (P0.2): [docs/observability_table.md](docs/observability_table.md).

## What this is

The literature reports dispatch-policy improvements of 1-10% on simulators
calibrated from partial data, without propagating the uncertainty in that
calibration into the comparison. This project builds a small, honest digital
twin plus a bootstrap-over-input-models harness to test whether those
rankings actually survive that uncertainty — and to report the indifference
set, not a winner, when they don't.

## Data status (read this before trusting any number out of this repo)

There is no Indian-city equivalent of NYC TLC's trip-level data. Two sources
fill the primary/secondary roles, and they are **not equally trustworthy**:

- **Bengaluru — Namma Yatri Open Data** (real, live, ward-level aggregates
  only; no trip-level timestamps or OD structure).
- **Delhi NCR — a Kaggle ride-booking dataset** (trip-level shape, but
  **treated as synthetic** — unverified provenance).

Neither has been downloaded into this repo yet (no Kaggle credentials or
scraper configured in this environment). Everything currently runs against
`dispatch_eval.sources.synthetic`, a generator that produces schema-valid
trip records from a chosen, arbitrary ground truth — useful for building and
testing the harness, not for any claim about a real city. See
`dispatch-evaluation-project-plan.md` §2 and `docs/observability_table.md`
for the full reasoning, and `sources/delhi_ncr.py` / `sources/bengaluru.py`
for what's needed to plug in the real files once downloaded (mainly:
confirming the actual column names against `DEFAULT_COLUMN_MAP` in each).

## Architecture

```
src/dispatch_eval/
  schema.py               canonical TRIP_RECORD / WARD_AGGREGATE shapes + validation
  models.py                runtime input models: NHPPArrivalModel, ODModel,
                            TravelTimeModel, FareModel, AbandonmentModel
  simulator/
    entities.py             Vehicle, Request, and their state enums
    events.py                Event + EventType for the heapq queue
    engine.py                SimulationEngine: the event loop itself
    runner.py                 wires fitted models + a policy + fleet size
                              into one run_simulation() call
  policies/
    base.py                  DispatchPolicy protocol (same shape for B0-B5)
    nearest_idle.py           B0 — nearest idle vehicle, greedy, no batching
  calibration/
    arrivals.py               fit NHPP rates per (zone, day_type, bin);
                              index-of-dispersion diagnostic
    od.py                     fit smoothed OD distribution per (origin, time bin)
    travel_time.py            fit lognormal travel time per (origin, dest, hour)
    fare.py                   fit linear fare ~ distance + duration
    fleet_size.py              calibrate the one latent quantity that most
                              matters: fleet size, matched to observed
                              wait-time median/p90; plus a KS-distance helper
                              for validation
  sources/
    synthetic.py              ground-truth generator + simulator-backed
                              trip generation, for development and tests
    delhi_ncr.py               adapter for the Kaggle Delhi NCR CSV (schema
                              unverified — see docstring)
    bengaluru.py                adapter for Namma Yatri's ward aggregates
                              (schema unverified — see docstring)
```

Design choices worth knowing about:

- **One canonical schema, many adapters.** The simulator, calibration code,
  and tests only ever see `TRIP_RECORD_SCHEMA` or `WARD_AGGREGATE_SCHEMA`
  (`schema.py`). Real-source adapters and the synthetic generator are the
  only things that know about a specific data source's raw shape.
- **Decision vs. realization are separate calls.** `TravelTimeModel.expected()`
  is what a dispatch policy uses to decide who's "nearest"; `.sample()` is
  what actually happens when a trip runs. Conflating them would make the
  variance in the fitted travel-time model pointless.
- **Fleet size and the abandonment hazard are never both fit from data.**
  Per the project plan, they aren't jointly identified from matched-trip
  data alone. Fleet size is calibrated (`calibration/fleet_size.py`) against
  the observed wait-time distribution; the abandonment hazard
  (`models.AbandonmentModel`) is a structural axis you sweep across a few
  specifications, not something this codebase tries to estimate.
- **The synthetic generator hides exactly what real data would hide.**
  Abandoned requests come out of `generate_synthetic_trips` folded into
  `"cancelled_customer"`, matching the same censoring every real source in
  this project has. Calibration tests only ever try to recover things that
  are supposed to be recoverable.

## Status against the phase plan

- **P0.2** (observability audit) — done: `docs/observability_table.md`.
- **P1** (minimal digital twin + calibration) — built and tested against
  synthetic data: event-driven engine, B0 policy, all five input-model
  fits, fleet-size calibration. **Not yet run against real Delhi NCR /
  Bengaluru data** (not downloaded in this environment).
- **P0.1, P2-P5** — not started.

## Running it

```bash
uv sync              # installs polars, numpy, scipy, pytest, ruff
uv run pytest -q     # 19 tests: engine correctness, calibration recovery, adapters
uv run ruff check .  # lint
```

To generate a small synthetic dataset and poke at it interactively:

```python
from datetime import datetime
from dispatch_eval.sources.synthetic import generate_synthetic_trips

df = generate_synthetic_trips(
    n_days=5, zones=["A", "B", "C"], start_date=datetime(2026, 1, 5), fleet_size=40
)
```

## Next steps

1. Get the real files: a Kaggle API token for `shayanzk/ola-ride-bookings-dataset`,
   and either a scrape of nammayatri.in/open or the Kaggle mirror of it —
   then confirm each `DEFAULT_COLUMN_MAP` against the actual headers.
2. Run P1 calibration against the real Delhi NCR data and validate against
   held-out days (KS distance on wait time, trip/vehicle-hour, hour-of-day
   shape — see the project plan's P1 validation section).
3. P2: implement the B1-B5 baseline ladder with enforced tuning parity.
