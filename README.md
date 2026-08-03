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
  scenario.py              pre-generates the exogenous request scenario
                            (arrivals, destinations, patience) once, upfront
                            — the fix for the CRN bug described below
  simulator/
    entities.py             Vehicle, Request, and their state enums
    events.py                Event + EventType for the heapq queue
    engine.py                SimulationEngine: the event loop itself, driven
                              by a pre-built Scenario rather than generating
                              requests reactively
    runner.py                 wires fitted models + a policy + fleet size
                              into one run_simulation() call
  policies/
    base.py                  DispatchPolicy protocol (same shape for B0-B5),
                              optional RepositioningPolicy protocol (B3+)
    nearest_idle.py           B0 — nearest idle vehicle, greedy, no batching
    batched_hungarian.py       B1 — batched optimal assignment (scipy Hungarian),
                              tunable batch window (Δ = dispatch_interval_seconds)
                              and matching radius r
    value_corrected_hungarian.py B2 — B1's assignment, but the cost matrix adds
                              a value-function correction so a policy can
                              avoid stranding a vehicle in a low-demand zone
    fluid_zone_balancing.py    B3 — wraps any DispatchPolicy unchanged, adds
                              reposition(): idle vehicles get moved toward a
                              fluid target allocation (idle-vehicle share per
                              zone proportional to that zone's demand share)
    sampling_lookahead.py       B4 — same wrapping pattern as B3, but
                              reposition() samples one realization of near-
                              future requests from the arrival model and
                              Hungarian-matches idle vehicles to them directly
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
    tuning.py                  fixed-budget random-search harness so every
                              policy in the ladder gets an identical tuning
                              budget (the P2 parity condition)
    value_function.py           backward-induction DP for B2: C(zone, time_bin)
                              = expected future idle time from being positioned
                              there, over the fitted arrival/OD/travel-time models
  sources/
    synthetic.py              ground-truth generator + simulator-backed
                              trip generation, for development and tests
    delhi_ncr.py               adapter for the Kaggle Delhi NCR CSV (schema
                              unverified — see docstring)
    bengaluru.py                adapter for Namma Yatri's ward aggregates
                              (schema unverified — see docstring)
  clairvoyant.py              B5 — offline upper bound, not a DispatchPolicy:
                              time-expanded min-cost flow over (zone, time
                              bin) nodes, since one vehicle can serve many
                              requests in sequence and that's a flow, not a
                              one-shot assignment (see its module docstring
                              for the pessimistic-pinning approximation it
                              makes, and why that trade-off was necessary)
  ranking_flip.py              P3 — the ranking-flip experiment: bootstrap
                              resampling, refitting all input models per
                              draw, the B x R x |policies| CRN loop, and
                              reporting (P(ranked first), Kendall's tau to
                              the nominal ranking, C1's variance
                              decomposition)
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
- **Assignment-cost terms that are constant across a row or column get
  dropped, not computed.** B2's value correction only ever adds `C` at the
  *candidate* destination, never at the vehicle's own current position —
  that term is the same for every entry in a vehicle's column, so it can
  never change which column a row prefers (a standard property of the
  assignment problem). Same reasoning is why B1's radius cutoff is a hard
  exclusion rather than a soft penalty term.

## Status against the phase plan

- **P0.2** (observability audit) — done: `docs/observability_table.md`.
- **P1** (minimal digital twin + calibration) — built and tested against
  synthetic data: event-driven engine, B0 policy, all five input-model
  fits, fleet-size calibration. **Not yet run against real Delhi NCR /
  Bengaluru data** (not downloaded in this environment).
- **P2** (baseline ladder) — in progress:
  - **B0** done (P1).
  - **B1** done: batched Hungarian assignment (`policies/batched_hungarian.py`)
    plus a generic fixed-budget tuning harness (`calibration/tuning.py`),
    tested for assignment correctness (including a case where greedy
    nearest-idle is provably worse than the batched optimum), radius-cutoff
    behaviour, and an end-to-end tuning run.
  - **B2** done: `calibration/value_function.py` computes `C(zone, time_bin)`
    via backward induction over the fitted arrival/OD/travel-time models — a
    mean-field, single-vehicle MDP where each idle bin costs `bin_seconds`
    of wasted capacity and serving a request (assumed to depart from the
    vehicle's own zone) costs nothing but chains to `C` at the destination.
    `policies/value_corrected_hungarian.py` adds `value_weight * C(dest,
    dropoff_bin)` to B1's pickup-time cost (`value_weight=0` recovers B1
    exactly; the vehicle's *own* current-position `C` is provably irrelevant
    to the assignment and is omitted, not computed for nothing — see the
    module docstring for why). Tested: terminal boundary, a busy zone
    getting a strictly lower cost-to-go than a quiet one, monotonic decay
    as the horizon closes, the value correction actually flipping a tied
    pickup-cost choice toward the better destination, and an end-to-end run.
  - **B3** done: `policies/fluid_zone_balancing.py` wraps any dispatch
    policy unchanged (in practice B2) and adds a genuinely different
    mechanism — not value-function-guided, deliberately, to keep it
    distinct from B2 rather than just "B2, but chase low `C`." Each zone's
    target share of idle vehicles is set proportional to its share of total
    expected arrival rate right now (largest-remainder apportionment to
    integers), and idle vehicles move from over- to under-supplied zones via
    the same least-cost bipartite assignment B1/B2 use, just matching movers
    to target zones instead of vehicles to requests. Required one engine
    change: `_handle_dispatch_tick` now also checks for an optional
    `policy.reposition(...)` (via `getattr`, so B0-B2 are unaffected) and
    calls it on whatever's still idle after normal dispatch. Tested:
    apportionment rounding, moving vehicles out of a zero-demand zone into a
    busy one, no-op when already balanced, the zero-total-rate edge case,
    delegation to the wrapped policy, and an end-to-end engine run that
    checks vehicles actually changed zones.
  - **B4** done: `policies/sampling_lookahead.py` is a sibling to B3, not a
    wrapper around it — same "delegate dispatch, add reposition()" shape,
    but `reposition` draws one Monte Carlo sample of the requests that might
    arrive over the next `lookahead_seconds` (reusing
    `NHPPArrivalModel.generate_arrival_minutes`, the exact method the engine
    itself uses to generate real arrivals) and Hungarian-matches idle
    vehicles directly to those sampled origin zones, instead of B3's smooth
    expected-rate target share. This needed one protocol change:
    `RepositioningPolicy.reposition` now also takes the engine's own `rng`
    (B3 ignores it; B4 needs it for sampling) — a policy-owned separate rng
    would have been simpler but would silently break P3's common-random-
    numbers requirement, since its draws wouldn't be tied to the shared
    per-replication seed. Tested: moving vehicles toward sampled demand, the
    no-idle-vehicles and nothing-sampled edge cases, never "repositioning" a
    vehicle to the zone it's already in, delegation, and an end-to-end run.
  - **B5** done: `clairvoyant.py` is a time-expanded min-cost flow (one node
    per (zone, time_bin), via `networkx.min_cost_flow`), not a
    `DispatchPolicy` — a single-tick bipartite match (like B1-B4) isn't a
    valid multi-request-per-vehicle upper bound, since one vehicle serving
    several requests in sequence is a flow-conservation problem, not a
    one-shot assignment. Free "wait" edges (same zone, next bin) and free
    "reposition" edges (any zone pair, `k` bins later) let an idle vehicle
    always relocate, matching what B3/B4 can do online — without those, B5
    would be a *looser* bound than the policies it's meant to normalize
    against. Each request gets a capacity-1 in→out edge carrying a large
    service bonus, so maximizing requests served dominates minimizing wait.
    Uses the same **pessimistic-pinning approximation** flagged when this
    was scoped out earlier (a request's exit point is pinned to its patience
    deadline, not its true solution-dependent pickup time) — turned out to
    matter more than expected: serving a request ties up its vehicle until
    *that request's own* patience deadline regardless of how fast it was
    actually picked up, confirmed while writing the tests (a two-request
    chain failed until the first request's patience was shortened). Tested:
    trivial same-zone service, an infeasible request correctly going
    unserved, a single vehicle chaining two sequential requests (the
    capability a bipartite match doesn't have), bonus dominance under that
    pinning cost, and a capacity check. See the module docstring for the
    full reasoning.
- **P3** (the ranking-flip experiment — the plan's centerpiece) — core loop
  built and tested: `ranking_flip.py` runs the full bootstrap × replication ×
  policy CRN procedure from the plan's pseudocode. `bootstrap_resample_trips`
  is a standard nonparametric bootstrap; `fit_all_models` refits arrival, OD,
  travel-time, and fare models on each resampled draw in one call (fleet size
  and the abandonment hazard are deliberately *not* refit per draw — see the
  module docstring for why). `RankingFlipResult` reports `P(policy ranked
  first)` and Kendall's tau to the nominal ranking; `variance_decomposition`
  is C1 (within-theta vs. across-theta variance on a paired policy
  difference, via the law of total variance over the nested (draw,
  replication) design). The CRN property was verified directly, not just
  assumed: two identically-behaved policies produce *exactly* identical
  metrics at every single (b, r) in the test suite, which would fail loudly
  if the scenario/rng wiring were broken the way it was before the earlier
  fix. **Not yet built**: the indifference set at level α and the
  minimum-detectable-effect curve (the plan's other two P3 outputs) — both
  are straightforward extensions of what's here, just not written yet. Also
  not yet run at the plan's real scale (B≈200-500) or against real data —
  only small-scale synthetic runs so far, and the plan's own "metamodel-
  assisted" fallback for when B×R×|policies| gets too expensive hasn't been
  needed (or built) at this scale.
- **P0.1, P4-P5** — not started.

### Fixed: the engine's shared RNG wasn't policy-independent

Found while scoping B5: `SimulationEngine` used to draw every random
number — arrival times, destinations, abandonment patience, *and*
travel-time realizations for whatever got dispatched — from one shared
`rng`, consumed in event-time order. Two different policies make different
numbers of dispatch decisions interleaved between the same two request
arrivals, which shifted how many draws had been consumed from the shared
stream by the time the *next* request's destination and patience got drawn.
So even with an identical seed, two policies did **not** see the same
realized request trace — breaking the "common random numbers across
policies" property the plan calls "non-negotiable" for P3.

Confirmed empirically before fixing it: with the old engine, an identical
seed run through `NearestIdlePolicy` vs. `BatchedHungarianPolicy` produced
111 requests either way, but 80 of them had a different destination and/or
abandonment deadline between the two — the count alone looked fine, only
the actual content differed.

**Fix** (`scenario.py`): the whole exogenous scenario — every request's
arrival time, origin, destination, and patience — is now generated once,
upfront, in an order that depends only on the zone list and fitted models,
never on anything a policy does. `SimulationEngine` no longer generates
requests reactively; it takes a pre-built, read-only `Scenario` and just
schedules arrival events against it (copying each request fresh per run, so
the same `Scenario` object can be safely reused across multiple policies'
engine runs without one run's mutations leaking into another's). Travel-time
*realizations* are still drawn reactively from the engine's own rng during
the event loop — deliberately: which (vehicle, request) pairs actually
occur depends on the policy, so there's no way to pre-fix "the" travel time
for a trip that might not even happen under some policy, and that's exactly
how CRN is meant to apply here (fix the arrival process, let realized trips
vary by policy). `tests/test_scenario.py` covers determinism, order-
independence, and the full-trace regression check described above.

## Validating the synthetic generator against real data

No public trip-level dataset exists for any Indian city, so the synthetic
generator had never been checked against something real. NYC TLC data is —
real, richly fielded, and downloadable with no credentials, unlike Bengaluru
or Delhi NCR. `sources/nyc_tlc.py` is a **verified** adapter (confirmed
against a live scan of the actual parquet schema, unlike the unverified
Delhi NCR / Bengaluru column-map guesses), and
`analysis/nyc_reference_comparison.py` fetches one day of real Uber trips and
compares the synthetic generator's output against it — not to match NYC's
absolute numbers (different currency, city, and fleet scale guarantee they
won't match), but to check whether the *shape* is comparable. Full write-up
with charts:
[Believable Shape, Not Borrowed Numbers](https://claude.ai/code/artifact/32decd9f-e4e5-4dff-8026-0f12160bcd63).

**The first pass at this got something backwards.** Comparing to NYC and
then recommending the generator's parameters move *toward* NYC's numbers
defeats the point of using NYC only as a real-data availability convenience
— the project's actual target is Bengaluru. The rule that survived a second
pass: a *distributional form* (e.g. "trip distance should be right-skewed")
generalizes across cities and is fair to borrow from anywhere; a *magnitude
or structural ratio* (e.g. how heavily fare depends on distance vs.
duration) does not, and should only change when grounded in a real
Bengaluru-specific fact — not NYC's.

What that produced, once real Bengaluru anchors were used instead of NYC's:

- **Trip distance** (`sources/synthetic.py`): now derived from each trip's
  simulated duration through Bengaluru's real average traffic speed
  (TomTom Traffic Index: ~18-22 km/h, among the world's slowest), with
  lognormal noise — not a NYC-shaped fix, a speed-based mechanism that's
  fair to use anywhere. Skewness moved from 0.18 to 1.6. Also fixed a latent
  inconsistency: distance and duration used to be drawn independently.
- **Fare structure**: anchored to BBMP's real regulated auto-rickshaw meter
  (₹36 for the first 2km, then ₹18/km, +50% surcharge 10pm-5am — effective
  Aug 2025) instead of NYC's TLC formula. The distance:duration weighting
  went from 7.5× to **36.7×** — the *opposite* direction from "move toward
  NYC's 3.1×," because Indian auto meters have essentially no continuous
  per-minute charge (only a waiting surcharge), unlike NYC's formula.
- **Driver pay fraction**: 0.75 → **1.0**, matching Namma Yatri's own stated
  zero-commission policy. The old 0.75 happened to sit close to NYC's real
  median (0.72) — a coincidence the first pass mistook for reassurance,
  masking a structurally different commission model.
- **Nightlife demand bump**: added, peaking ~1am (BBMP raised bar/club
  closing hours to 1am in 2024) rather than borrowing NYC's midnight shape.
  Net effect: hourly-shape similarity to NYC *dropped* (96.4% → 93.6%) —
  expected, since the goal was never to resemble NYC.
- **Left alone, on purpose**: OD/zone concentration. No real Bengaluru hub
  data exists to anchor a number, so the arbitrary near-uniform default
  stays arbitrary rather than borrowing NYC's hub geography.
- **The identifiability finding is unchanged and isn't a generator bug**:
  only 13% of NYC's own zone×zone×hour cells have ≥5 trips, covering 48% of
  trips — even 454,813 trips in one day isn't enough to densely populate a
  fine-grained OD-time cube. Supporting evidence for the project's actual
  thesis, not something to fix.

## Running it

```bash
uv sync              # installs polars, numpy, scipy, pytest, ruff, networkx
uv run pytest -q     # 64 tests: engine correctness, calibration recovery, adapters, B1-B5, tuning, CRN scenario, P3
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
3. P3's core loop is built (`ranking_flip.py`) but two of the plan's three
   P3 outputs are still missing: the indifference set at level α, and the
   minimum-detectable-effect curve.
4. Give each of B1-B4 its tuning budget (`calibration/tuning.py`) instead of
   the arbitrary defaults used so far, and wire B5's clairvoyant bound into
   the ranking-flip loop to report "fraction of clairvoyant gap closed" per
   policy per draw — neither done yet.
5. Run P3 at something closer to the plan's real scale (B≈200-500) — only
   small (B≤5) synthetic smoke runs so far. Watch whether B×R×|policies|
   actually needs the plan's metamodel-assisted fallback at that scale
   before building it preemptively.
6. Still open from before: real Delhi NCR / Bengaluru data (a Kaggle token
   and a Namma Yatri scrape, `DEFAULT_COLUMN_MAP` unverified against actual
   headers), and P1 validation against held-out real days.
