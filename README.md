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

**Amended 2026-08-04** (see `dispatch-evaluation-project-plan.md` §2 and
`docs/pre_registration.md` for the full, dated rationale — made before any
real-data validation was attempted, not after one failed): the original
plan treated Bengaluru's Namma Yatri aggregates as the primary validation
source. That turned out untenable, not just imperfect — those aggregates
have **no `request_datetime`/`pickup_datetime` split at all**, so wait
time (this project's central metric) isn't observable from them under any
circumstance. Roles restructured accordingly:

- **NYC TLC data — the primary methodological study.** Real, verified
  (`sources/nyc_tlc.py`, checked against the live parquet schema), with
  real trip-level timestamps — the only source here that can actually
  falsify P1's validation thresholds. P1, C1, C2, C3, and C6's
  "real trip-level data" direction all run against this.
- **Bengaluru — Namma Yatri Open Data — a dedicated applicability
  study.** Real, live, ward-level aggregates only (no trip-level
  timestamps or OD). Can't support P1/C2, but is exactly the right target
  for C6's actual question: what can honestly be concluded from
  ward-level Indian open mobility data alone? A separate contribution,
  not downstream of the NYC study.
- **Delhi NCR — a Kaggle ride-booking dataset — dropped from every
  result.** Trip-level shape, but unverified, plausibly-synthetic
  provenance; presenting it as real would undercut a project about
  statistical honesty. The schema adapter (`sources/delhi_ncr.py`) and
  its tests stay as a working capability, unused for any claimed result.

Neither NYC nor Bengaluru data has actually been fetched and run against
the pre-registered thresholds yet — `analysis/nyc_reference_comparison.py`
has already pulled and cached one day of real NYC data for shape
validation, which is the natural starting point, but the full P1
validation run against it is still open. Everything in this repo's own
test suite and analysis scripts currently runs against
`dispatch_eval.sources.synthetic`, a generator that produces schema-valid
trip records from a chosen, arbitrary ground truth — useful for building
and testing the harness (and, per `docs/forecast_degradation.md` and
`docs/indifference_search.md`, for real methodology development that
already found and fixed real bugs), not for any claim about a real city.
See `docs/observability_table.md` for the full quantity-by-quantity
observability audit.

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
                              draw, the B x R x |policies| CRN loop, and all
                              four of the plan's P3 outputs (P(ranked
                              first), Kendall's tau to nominal, C1's
                              variance decomposition, the indifference set,
                              and the minimum-detectable-effect curve)
  decision_currency.py          C3 — express a policy's per-draw wait-time
                              gain (reusing ranking_flip's own bootstrap
                              output, no extra simulation) as "worth N
                              vehicles" relative to a reference policy's
                              fleet-size curve, plus Δdriver-hours (exact)
                              and Δ$/day (a documented flat-fare
                              approximation)
  compute_parity.py             C4 — wall-clock dispatch()/reposition()
                              latency as a function of open requests and
                              idle vehicles, and which problem sizes fit a
                              given real-time budget (measurement half
                              only — see docs/compute_parity.md for why
                              the "degrade + re-measure quality" half
                              needs a mechanism this codebase doesn't have)
  coarsening_ladder.py           C6 — re-runs C2's exact bootstrap-CRN
                              experiment on progressively coarsened trip
                              data (coarser time/OD binning, fare/distance
                              collapsed to route means, destinations
                              shuffled to destroy OD structure), reports
                              the ranking shift at each rung
  geo/
    crosswalk.py               C5 prep, dev-time only (geopandas/shapely/
                              pyproj are dev dependencies, not simulator
                              runtime deps): `areal_interpolate` (sums
                              extensive/count variables — population,
                              SC/ST) and `areal_interpolate_weighted_average`
                              (area-weighted averages intensive/rate
                              variables — household amenity percentages,
                              where summing would be meaningless), plus a
                              validator that checks an interpolated column
                              against an authoritative reference. Applied
                              to Bengaluru's specific 2011-to-369-ward
                              reallocation in analysis/ward_crosswalk.py
                              and analysis/ward_amenities_crosswalk.py
                              (see docs/census_bbmp_data.md).
  forecast_degradation.py       decouples "models the true simulated world
                              runs on" from "models a policy believes" —
                              homogenize_arrival_model / spatially_uniform_
                              arrival_model / marginalize_od_model /
                              flatten_travel_time_model each collapse a
                              fitted model to a deliberately wrong
                              structural family (or fit on fewer days via
                              build_degraded_models), and
                              run_degradation_sweep re-runs C2's bootstrap
                              experiment once per degradation level while
                              the true world stays correctly specified.
                              Built to check whether B2-B4's advantage is
                              an oracle-forecast artifact (see
                              docs/forecast_degradation.md for the real,
                              negative result) and to give C1 a way to
                              probe structural ambiguity, which
                              variance_decomposition's own bootstrap loop
                              can't touch by construction.
```

At the project root: `run_study.py` — the single entrypoint (per the
plan's reproducibility-discipline requirement) that runs P1 calibration,
P2 tuning, and the P3 bootstrap-CRN experiment (B5 wired in) end to end
from one command, writing every `(policy, bootstrap_draw, replication) →
metrics` cell to one parquet file rather than a summary. See "Running it"
below.

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

## Test coverage

147 tests across 18 files, all in `tests/`, all passing as of the latest
commit (`uv run pytest -q`). Every test is synthetic-data-only — nothing
here depends on real Delhi NCR/Bengaluru/NYC files being present.

| File | Count | Covers |
|---|---|---|
| `test_models.py` | 8 | Direct, focused proof that `generate_arrival_minutes` never produces an arrival outside its requested window (aligned/unaligned bin boundaries, a full day, a narrow sub-bin window, the zero-rate case) — part of the silent-exclusion audit. |
| `test_engine.py` | 9 | Event-loop correctness (internally consistent completed trips, undersupply → abandonment, repositioning state transitions, deterministic replay given a fixed seed); the `policy_travel_time_model` split (defaults to the true model, a distinct override actually reaches the policy, realized durations track the *true* model even when the policy's belief is wrong); `unresolved_at_horizon` correctly counts requests neither completed nor abandoned when the horizon cuts a run off, and stays near zero under ample supply. |
| `test_scenario.py` | 3 | The CRN fix — scenario generation is deterministic, order-independent, and a full-trace regression check. |
| `test_calibration.py` | 13 | Every fitted model (arrival, OD, travel time, fare, fleet size) recovers known ground truth from synthetic data within tolerance; `arrival_sparsity_report`'s grid-coverage arithmetic; a direct proof the OD fit never assigns exactly zero probability to any zone; `calibrate_fleet_size`'s `hit_boundary` flag, both when the optimum is interior and when it's forced outside the candidate range. |
| `test_tuning.py` | 4 | `random_search` recovers a known optimum, is reproducible given a seed, respects `minimize`, and improves on a deliberately bad fixed B1 configuration. |
| `test_batched_hungarian.py` | 4 | B1 finds the globally optimal assignment (a case where greedy would provably do worse), radius-cutoff behavior. |
| `test_value_function.py` | 3 | B2's value function: terminal boundary is zero, a busier zone gets a strictly lower cost-to-go, monotonic decay as the horizon closes. |
| `test_value_corrected_hungarian.py` | 4 | B2's cost correction actually flips a tied pickup-cost choice toward the better-value destination; `value_weight=0` recovers B1 exactly. |
| `test_fluid_zone_balancing.py` | 7 | B3's apportionment rounding, moving vehicles out of a zero-demand zone, no-op when balanced, delegation to the wrapped policy, an end-to-end engine run. |
| `test_sampling_lookahead.py` | 6 | B4's sampled-lookahead repositioning, no-idle/nothing-sampled edge cases, never repositioning a vehicle to its own zone. |
| `test_clairvoyant.py` | 9 | B5: trivial same-zone service, infeasible requests correctly unserved, multi-request chaining (the thing a bipartite match can't do), the relaxation over-serving a true simultaneous-capacity tie (documented as intended, not a bug), a direct regression check that a long patience window on one request no longer starves a later one, and the discretization-exclusion diagnostic (coarse vs. fine `bin_minutes` on the same request). |
| `test_ranking_flip.py` | 19 | The full B×R×policy CRN loop; identical policies get exactly identical per-(b,r) metrics (the CRN property, checked directly not assumed); `variance_decomposition` recovers known within/across variance structure; `indifference_set` correctly separates and correctly merges policies; the MDE curve's monotonicity and its unscaled-standard-error edge case; `validate_mde_scaling`'s anchor-point arithmetic; `compute_clairvoyant`'s output shape, plausibility, and discretization-exclusion reporting; `fraction_of_gap_closed`'s arithmetic on a hand-built fake result. |
| `test_forecast_degradation.py` | 13 | Each degrading function's exact arithmetic (average-preserving, zone-independence, spatial-vs-temporal severity ordering, OD-marginal normalization, sigma zeroing); `build_degraded_models`'s flag composition; `build_policy_ladder` actually wires degraded models into B2-B4, not the true ones; an end-to-end two-level sweep. |
| `test_decision_currency.py` | 10 | `FleetWaitCurve.invert`'s interpolation and extrapolation (both directions, flagged); `build_fleet_wait_curve`'s monotonicity; the `decision_currency` sign convention and arithmetic on a fake result. |
| `test_compute_parity.py` | 7 | Latency measurement shapes and problem-size labeling; `feasible_within_budget`'s threshold logic; the empirical proof that `matching_radius_seconds` doesn't shrink the matrix `scipy` actually solves (spied by object, not inferred from timing). |
| `test_coarsening_ladder.py` | 7 | Each coarsening transform's exact effect (route-mean imputation, null-preservation for incomplete trips, OD-shuffle preserves the marginal but breaks a perfect pairing); an end-to-end two-rung ladder run. |
| `test_crosswalk.py` | 13 | `areal_interpolate`'s area-weighted split and conservation properties; `areal_interpolate_weighted_average`'s weighted-mean-not-sum behavior on hand-computable synthetic polygons; `interpolation_error`'s arithmetic; CRS/empty-input error handling for both functions. |
| `test_sources.py` | 8 | Every real-source adapter (`nyc_tlc`, `delhi_ncr`, `bengaluru`) against schema-valid fixtures; required-column enforcement. |

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
  fix. **All four of the plan's P3 outputs are now built**: `indifference_set`
  (the bootstrap-percentile method — for every policy other than the
  grand-mean best, build a (1-α) percentile interval on its paired
  per-draw-mean difference from best; if it contains zero, the policy joins
  the indifference set; no multiple-comparisons correction across the
  simultaneous checks, documented as a first-cut simplification) and
  `minimum_detectable_effect_curve` (extrapolates MDE(n_days) from *one*
  measured variance decomposition, assuming across-theta variance shrinks
  as 1/n_days — a standard asymptotic scaling). **That assumption is now
  empirically checked, not just assumed**: `validate_mde_scaling` (tested)
  re-runs the full bootstrap experiment at several real `n_days` values via
  `subset_trips_by_days` and compares measured `across_theta_variance`
  against the 1/n prediction — see `docs/mde_scaling_validation.md`. Real
  result (24 days synthetic, B0 vs. B1, checked at n_days=3/6/12/24,
  n_bootstrap=50): the assumption does **not** hold exactly — measured
  variance at small n_days is only 18-41% of what 1/n predicts, a real,
  monotonic pattern (confirmed by re-running at a larger bootstrap budget
  after a first, noisier pass), not sampling noise. The direction is the
  safe one: the MDE curve's output is *conservative* (overstates required
  data) rather than misleadingly optimistic, but the underlying cause
  (which of the four jointly-refit sub-models drives it) isn't established
  by this check alone. Still open: running
  at the plan's real scale (B≈200-500) or against real data — only
  small-scale synthetic runs so far (B≤5 for the full experiment, B≤200 for
  the variance-decomposition-only unit tests), and the plan's own
  "metamodel-assisted" fallback for when B×R×|policies| gets too expensive
  hasn't been needed (or built) at this scale. **B5's clairvoyant bound is
  wired in and fixed**: `run_ranking_flip_experiment(..., compute_clairvoyant=True)`
  solves B5 once per (b, r) (not per policy — it doesn't depend on which
  one) on the same realized scenario every policy sees, and
  `RankingFlipResult.fraction_of_gap_closed(baseline)` reports the plan's
  own normalization. First real run against the tuned B0-B4 ladder found
  every policy from B1 up beating the "upper" bound by 5-10% — not a
  finding about these policies, but two real bugs in B5 itself: a
  pessimistic exit-pinning that over-constrained capacity (fixed into a
  genuine earliest-feasible-release relaxation) and a `bin_minutes` too
  coarse for `mean_patience_seconds=300.0`, silently excluding 67.5% of
  requests as "unservable" before the solver ever saw them (fixed by
  using a finer bin at this project's small zone counts, and made visible
  going forward via `ClairvoyantResult.fraction_excluded_by_discretization`).
  Both fixes together, re-measured at B=200: B1/B3/B4 close 90-93% of the
  gap and B2 closes 95%, all properly below 1.0 — a believable result,
  not the old bugs. See `src/dispatch_eval/clairvoyant.py`'s module
  docstring for the full diagnosis and `docs/pre_registration.md` for the
  updated gap-closed reporting commitment.
- **The oracle-forecast confound — checked, and ruled out for the axes
  tested**: every rigorous comparison in `docs/indifference_search.md`
  let B2-B4 forecast the world through models fit *the same way, on the
  same kind of data*, as the model generating the true world — an
  advantage no real deployment has, and a real candidate explanation for
  why no indifference ever showed up between mechanisms.
  `src/dispatch_eval/forecast_degradation.py` (tested) decouples "models
  the true world runs on" from "models a policy believes," via a new
  `policy_travel_time_model` split in the engine/runner plus deliberately
  degraded arrival/OD/travel-time models fed into B2-B4's construction.
  Real 9-level sweep (`analysis/forecast_degradation_sweep.py`,
  `docs/forecast_degradation.md`): B2 wins cleanly at *every* level,
  including one built to remove all spatial knowledge ("which zone is
  busier") from the forecast, not just temporal detail — the maximally
  severe single-axis test. The hypothesis predicted collapse somewhere
  along this axis; it didn't happen anywhere, reported as found rather
  than reframed. Real open question this raises instead: whether B2's
  edge is a residual temporal (not spatial) signal survives even a flat
  forecast, or is actually about its independently-tuned matching radius
  rather than the value function's forecast content at all — not yet
  isolated. `variance_decomposition`'s own docstring now carries the
  matching caveat: `input_uncertainty_ratio` is a floor on input
  uncertainty, not an estimate of it, since the bootstrap loop it comes
  from never introduces structural misspecification by construction.
- **P0.1** (novelty check) — done, provisional GO: `docs/p0_novelty_check.md`.
  A web-search pass (not a full Scholar/WSC-archive traversal — see the
  memo's own caveat) across the plan's five required strings plus the
  named forward-citation sweep found no paper combining
  bootstrap-over-input-models, CRN, and a ranking-flip/indifference-set
  report across dispatch policies for ride-hailing. No pivot triggered.
  Redo more rigorously before submitting anywhere.
- **Pre-registration done**: `docs/pre_registration.md`, committed
  2026-08-04. Locks in P1's validation acceptance thresholds (KS distance
  ≤ 0.10, hour-of-day cosine similarity ≥ 0.90, ward-count/cancellation-
  rate bands), the abandonment-hazard sweep specs, mean wait time as the
  primary P3 metric, α=0.05 for the indifference set, a minimum
  `n_bootstrap≥40` (grounded in `docs/mde_scaling_validation.md`'s own
  finding that 15 was unstable), and a B5 gap-closed reporting commitment
  (grounded in the finding below that 300s patience makes B5 not a real
  upper bound). Explicitly scoped to the not-yet-run confirmatory study
  against real data — see the doc for why this counts as *before* that
  run despite the extensive exploratory/methodology work already in this
  repo.
- **C5's ward-boundary crosswalk built and validated**:
  `docs/census_bbmp_data.md`. Census 2011 ward-level demographics
  (population, SC/ST share, household amenities) are real and downloadable
  via OpenCity/data.gov.in, but Bengaluru's wards have been redrawn at
  least three times since 2011 (198 → 243 → 225 → 369, the last by the
  September 2025 BBMP→Greater Bengaluru Authority restructuring), so
  joining this to Namma Yatri's ward-level data needs a real GIS crosswalk
  between boundary eras, not an ID match. `src/dispatch_eval/geo/crosswalk.py`
  is a generic, tested area-weighted-interpolation crosswalk
  (`tests/test_crosswalk.py`); `analysis/ward_crosswalk.py` applies it to
  the actual 2011→369-ward reallocation and validates it two ways: (1)
  geometry coverage — the two boundary eras cover essentially the same
  ground (717 km² current vs. 712 km² in 2011, 99.6% mean per-ward
  overlap), so interpolation error isn't from missing source area; (2)
  accuracy against the GBA's own official reallocation — the 2025
  delimitation file already carries `TOT_P`/`SC_P`/`ST_P` re-tabulated
  onto the new 369 wards, matching the 2011 Census citywide totals to
  within 0.3%, so it's a real ground truth to check against, not another
  guess. This project's own interpolation matches at the citywide level
  but has substantial per-ward error (~24% median for population, worse
  for the smaller SC/ST counts) — expected for area-weighting's
  uniform-density assumption, and now measured rather than asserted.
  **Practical upshot**: use the GBA file's own population/SC/ST columns
  directly for C5, and reserve this project's interpolation for Census
  variables that have no official reallocation.
  **Household amenities are now crosswalked too**
  (`analysis/ward_amenities_crosswalk.py`), using a *second* new
  function — `areal_interpolate_weighted_average` — because these are
  percentage/rate columns, not counts: summing two wards' "80%
  electrified" and "40% electrified" is meaningless, so this weights by
  overlap-area-averaged rather than overlap-area-summed. Along the way,
  found that OpenCity's same-page "Household Assets" resource is actually
  district/village-level despite its name (checked directly, not used),
  while "Housing and Houselisting Data" genuinely is ward-level and does
  carry electricity/water/vehicle-ownership/sanitation/housing-condition
  rates. Citywide means hold up within a point after crosswalking, except
  treated-water access (a 5.6-point shift — the one variable with real
  spatial heterogeneity, flagged for a wider error bar than the rest).
  Literacy/worker-participation rates are still not obtained: they're in
  data.gov.in's Karnataka PCA file specifically, whose modern portal is a
  client-side-rendered SPA that returned a 403 to `WebFetch` and an empty
  shell to `curl` against a guessed API endpoint — a real, documented
  access obstacle for scripted tools, not a cursory miss. SECC caste data
  was never released; Karnataka's own 2015 caste survey isn't a stable
  public dataset yet; Census 2027 won't have usable tables for years.
  Still unverified: whether Namma Yatri's actual data uses the 369-ward
  scheme this crosswalk targets.
- **C3 (decision currency) built**: `src/dispatch_eval/decision_currency.py`
  (tested, `tests/test_decision_currency.py`). Sweeps a reference policy's
  fleet size to build a wait-time-vs-fleet curve (`build_fleet_wait_curve`),
  then inverts it (`FleetWaitCurve.invert`, with linear extrapolation
  outside the swept range, flagged per-draw rather than silently clipped)
  to convert any policy's per-bootstrap-draw mean wait time — already
  computed by C2's `ranking_flip.py`, reused at zero extra simulation
  cost — into "worth N vehicles" relative to that reference, with a
  bootstrap-percentile interval. Δdriver-hours is exact (vehicle-hours
  scale linearly with fleet size in the engine); Δ$/day is a documented
  approximation (flat fare-per-trip from input data × simulated
  trips-per-vehicle-per-day), since the simulator never assigns a fare to
  a simulated trip. Not yet run end-to-end against a real bootstrap study
  (only unit-tested with fake/small inputs) — that's one call away once
  P3 itself runs at real scale.
- **C4 (compute parity) measurement half built**:
  `src/dispatch_eval/compute_parity.py` (tested,
  `tests/test_compute_parity.py`) + `analysis/compute_parity_sweep.py` —
  see `docs/compute_parity.md`. Times every B0-B4 policy's `dispatch()`
  (+ `reposition()` for B3/B4) at synthetic problem sizes from 10x10 to
  600x600, no real data needed. Real result: every policy fits a
  multi-second production cycle even at 600 simultaneous open
  requests/idle vehicles; the expected O(n³) Hungarian-solver scaling is
  visible in the trend (B1-B4 roughly cubic; B2-B4 run ~4x B0 at matched
  size) but only crosses a tight 100ms budget at the most extreme size
  tested. **The "degrade + re-measure quality" half is not built**: B1/B2's
  `matching_radius_seconds` looked like the plan's "truncate the candidate
  graph" lever but is actually cost-masking — the solved assignment matrix
  is the same size regardless of radius (confirmed by spying on the exact
  call, not inferred from timing —
  `test_matching_radius_does_not_shrink_the_solved_cost_matrix`), so it
  changes assignment quality, not latency. A real graph-truncation
  mechanism (e.g. restricting each request to its k nearest vehicles
  before the matrix is built) doesn't exist yet; that's the prerequisite
  for the rest of C4.
- **C6 (coarsening ladder) built and run twice**: `src/dispatch_eval/coarsening_ladder.py`
  (tested, `tests/test_coarsening_ladder.py`) + `analysis/coarsening_ladder_sweep.py`
  + `analysis/coarsening_ladder_close_pair_run.py` — see
  `docs/coarsening_ladder.md`. Re-runs C2's exact bootstrap-CRN loop
  unchanged (`ranking_flip.py` now takes `bin_minutes`/`od_time_bin_minutes`
  for exactly this reuse) on four progressively coarsened versions of the
  same synthetic data — coarser time/OD binning, then fare/distance
  collapsed to route-level means, then destinations shuffled to destroy
  true OD structure, working toward what Bengaluru's real published data
  actually offers. First run (B0 vs. B1, 14 days synthetic, B=20, R=4):
  ranking survives all four rungs, zero erosion in P(ranked first) —
  expected, since that pair's gap already sits well outside C2's own
  indifference zone at full resolution, not a strong test. **The sharper
  version is now run too**, on a genuinely close pair found via a
  dedicated search (`docs/indifference_search.md` — see below): B2's
  `value_weight` at 0.5 vs. 1.5, two members of a real indifference set.
  That run found something real and worth knowing: **the single nominal
  (un-resampled) fit's point estimate disagrees with the bootstrap-
  majority winner at every rung** (P(weight_1.5 ranked first) = 0.60-0.70
  throughout, while the nominal fit alone favors weight_0.5) — a concrete
  instance of exactly the failure mode this project's "report the
  distribution, not a point estimate" methodology exists to catch, not a
  hypothetical one. Coarsening itself didn't obviously make the
  uncertainty worse across the four rungs tested, for this pair.
- **P4 (C3, C4, C5, C6) built or partially built; P5 not started.**

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
uv run pytest -q     # 147 tests, all synthetic-data-only — see "Test coverage" below for the file-by-file breakdown
uv run ruff check .  # lint
```

To run the confirmatory study end to end (P1 calibration → P2 tuning →
P3 bootstrap-CRN with B5 wired in), per `docs/pre_registration.md`'s
committed settings:

```bash
uv run python run_study.py --n-bootstrap 40 --n-replications 4   # ~4.5 min; writes results/study_results.parquet
uv run python run_study.py --n-bootstrap 200 --n-replications 5  # ~26 min, the plan's target scale
```

Run at both scales already (`docs/pre_registration.md` has the full
comparison): the ranking itself is identical at B=40 and B=200 — same
nominal ranking, same indifference set — which validates the
pre-registered `n_bootstrap ≥ 40` minimum for *ranking* stability
specifically. The variance decomposition itself (`input_uncertainty_ratio`)
moved more between the two scales (0.314 → 0.239, same qualitative
story), so report that one off the full 200 draws, not 40. Gap-closed
fractions are only meaningful from the B=200 run onward — the B=40 run
predates the B5 relaxation/discretization fix described below and its
gap-closed numbers are superseded, not a second data point.

To generate a small synthetic dataset and poke at it interactively:

```python
from datetime import datetime
from dispatch_eval.sources.synthetic import generate_synthetic_trips

df = generate_synthetic_trips(
    n_days=5, zones=["A", "B", "C"], start_date=datetime(2026, 1, 5), fleet_size=40
)
```

## Documentation index

Every `docs/*.md` file, what it's for, and its status:

| File | Status | What it's for |
|---|---|---|
| `docs/observability_table.md` | done (P0.2) | Quantity-by-quantity: is it observable in public data, and how does it enter the simulator. |
| `docs/p0_novelty_check.md` | done, provisional GO | Literature novelty check against the plan's five required search strings. |
| `docs/pre_registration.md` | done, amended 2026-08-04 | P1 validation thresholds, indifference-set α, primary metric, bootstrap-budget minimum, B5 reporting commitment — all committed before the confirmatory run. |
| `docs/census_bbmp_data.md` | crosswalk built, run | C5: what Census/BBMP data exists, the ward-boundary crosswalk (population/SC/ST + household amenities), measured error rates, and the data.gov.in access obstacle for literacy/worker data. |
| `docs/compute_parity.md` | measurement half built | C4: real dispatch-latency numbers across problem sizes; why `matching_radius_seconds` doesn't give a latency-degradation lever. |
| `docs/coarsening_ladder.md` | built, run twice | C6: the coarsening-ladder result for both a large-effect pair (survives trivially) and a genuinely close pair (a real, reported point-estimate-vs-bootstrap-majority disagreement). |
| `docs/mde_scaling_validation.md` | checked | Empirical check of the MDE curve's 1/n_days assumption — it doesn't hold exactly; the curve is conservative, not misleading. |
| `docs/indifference_search.md` | found | Whether this project's own baseline ladder ever produces a real indifference set — not between mechanisms, but between close settings of one (B2's `value_weight`); cross-references the forecast-degradation result. |
| `docs/forecast_degradation.md` | built, run (9 levels) | Whether B2-B4's advantage is an oracle-forecast artifact — checked and ruled out on every axis tried, including removing all spatial demand knowledge; what's still open instead. |
| `docs/silent_exclusion_audit.md` | done | Four places checked for siblings of the B5 discretization bug — one real bug found and fixed (fleet-size calibration silently returning an unconverged boundary value), three others made measurable via new diagnostics. |

## Analysis scripts

Every `analysis/*.py` script is independently runnable
(`uv run python analysis/<name>.py`) and caches its own output under
`analysis/cache/` (gitignored). None require real data.

| Script | What it does |
|---|---|
| `nyc_reference_comparison.py` | Fetches/caches one day of real NYC TLC data; compares the synthetic generator's shape against it. |
| `ward_crosswalk.py` | C5: population/SC/ST crosswalk, 2011→369-ward, with validation against the GBA's official reallocation. |
| `ward_amenities_crosswalk.py` | C5: household-amenity-rate crosswalk using the weighted-average method. |
| `compute_parity_sweep.py` | C4: real B0-B4 latency numbers across problem sizes 10×10 to 600×600. |
| `coarsening_ladder_sweep.py` | C6: the first coarsening-ladder run (B0 vs. B1, large effect). |
| `coarsening_ladder_close_pair_run.py` | C6: the sharper coarsening-ladder run on a genuine indifference-set pair. |
| `policy_tuning_run.py` | P2: identical-budget random-search tuning for B1-B4; writes `cache/policy_tuning_results.json`, consumed by most scripts below. |
| `tuned_ranking_flip_run.py` | The tuned B0-B4 ladder run through a real bootstrap-CRN comparison. |
| `fleet_size_sensitivity_run.py` | Same comparison swept across fleet_size ∈ {15, 30, 60}, looking for indifference. |
| `b3_vs_b4_run.py` | The closest methodological pair in the ladder, head to head, larger bootstrap budget. |
| `value_weight_sweep_run.py` | B2 at several `value_weight` settings against each other — where the real indifference set was found. |
| `clairvoyant_gap_closed_run.py` | B5 wired into the tuned ladder; the real gap-closed numbers, post-fix. |
| `forecast_degradation_sweep.py` | The 9-level forecast-degradation sweep. |

## Next steps

Everything below is genuinely open as of the latest commit — not
organically stale leftovers. Grouped by what's blocking each one.

**Blocked on real data access:**

1. NYC TLC data (the new primary methodological study — see "Data
   status"): `analysis/nyc_reference_comparison.py` already fetches and
   caches one day for shape validation; the full P1 run against it
   (calibrate, then check the pre-registered KS-distance/cosine-shape
   thresholds on *held-out* days) is not done.
2. Bengaluru (Namma Yatri) applicability study: needs an actual scrape or
   the Kaggle mirror — not available in this environment. Once obtained,
   confirm which ward scheme it reports against before trusting any join
   to the C5 crosswalk (assumed 369-ward GBA, unverified).
3. C5's literacy/worker-participation Census variables: data.gov.in's
   Karnataka PCA file specifically resisted scripted access (client-side
   SPA — see `docs/census_bbmp_data.md`); population/SC/ST and household
   amenities are already crosswalked without it.

**Not blocked, not yet done:**

4. C4's "degrade + re-measure quality" half needs an actual
   candidate-graph-truncation mechanism (e.g. restrict each request to
   its *k* nearest vehicles before building the cost matrix) — confirmed
   `matching_radius_seconds` doesn't provide this (`docs/compute_parity.md`).
   Estimated small (~20 lines); unblocks the rest of C4.
5. Fold structural ambiguity into `variance_decomposition`'s own
   arithmetic as a genuine third term, rather than checking it via the
   separate `forecast_degradation.py` sweep it's currently limited to
   (`docs/forecast_degradation.md`).
6. The MDE curve's 1/n_days assumption doesn't hold exactly
   (`docs/mde_scaling_validation.md`) — the cause isn't isolated. An
   ablation refitting one sub-model per bootstrap draw while holding the
   other three at nominal, swept across `n_days`, would isolate which one
   breaks the assumption (arrival fit's sparse-cell fallback is the
   leading guess, not confirmed).
7. `docs/forecast_degradation.md`'s own open question: whether B2's
   surviving advantage (even under the most severe forecast degradation
   tested) is a residual temporal signal from the value function's
   finite-horizon boundary, or actually comes from its independently
   tuned matching radius rather than forecast content at all. Needs a
   further, more surgical experiment (e.g. B1 with B2's exact tuned
   radius but `value_weight=0`) — not run.
8. C3 (`decision_currency.py`) is tested but has never been run
   end-to-end against a real bootstrap study, only fake/small inputs.
9. P5 (RL entry) — not started; explicitly gated behind P3 in the plan
   and requires its own scoping (environment design, seed-variance and
   budget-parity accounting) before starting.
