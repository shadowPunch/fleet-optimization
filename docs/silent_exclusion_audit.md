# Silent-exclusion audit

**Status: done. Four places checked, each now either measurably safe or
carries a real, new diagnostic. One genuine bug found and fixed.**

Prompted by the B5 discretization bug (67.5% of requests silently dropped
as "unservable" from rounding, not real infeasibility — see
`clairvoyant.py`): are there siblings of that failure mode elsewhere in
the pipeline? Four candidate spots, checked directly rather than assumed
clean.

## 1. Fleet-size calibration — real bug, fixed

`calibrate_fleet_size` is a plain grid search over `candidate_fleet_sizes`
that returns `min(losses, key=losses.get)`. If the true optimum is outside
the range tried, the search silently returns the largest (or smallest)
candidate with no indication it never actually converged — indistinguishable
from a genuine interior optimum in the returned result. **This already
happened in this project's own `run_study.py` runs**: `calibrated
fleet_size=60` every time, with 60 being the literal maximum of
`CANDIDATE_FLEET_SIZES = [10, ..., 60]` — never checked at the time.

Fixed: `FleetSizeCalibrationResult.hit_boundary: bool`, true whenever the
returned fleet size is the min or max of the candidates tried. Tested both
ways — an interior optimum reports `False`, an out-of-range true optimum
(forced in a synthetic test) reports `True` and returns the boundary value
with the flag set (`tests/test_calibration.py`).

## 2. NHPP arrival fit — not a bug, but a real, now-measured asymmetry

A `(zone, day_type, bin_index)` cell with zero observed requests is simply
absent from `NHPPArrivalModel.rates`; `rate_per_minute` falls back to
0.0. For a cell genuinely covered by the fitting window, that's the
correct MLE — not a bug. But unlike `fit_od_model` (which Dirichlet-smooths
every zone into nonzero mass specifically so a sparse zero-count isn't
mistaken for structural impossibility), the arrival fit applies **no
smoothing at all**. A zone/bin that saw zero arrivals by chance in a short
window is treated with full confidence, identically to one that
structurally never has demand there.

Added `arrival_sparsity_report(model, zones, day_types)`: reports what
fraction of the full possible grid rests on the unsmoothed zero fallback
vs. an actual observed count. Also a real, plausible explanation — not yet
confirmed — for `docs/mde_scaling_validation.md`'s own finding that
across-theta variance shrinks *faster* than 1/n_days: an unsmoothed
zero-fallback estimator is *lower*-variance (more confidently wrong, not
more uncertain) at small n than a smoothed one, which would produce
exactly that direction of deviation from the 1/n asymptotic.

## 3. OD smoothing — checked, confirmed correct, now proven not just claimed

`fit_od_model`'s own docstring already claims every zone gets smoothed,
nonzero mass. Checked directly: `all_zones` is built from the *union* of
every zone observed as either an origin or a destination anywhere in the
fitting data (not per-group), so every `(origin, time_bin)` cell's
`dest_probs` — and the global `fallback_probs` — include every zone with
positive probability, including zones never observed as a destination for
that specific cell. `test_fit_od_model_never_assigns_exactly_zero_to_any_zone`
proves this directly on a real fit, rather than resting on the docstring's
own claim.

## 4. Scenario generation / horizon truncation — one real gap, closed

Two sub-checks:

- **`generate_arrival_minutes` never produces an arrival outside its
  requested `[start_minute, end_minute)` window** — checked directly
  (`tests/test_models.py`, new file) across aligned and unaligned window
  boundaries, a full day, a narrow sub-bin window, and the zero-rate case.
  No bug found; this one was already correct by construction.
- **A real gap, found and closed**: `SimulationEngine.run()`'s event loop
  breaks as soon as it pops an event past `horizon_seconds`. A request
  that arrives near the end of the horizon, with a patience window
  extending past it, never gets its own abandonment event processed —
  it's left permanently WAITING, counted in `total_requests` but in
  neither `completed_requests` nor `abandoned_requests`. Not a bug (every
  discrete-event simulation with a finite horizon has this boundary
  effect), but it was **invisible** — `fraction_served` and
  `mean_wait_seconds` were silently computed as if this never happened.
  Added `SimulationResult.unresolved_at_horizon`. Real number, one
  representative run (ample fleet, moderate demand, 542 total requests):
  **12 requests (2.2%) unresolved** — small, but no longer invisible, and
  worth checking against on any run where the horizon is short relative
  to typical patience.

## What this means going forward

Every metric this project reports off `SimulationResult` or a fitted
input model now has a way to check whether it silently rests on excluded
or fallback data: `ClairvoyantResult.fraction_excluded_by_discretization`
(B5), `FleetSizeCalibrationResult.hit_boundary`, `arrival_sparsity_report`,
`SimulationResult.unresolved_at_horizon`. None of these are checked
automatically as a hard gate — they're diagnostics to report alongside
any headline number, the same discipline the project already applies to
every other documented approximation.
