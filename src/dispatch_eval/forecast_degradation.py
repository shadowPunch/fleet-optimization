"""Forecast-degradation axis: decouple what the simulator's *true* world
runs on from what a *policy* forecasts about it — and the third piece of
C1's variance decomposition (structural ambiguity), which didn't exist
until this module.

**The confound this exists to fix.** Every policy's belief about travel
time, and B2-B4's value function / arrival-rate reference specifically,
has been built from models fit the *same way*, on the *same kind of data*,
as the model that generates the true simulated world. A value-function or
sampled-lookahead policy is therefore forecasting a world whose generating
process it has, for practical purposes, exactly right — an advantage no
real deployment has, since real forecasts are fit on finite, noisy data
and are frequently structurally wrong (a homogeneous-rate assumption where
demand actually has rush hours; a marginal destination guess where routes
actually cluster; a point travel-time estimate where duration is actually
heavy-tailed). Two consequences follow directly from this, neither
established until measured here:

- The finding in `TECHNICAL_REPORT.md` — four separate rigorous
  comparisons, zero indifference between *mechanisms* (B0 vs. B1 vs.
  B2-B4), only within one (B2's own `value_weight`) — is consistent with
  "these mechanisms genuinely differ this much" but equally consistent
  with "model-based policies are unrealistically advantaged, inflating
  the gap." This module is what tells the two apart.
- `variance_decomposition`'s `input_uncertainty_ratio` (C1) only ever
  captures parameter-estimation error *within a correctly-specified
  family* — the bootstrap resamples trips and refits the same NHPP /
  smoothed-OD / lognormal forms every time. Structural misspecification,
  usually the dominant real-world forecast-error term, is exactly zero
  here by construction. Every `input_uncertainty_ratio` reported so far
  (`TECHNICAL_REPORT.md`, `run_study.py`) is a floor, not an
  estimate — this module supplies the missing third variance component
  the project plan's C1 always specified (intrinsic noise, input-model
  estimation error, *structural ambiguity*) but never built.

**The mechanism.** `SimulationEngine`/`run_simulation` now accept a
`policy_travel_time_model` distinct from the true `travel_time_model`
(see their own docstrings). This module adds the other half: functions
that take a fitted model and deliberately degrade it — either by starving
it of data (`subset_trips_by_days` — already existed, unused for this
until now) or by collapsing it to a structurally wrong family — plus
`build_degraded_models`, which does both, and an orchestration function
that sweeps degradation level and re-runs the ranking-flip experiment at
each point, holding the true world fixed at the correctly-specified,
bootstrap-varying fit throughout. Every policy in the ladder — not just
B2-B4 — sees the same degraded belief via `policy_travel_time_model`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import polars as pl

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import (
    FittedModels,
    RankingFlipResult,
    fit_all_models,
    run_ranking_flip_experiment,
    subset_trips_by_days,
)
from dispatch_eval.simulator.runner import StudyConfig


def homogenize_arrival_model(model: NHPPArrivalModel) -> NHPPArrivalModel:
    """Collapse the time-varying NHPP rate to a flat rate per (zone,
    day_type) — a homogeneous Poisson process. Preserves each
    (zone, day_type)'s average rate exactly (so daily volume is
    unchanged) while destroying the time-of-day shape (rush hour, etc.):
    a policy using this believes demand is uniform across the day.
    """
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for (zone, day_type, _bin_index), rate in model.rates.items():
        key = (zone, day_type)
        sums[key] += rate
        counts[key] += 1
    averages = {key: sums[key] / counts[key] for key in sums}

    new_rates = {
        (zone, day_type, bin_index): averages[(zone, day_type)]
        for (zone, day_type, bin_index) in model.rates
    }
    return NHPPArrivalModel(rates=new_rates, bin_minutes=model.bin_minutes)


def spatially_uniform_arrival_model(model: NHPPArrivalModel) -> NHPPArrivalModel:
    """Collapse the NHPP rate to a *single* rate shared by every zone and
    time bin — strictly more severe than `homogenize_arrival_model`, which
    still leaves each zone's own average rate intact. This destroys the
    spatial signal too: a policy using this can't tell a busy zone from a
    quiet one, only the citywide total. Preserves the citywide average
    rate exactly (`TECHNICAL_REPORT.md`: the natural follow-up to
    check whether B2-B4's advantage over reactive baselines survives
    losing "which zone is busier," not just "when" — none of this
    module's other degradations touch that signal).
    """
    total = sum(model.rates.values())
    n = len(model.rates)
    overall_average = total / n if n else 0.0
    new_rates = dict.fromkeys(model.rates, overall_average)
    return NHPPArrivalModel(rates=new_rates, bin_minutes=model.bin_minutes)


def marginalize_od_model(model: ODModel) -> ODModel:
    """Collapse destination | (origin, time_bin) to a single marginal
    destination distribution, applied regardless of origin or time —
    the average, across every fitted (origin, time_bin) cell, of that
    cell's own destination distribution. Preserves overall citywide
    destination popularity while destroying which routes are actually
    common: a policy using this can't tell "downtown mostly goes to the
    airport in the morning" from "downtown goes everywhere equally."
    """
    zones: set[str] = set()
    for probs in model.dest_probs.values():
        zones.update(probs.keys())
    zones.update(model.fallback_probs.keys())
    zone_list = sorted(zones)

    accum = dict.fromkeys(zone_list, 0.0)
    n = 0
    for probs in model.dest_probs.values():
        for z in zone_list:
            accum[z] += probs.get(z, 0.0)
        n += 1

    marginal = dict(model.fallback_probs) if n == 0 else {z: accum[z] / n for z in zone_list}
    total = sum(marginal.values())
    if total > 0:
        marginal = {z: p / total for z, p in marginal.items()}

    new_dest_probs = {key: dict(marginal) for key in model.dest_probs}
    return ODModel(dest_probs=new_dest_probs, fallback_probs=dict(marginal))


def flatten_travel_time_model(model: TravelTimeModel) -> TravelTimeModel:
    """Zero every cell's sigma, collapsing the lognormal to its own mean —
    a policy using this believes travel time is a deterministic point
    estimate (`.expected()` returns `exp(mu)` instead of
    `exp(mu + sigma^2/2)`) and has no notion that duration is uncertain
    at all. `.sample()` becomes degenerate too (`rng.lognormal(mu, 0)`),
    which is exactly why this must only ever be used as a
    `policy_travel_time_model`, never as the true, realization-generating
    one — `.expected()` is all any policy in this codebase actually calls.
    """
    new_params = {key: (mu, 0.0) for key, (mu, _sigma) in model.params.items()}
    new_fallback = (model.fallback_params[0], 0.0)
    return TravelTimeModel(params=new_params, fallback_params=new_fallback)


@dataclass
class DegradedModels:
    arrival: NHPPArrivalModel
    od: ODModel
    travel_time: TravelTimeModel


def build_degraded_models(
    trips_df: pl.DataFrame,
    n_days: int | None = None,
    homogenize_arrival: bool = False,
    spatially_uniform_arrival: bool = False,
    marginalize_od: bool = False,
    flatten_travel_time: bool = False,
    day_type_col: str | None = None,
    bin_minutes: int = 15,
    od_time_bin_minutes: int = 60,
) -> DegradedModels:
    """Fit the models a *policy* sees, deliberately degraded relative to
    the true generating process.

    `n_days`, if given, fits on only the first `n_days` calendar days of
    `trips_df` (via `subset_trips_by_days`) instead of the full dataset —
    a policy forecasting from less data than actually exists. `None`
    means fit on the full dataset, isolating structural degradation
    (the boolean flags) from data-volume degradation. The flags
    independently collapse the corresponding fitted model to a
    structurally simpler (wrong) family — see each degrading function's
    own docstring for exactly what's destroyed. `spatially_uniform_arrival`
    is strictly more severe than `homogenize_arrival` (destroys the
    *spatial* signal — which zone is busier — not just the temporal one);
    setting both is redundant, `spatially_uniform_arrival` wins if both
    are set.
    """
    fit_df = trips_df if n_days is None else subset_trips_by_days(trips_df, n_days)
    fitted: FittedModels = fit_all_models(
        fit_df,
        day_type_col=day_type_col,
        bin_minutes=bin_minutes,
        od_time_bin_minutes=od_time_bin_minutes,
    )
    if spatially_uniform_arrival:
        arrival = spatially_uniform_arrival_model(fitted.arrival)
    elif homogenize_arrival:
        arrival = homogenize_arrival_model(fitted.arrival)
    else:
        arrival = fitted.arrival
    od = marginalize_od_model(fitted.od) if marginalize_od else fitted.od
    travel_time = (
        flatten_travel_time_model(fitted.travel_time) if flatten_travel_time else fitted.travel_time
    )
    return DegradedModels(arrival=arrival, od=od, travel_time=travel_time)


@dataclass
class DegradationLevel:
    name: str
    n_days: int | None = None
    homogenize_arrival: bool = False
    spatially_uniform_arrival: bool = False
    marginalize_od: bool = False
    flatten_travel_time: bool = False


def build_policy_ladder(
    degraded: DegradedModels, tuned_params: dict, zones: list[str], day_type: str
) -> dict[str, DispatchPolicy]:
    """B0-B4, with B2-B4's value function / reposition arrival-rate
    reference built from `degraded` instead of the true generating
    models. Each policy's own tuned hyperparameters (matching radius,
    value weight, lookahead window — from `analysis/policy_tuning_run.py`'s
    output) are reused unchanged from `tuned_params`, so forecast quality
    is the only thing varying across a degradation sweep, not the
    policies' own tuning — changing two things at once would leave the
    sweep's result ambiguous about which one moved the ranking.
    """
    b1 = tuned_params["B1_batched_hungarian"]["best_params"]
    b2 = tuned_params["B2_value_corrected"]["best_params"]
    b3 = tuned_params["B3_fluid_balancing"]["best_params"]
    b4 = tuned_params["B4_sampling_lookahead"]["best_params"]

    value_function = compute_value_function(
        zones,
        degraded.arrival,
        degraded.od,
        degraded.travel_time,
        day_type=day_type,
        horizon_seconds=24 * 3600.0,
    )

    b2_policy = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=b2["value_weight"], matching_radius_seconds=b2["radius"]
    )
    b3_inner = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=b3["value_weight"], matching_radius_seconds=b3["radius"]
    )
    b4_inner = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=b4["value_weight"], matching_radius_seconds=b4["radius"]
    )

    return {
        "B0_nearest_idle": NearestIdlePolicy(),
        "B1_batched_hungarian": BatchedHungarianPolicy(matching_radius_seconds=b1["radius"]),
        "B2_value_corrected": b2_policy,
        "B3_fluid_balancing": FluidZoneBalancingPolicy(
            dispatch_policy=b3_inner, arrival_model=degraded.arrival, day_type=day_type
        ),
        "B4_sampling_lookahead": SamplingLookaheadPolicy(
            dispatch_policy=b4_inner, arrival_model=degraded.arrival, day_type=day_type,
            lookahead_seconds=b4["lookahead"],
        ),
    }


def run_degradation_sweep(
    trips_df: pl.DataFrame,
    zones: list[str],
    tuned_params: dict,
    fleet_size: int,
    abandonment_model: AbandonmentModel,
    config: StudyConfig,
    n_bootstrap: int,
    n_replications: int,
    seed: int,
    levels: list[DegradationLevel],
) -> dict[str, RankingFlipResult]:
    """Run the full bootstrap-CRN ranking-flip experiment once per
    degradation level, with the true world held at the correctly
    specified, bootstrap-varying fit throughout (`trips_df` and its
    resampling are untouched) and every policy's *belief* — value
    function, reposition reference, per-tick travel-time estimate —
    built from that level's deliberately degraded models instead.
    """
    results: dict[str, RankingFlipResult] = {}
    for level in levels:
        degraded = build_degraded_models(
            trips_df,
            n_days=level.n_days,
            homogenize_arrival=level.homogenize_arrival,
            spatially_uniform_arrival=level.spatially_uniform_arrival,
            marginalize_od=level.marginalize_od,
            flatten_travel_time=level.flatten_travel_time,
        )
        policies = build_policy_ladder(degraded, tuned_params, zones, config.day_type)
        results[level.name] = run_ranking_flip_experiment(
            trips_df,
            zones,
            policies,
            fleet_size,
            abandonment_model,
            config,
            n_bootstrap,
            n_replications,
            seed,
            policy_travel_time_model=degraded.travel_time,
        )
    return results
