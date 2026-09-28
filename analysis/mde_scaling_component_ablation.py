"""Which sub-model's refit-from-less-data behavior breaks the MDE curve's
1/n_days assumption?

`TECHNICAL_REPORT.md` found real across-theta variance shrinking
*faster* than 1/n_days as n_days drops (e.g. only 18% of the 1/n-predicted
variance at n_days=3) and named two candidate causes, neither confirmed:
the NHPP arrival fit's unsmoothed zero-fallback (`TECHNICAL_REPORT.md`
already flagged this asymmetry against the OD fit's Dirichlet smoothing) or
simply that `fit_all_models` jointly refits four sub-models from the same
`n_days` at once, not the single-parameter estimator the classical 1/n
result was derived for.

This isolates each candidate: for each n_days, run the bootstrap loop with
only ONE sub-model (arrival / OD / travel-time / fare) refit from that
n_days's resampled subset, while the other three stay fixed at their
nominal fit (the full, `ANCHOR_N_DAYS`-day dataset) — the same "all four
refit together" experiment `validate_mde_scaling` already ran is the
baseline this compares against. Whichever single-model ablation reproduces
most of the full deviation is the dominant cause.

Usage: uv run python analysis/mde_scaling_component_ablation.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.arrivals import fit_arrival_model
from dispatch_eval.calibration.fare import fit_fare_model
from dispatch_eval.calibration.od import fit_od_model
from dispatch_eval.calibration.travel_time import fit_travel_time_model
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import (
    FittedModels,
    bootstrap_resample_trips,
    fit_all_models,
    subset_trips_by_days,
    variance_decomposition,
)
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.synthetic import generate_synthetic_trips

OUTPUT_PATH = Path(__file__).parent / "cache" / "mde_scaling_component_ablation_results.json"
RAW_CHECKPOINT_PATH = Path(__file__).parent / "cache" / "mde_scaling_component_ablation_raw.json"

# Matches the run documented in TECHNICAL_REPORT.md §7 exactly, so results are comparable.
ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 50
N_REPLICATIONS = 4
SEED = 11
ANCHOR_N_DAYS = 24
CANDIDATE_N_DAYS = [3, 6, 12, 24]

COMPONENTS = ["arrival", "od", "travel_time", "fare"]
# SeedSequence entropy must be plain non-negative ints, not strings
# (matches ranking_flip.run_ranking_flip_experiment's own convention).
_COMPONENT_INDEX = {name: i for i, name in enumerate(COMPONENTS)}


def _refit_one_component(
    component: str, resampled_df, nominal: FittedModels, bin_minutes: int, od_time_bin_minutes: int
) -> FittedModels:
    """`nominal` unchanged except for `component`, which is refit from
    `resampled_df` -- the mixed nominal/resampled model an isolated
    component-refit bootstrap draw actually uses.
    """
    if component == "arrival":
        return replace(nominal, arrival=fit_arrival_model(resampled_df, bin_minutes=bin_minutes))
    if component == "od":
        return replace(nominal, od=fit_od_model(resampled_df, time_bin_minutes=od_time_bin_minutes))
    if component == "travel_time":
        return replace(nominal, travel_time=fit_travel_time_model(resampled_df))
    if component == "fare":
        return replace(nominal, fare=fit_fare_model(resampled_df))
    raise ValueError(component)


def run_component_ablation(
    trips_df, n_days: int, component: str, nominal: FittedModels, config: StudyConfig,
    fleet_size: int, abandonment_model: AbandonmentModel, policy_a, policy_b,
    n_bootstrap: int, n_replications: int, seed: int,
) -> dict[str, float]:
    subset = subset_trips_by_days(trips_df, n_days)
    component_idx = _COMPONENT_INDEX[component]
    resample_rng = np.random.default_rng(np.random.SeedSequence([seed, n_days, component_idx]))

    metric_a = np.zeros((n_bootstrap, n_replications))
    metric_b = np.zeros((n_bootstrap, n_replications))
    for b in range(n_bootstrap):
        resampled = bootstrap_resample_trips(subset, resample_rng)
        models_b = _refit_one_component(component, resampled, nominal, bin_minutes=15, od_time_bin_minutes=60)
        for r in range(n_replications):
            for policy, metric_array in ((policy_a, metric_a), (policy_b, metric_b)):
                # Fresh, identically-seeded rng per policy at this (b, r) --
                # CRN: both policies see the same realized scenario, since
                # scenario generation depends only on the seed and models,
                # never on which policy is running (see scenario.py).
                rng = np.random.default_rng(np.random.SeedSequence([seed, n_days, component_idx, b, r]))
                result = run_simulation(
                    fleet_size, models_b.arrival, models_b.od, models_b.travel_time, abandonment_model,
                    policy, config, rng,
                )
                metric_array[b, r] = result.mean_wait_seconds

    decomp = variance_decomposition(metric_a, metric_b)
    return {"across_theta_variance": decomp["across_theta_variance"]}


def main() -> None:
    trips_df = generate_synthetic_trips(
        n_days=ANCHOR_N_DAYS, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=11
    )
    nominal = fit_all_models(trips_df, bin_minutes=15, od_time_bin_minutes=60)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)
    policy_a = NearestIdlePolicy()
    policy_b = BatchedHungarianPolicy()

    results: dict[str, dict[int, float]] = {c: {} for c in COMPONENTS}
    if RAW_CHECKPOINT_PATH.exists():
        saved = json.loads(RAW_CHECKPOINT_PATH.read_text())
        results = {c: {int(k): v for k, v in saved.get(c, {}).items()} for c in COMPONENTS}
        print(f"Resuming from checkpoint: {sum(len(v) for v in results.values())} cells already done.")

    for n_days in CANDIDATE_N_DAYS:
        print(f"\n=== n_days={n_days} ===")
        for component in COMPONENTS:
            if n_days in results[component]:
                print(f"  {component}: already done (across_theta_variance={results[component][n_days]:.2f})")
                continue
            ablation = run_component_ablation(
                trips_df, n_days, component, nominal, config, FLEET_SIZE, abandonment_model,
                policy_a, policy_b, N_BOOTSTRAP, N_REPLICATIONS, SEED,
            )
            results[component][n_days] = ablation["across_theta_variance"]
            print(f"  {component}: across_theta_variance={ablation['across_theta_variance']:.2f}")
            RAW_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_CHECKPOINT_PATH.write_text(json.dumps(results, indent=2))

    # Same 1/n-line-anchored-at-largest-n comparison TECHNICAL_REPORT.md uses.
    summary = {}
    for component in COMPONENTS:
        anchor_variance = results[component][ANCHOR_N_DAYS]
        component_summary = {}
        for n_days in CANDIDATE_N_DAYS:
            predicted = anchor_variance * (ANCHOR_N_DAYS / n_days)
            measured = results[component][n_days]
            component_summary[n_days] = {
                "measured_across_theta_variance": measured,
                "predicted_by_1_over_n": predicted,
                "ratio_measured_to_predicted": measured / predicted if predicted > 0 else float("nan"),
            }
        summary[component] = component_summary

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}")
    print("RATIO (measured / 1-over-n-predicted) BY COMPONENT AND n_days")
    print(f"{'='*70}")
    header = "component".ljust(14) + "".join(f"n={n:<8}" for n in CANDIDATE_N_DAYS)
    print(header)
    for component in COMPONENTS:
        row = component.ljust(14)
        for n_days in CANDIDATE_N_DAYS:
            row += f"{summary[component][n_days]['ratio_measured_to_predicted']:<10.3f}"
        print(row)
    print("\n(all four jointly, from TECHNICAL_REPORT.md: n=3:0.18  n=6:0.31  n=12:0.41  n=24:1.00)")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
