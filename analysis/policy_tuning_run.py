"""Give B1-B4 an identical tuning budget, replacing the arbitrary fixed
parameters used everywhere else in this project so far (e.g.
`matching_radius_seconds=900.0`, `value_weight=1.0`, `lookahead_seconds=300.0`
in `analysis/compute_parity_sweep.py`).

The plan's P2 parity condition: every policy in the baseline ladder gets
the *same* evaluation budget (`n_evaluations`) and the *same* seed-per-
candidate protocol (`calibration/tuning.py`'s `random_search`), so "policy
X beats policy Y" can't secretly mean "policy X got a better search." No
real data needed — nominal fitted models come from this project's own
synthetic generator, same pattern as every other analysis script here.

B3 gets its own tuning run rather than inheriting B2's: its `dispatch()`
delegates to the wrapped policy unchanged, but its `reposition()` changes
which vehicles are idle *where* going into the next dispatch tick, so the
same (value_weight, matching_radius_seconds) that's optimal for B2 alone
isn't guaranteed to still be optimal once repositioning is added — this
is checked empirically below (see whether the tuned params actually
differ), not assumed either way.

Usage: uv run python analysis/policy_tuning_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.tuning import ParameterSpec, random_search
from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.synthetic import generate_synthetic_trips

OUTPUT_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_EVALUATIONS = 25  # identical across every policy -- the parity condition
TUNING_SEED = 17
EVAL_RNG_SEED = 0  # same seed for every candidate within a policy's own search


def build_nominal_models():
    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all",
        horizon_seconds=24 * 3600.0,
    )
    return models, value_function


def _evaluate(policy, models, abandonment_model, dispatch_interval_seconds: float) -> float:
    config = StudyConfig(
        zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS,
        dispatch_interval_seconds=dispatch_interval_seconds,
    )
    result = run_simulation(
        FLEET_SIZE, models.arrival, models.od, models.travel_time, abandonment_model,
        policy, config, np.random.default_rng(EVAL_RNG_SEED),
    )
    return result.mean_wait_seconds


def tune_b1(models, abandonment_model):
    def evaluate_fn(params):
        policy = BatchedHungarianPolicy(matching_radius_seconds=params["radius"])
        return _evaluate(policy, models, abandonment_model, params["delta"])

    return random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("delta", low=5.0, high=120.0),
            ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True),
        ],
        n_evaluations=N_EVALUATIONS,
        seed=TUNING_SEED,
    )


def tune_b2(models, abandonment_model, value_function):
    def evaluate_fn(params):
        policy = ValueCorrectedHungarianPolicy(
            value_function=value_function,
            value_weight=params["value_weight"],
            matching_radius_seconds=params["radius"],
        )
        return _evaluate(policy, models, abandonment_model, params["delta"])

    return random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("delta", low=5.0, high=120.0),
            ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True),
            ParameterSpec("value_weight", low=0.0, high=3.0),
        ],
        n_evaluations=N_EVALUATIONS,
        seed=TUNING_SEED,
    )


def tune_b3(models, abandonment_model, value_function):
    def evaluate_fn(params):
        inner = ValueCorrectedHungarianPolicy(
            value_function=value_function,
            value_weight=params["value_weight"],
            matching_radius_seconds=params["radius"],
        )
        policy = FluidZoneBalancingPolicy(
            dispatch_policy=inner, arrival_model=models.arrival, day_type="all"
        )
        return _evaluate(policy, models, abandonment_model, params["delta"])

    return random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("delta", low=5.0, high=120.0),
            ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True),
            ParameterSpec("value_weight", low=0.0, high=3.0),
        ],
        n_evaluations=N_EVALUATIONS,
        seed=TUNING_SEED,
    )


def tune_b4(models, abandonment_model, value_function):
    def evaluate_fn(params):
        inner = ValueCorrectedHungarianPolicy(
            value_function=value_function,
            value_weight=params["value_weight"],
            matching_radius_seconds=params["radius"],
        )
        policy = SamplingLookaheadPolicy(
            dispatch_policy=inner, arrival_model=models.arrival, day_type="all",
            lookahead_seconds=params["lookahead"],
        )
        return _evaluate(policy, models, abandonment_model, params["delta"])

    return random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("delta", low=5.0, high=120.0),
            ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True),
            ParameterSpec("value_weight", low=0.0, high=3.0),
            ParameterSpec("lookahead", low=30.0, high=600.0),
        ],
        n_evaluations=N_EVALUATIONS,
        seed=TUNING_SEED,
    )


def main() -> None:
    models, value_function = build_nominal_models()
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Tuning B1-B4, {N_EVALUATIONS} evaluations each (identical budget)...")
    results = {
        "B1_batched_hungarian": tune_b1(models, abandonment_model),
        "B2_value_corrected": tune_b2(models, abandonment_model, value_function),
        "B3_fluid_balancing": tune_b3(models, abandonment_model, value_function),
        "B4_sampling_lookahead": tune_b4(models, abandonment_model, value_function),
    }

    output = {
        name: {"best_params": r.best_params, "best_score_mean_wait_seconds": r.best_score}
        for name, r in results.items()
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    for name, r in results.items():
        print(f"{name}: best_score={r.best_score:.2f}s  params={r.best_params}")


if __name__ == "__main__":
    main()
