"""What is B2's surviving advantage, mechanically?

`forecast_degradation_sweep.py` found B2 beating every reactive baseline at
*every* degradation level, including the maximally severe spatially-uniform
one — the opposite of the predicted collapse. That result says the
advantage survives; it doesn't say what it's made of. Three checks, same
setup (zones, tuning, degradation levels) as `forecast_degradation_sweep.py`
so the numbers are directly comparable:

1. `var_across_zones(C[:, t])` at each degradation level: how much spatial
   signal is actually left in B2's value function for it to exploit. If
   this collapses toward zero at `spatially_uniform_arrival_only` while B2
   still wins, the win at that level can't be coming from the value
   function reading spatial demand imbalance — it has to be coming from
   something else (the matching radius, most likely candidate below).
2. B1 running with B2's tuned radius and `value_weight=0`. `value_weight=0`
   makes `ValueCorrectedHungarianPolicy` mechanically identical to B1's own
   cost function (pickup time only) — this arm isolates whether B2's
   *radius* (154s, vs B1's own tuned 205s) already explains part of the
   gap, independent of any value-weighting at all.
3. Whether B1/B2 were retuned per degradation level (`build_policy_ladder`'s
   own docstring says no, deliberately, to isolate forecast quality from
   tuning as confounds) — checked, confirmed, and then done anyway here as
   a robustness check: retune both against each level's degraded forecast
   and see whether the fixed-tuning finding survives.

Usage: uv run python analysis/b2_mechanism_diagnostic.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.tuning import ParameterSpec, random_search
from dispatch_eval.calibration.value_function import ValueFunction, compute_value_function
from dispatch_eval.forecast_degradation import DegradationLevel, build_degraded_models
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models, indifference_set, run_ranking_flip_experiment
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "b2_mechanism_diagnostic_results.json"

# Matches forecast_degradation_sweep.py exactly, so results are comparable.
ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 25
N_REPLICATIONS = 4
SEED = 23
N_DAYS_TOTAL = 10
N_TUNING_EVALUATIONS = 25
TUNING_SEED = 17
EVAL_RNG_SEED = 0

LEVELS = [
    DegradationLevel("true_forecast"),
    DegradationLevel("n_days_1", n_days=1),
    DegradationLevel("spatially_uniform_arrival_only", spatially_uniform_arrival=True),
    DegradationLevel("marginal_od_only", marginalize_od=True),
    DegradationLevel("flat_travel_time_only", flatten_travel_time=True),
    DegradationLevel(
        "fully_degraded", n_days=1, spatially_uniform_arrival=True, marginalize_od=True,
        flatten_travel_time=True,
    ),
]


def var_across_zones(value_function: ValueFunction, zones: list[str]) -> np.ndarray:
    """Variance across zones of C(zone, t), one value per time bin t.

    High variance means the value function still distinguishes "good" from
    "bad" zones to be in at time t -- signal B2's cost correction can
    actually use. Near-zero means every zone looks equally (un)attractive:
    the value term becomes a constant added to every cell of a Hungarian
    assignment column, which changes nothing about which column it prefers.
    """
    return np.array(
        [np.var([value_function.cost_to_go(z, t) for z in zones]) for t in range(value_function.n_bins)]
    )


def _degraded_evaluate(policy, true_models, abandonment_model, degraded_travel_time, dispatch_interval_seconds):
    config = StudyConfig(
        zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS,
        dispatch_interval_seconds=dispatch_interval_seconds,
    )
    result = run_simulation(
        FLEET_SIZE, true_models.arrival, true_models.od, true_models.travel_time, abandonment_model,
        policy, config, np.random.default_rng(EVAL_RNG_SEED), policy_travel_time_model=degraded_travel_time,
    )
    return result.mean_wait_seconds


def retune_b1_b2_for_level(true_models, abandonment_model, degraded, value_function) -> dict:
    """Retune B1 and B2 against this level's degraded forecast, deployed
    against the true (undegraded) world -- the same protocol
    `run_degradation_sweep` uses to *evaluate* the fixed-tuning ladder,
    applied here to *tuning* instead, so the two are directly comparable.
    """

    def eval_b1(params):
        policy = BatchedHungarianPolicy(matching_radius_seconds=params["radius"])
        return _degraded_evaluate(policy, true_models, abandonment_model, degraded.travel_time, params["delta"])

    def eval_b2(params):
        policy = ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=params["value_weight"],
            matching_radius_seconds=params["radius"],
        )
        return _degraded_evaluate(policy, true_models, abandonment_model, degraded.travel_time, params["delta"])

    b1_result = random_search(
        eval_b1,
        [ParameterSpec("delta", low=5.0, high=120.0), ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True)],
        n_evaluations=N_TUNING_EVALUATIONS, seed=TUNING_SEED,
    )
    b2_result = random_search(
        eval_b2,
        [
            ParameterSpec("delta", low=5.0, high=120.0),
            ParameterSpec("radius", low=120.0, high=2400.0, log_scale=True),
            ParameterSpec("value_weight", low=0.0, high=3.0),
        ],
        n_evaluations=N_TUNING_EVALUATIONS, seed=TUNING_SEED,
    )
    return {
        "B1_retuned": {"best_params": b1_result.best_params, "best_score_mean_wait_seconds": b1_result.best_score},
        "B2_retuned": {"best_params": b2_result.best_params, "best_score_mean_wait_seconds": b2_result.best_score},
    }


def main() -> None:
    tuned_params = json.loads(TUNING_RESULTS_PATH.read_text())
    b1_fixed = tuned_params["B1_batched_hungarian"]["best_params"]
    b2_fixed = tuned_params["B2_value_corrected"]["best_params"]

    trips_df = generate_synthetic_trips(
        n_days=N_DAYS_TOTAL, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    true_models = fit_all_models(trips_df)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    summary = {}
    for level in LEVELS:
        print(f"\n=== {level.name} ===")
        degraded = build_degraded_models(
            trips_df, n_days=level.n_days, homogenize_arrival=level.homogenize_arrival,
            spatially_uniform_arrival=level.spatially_uniform_arrival, marginalize_od=level.marginalize_od,
            flatten_travel_time=level.flatten_travel_time,
        )
        value_function = compute_value_function(
            ZONES, degraded.arrival, degraded.od, degraded.travel_time, day_type="all",
            horizon_seconds=24 * 3600.0,
        )

        # 1. How much spatial signal is left in C for B2 to exploit.
        variance_curve = var_across_zones(value_function, ZONES)
        print(f"  var_across_zones(C[:,t]): mean={variance_curve.mean():.4f}  "
              f"max={variance_curve.max():.4f}  min={variance_curve.min():.4f}")

        # 2. B0-B4 (fixed tuning, matching forecast_degradation_sweep.py) plus
        #    the B1-with-B2's-radius ablation arm, added because CRN is keyed
        #    off (seed, bootstrap, replication) per policy, not policy count
        #    or order -- adding an arm cannot change any other arm's result.
        policies = {
            "B1_batched_hungarian": BatchedHungarianPolicy(matching_radius_seconds=b1_fixed["radius"]),
            "B2_value_corrected": ValueCorrectedHungarianPolicy(
                value_function=value_function, value_weight=b2_fixed["value_weight"],
                matching_radius_seconds=b2_fixed["radius"],
            ),
            "B1_radius_from_B2": ValueCorrectedHungarianPolicy(
                value_function=value_function, value_weight=0.0, matching_radius_seconds=b2_fixed["radius"],
            ),
        }
        result = run_ranking_flip_experiment(
            trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config, N_BOOTSTRAP, N_REPLICATIONS, SEED,
            policy_travel_time_model=degraded.travel_time,
        )
        indifferent = indifference_set(result, alpha=0.05)
        probs = result.probability_ranked_first()
        print(f"  nominal ranking (B1 / B2 / B1-with-B2-radius): {result.nominal_ranking}")
        print(f"  P(ranked first): {probs}")
        print(f"  indifference set: {sorted(indifferent)}")

        # 3. Retune B1 and B2 fresh against this level's degraded forecast.
        retuned = retune_b1_b2_for_level(true_models, abandonment_model, degraded, value_function)
        print(f"  B1 fixed-tuning params: {b1_fixed}")
        print(f"  B1 retuned for this level: {retuned['B1_retuned']}")
        print(f"  B2 fixed-tuning params: {b2_fixed}")
        print(f"  B2 retuned for this level: {retuned['B2_retuned']}")

        summary[level.name] = {
            "var_across_zones_mean": float(variance_curve.mean()),
            "var_across_zones_max": float(variance_curve.max()),
            "nominal_ranking": result.nominal_ranking,
            "probability_ranked_first": probs,
            "indifference_set_alpha_0.05": sorted(indifferent),
            **retuned,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
