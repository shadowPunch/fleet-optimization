"""The closest methodological pair in the ladder, head to head: does a
genuine indifference set finally show up?

Neither the full B0-B4 comparison (`tuned_ranking_flip_run.py`) nor a
fleet-size sweep of it (`fleet_size_sensitivity_run.py`) found any
indifference beyond the winner itself — B2 (value-corrected) cleanly beat
every other policy at every fleet size tried. But B3 and B4 both wrap the
*same* tuned B2 dispatch logic and differ only in repositioning heuristic
(B3: smooth fluid target allocation from expected arrival rates; B4: one
Monte Carlo sample of near-future demand) — the closest pair in the whole
ladder, and the natural next place to look for genuine indifference before
concluding this project's baseline ladder just doesn't produce it.

Only two policies now, so this affords a larger bootstrap budget than the
five-policy runs for a similarly-sized compute cost (B=50 vs. those runs'
25-40).

Usage: uv run python analysis/b3_vs_b4_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import (
    fit_all_models,
    indifference_set,
    run_ranking_flip_experiment,
    variance_decomposition,
)
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "b3_vs_b4_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 50
N_REPLICATIONS = 4
SEED = 23


def main() -> None:
    tuned = json.loads(TUNING_RESULTS_PATH.read_text())
    b3 = tuned["B3_fluid_balancing"]["best_params"]
    b4 = tuned["B4_sampling_lookahead"]["best_params"]

    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    b3_inner = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=b3["value_weight"], matching_radius_seconds=b3["radius"]
    )
    b4_inner = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=b4["value_weight"], matching_radius_seconds=b4["radius"]
    )
    policies = {
        "B3_fluid_balancing": FluidZoneBalancingPolicy(
            dispatch_policy=b3_inner, arrival_model=models.arrival, day_type="all"
        ),
        "B4_sampling_lookahead": SamplingLookaheadPolicy(
            dispatch_policy=b4_inner, arrival_model=models.arrival, day_type="all",
            lookahead_seconds=b4["lookahead"],
        ),
    }

    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running B3 vs B4: B={N_BOOTSTRAP}, R={N_REPLICATIONS}...")
    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED,
    )

    indifferent = indifference_set(result, alpha=0.05)
    probs = result.probability_ranked_first()
    decomp = variance_decomposition(
        result.metric_by_policy["B3_fluid_balancing"], result.metric_by_policy["B4_sampling_lookahead"]
    )

    output = {
        "nominal_ranking": result.nominal_ranking,
        "probability_ranked_first": probs,
        "indifference_set_alpha_0.05": sorted(indifferent),
        "variance_decomposition": decomp,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    print(f"Nominal ranking: {result.nominal_ranking}")
    print(f"P(ranked first): {probs}")
    print(f"Indifference set (alpha=0.05): {sorted(indifferent)}")
    print(f"Variance decomposition: {decomp}")


if __name__ == "__main__":
    main()
