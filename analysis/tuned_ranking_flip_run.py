"""A rigorous B0-B4 comparison using the tuned parameters from
`policy_tuning_run.py`, instead of that script's own single-seed
"best_score" numbers (explicitly flagged there as not a rigorous
comparison). This is the follow-up promised in that script's docstring and
in the README: plug the tuned configs into an actual bootstrap-CRN
`ranking_flip.py` experiment, which *is* built for comparing policies
against each other properly.

**A real methodological wrinkle, found while building this, not assumed
going in**: `policy_tuning_run.py` tuned each policy's own
`dispatch_interval_seconds` (Δ) independently, since its evaluate_fn built
a fresh `StudyConfig` per candidate. But `run_ranking_flip_experiment`
takes *one* `StudyConfig` shared by every policy in the comparison — Δ is
an engine parameter, not a policy one (see `policies/batched_hungarian.py`'s
own docstring), and CRN pairing requires every policy to see the same
dispatch-tick schedule. The tuned Δ values split into two clusters (B1/B4
~5.8s, B2/B3 ~28-31s) — there is no free-lunch shared choice here. This
script uses `StudyConfig`'s own project-wide default (5.0s) rather than
introduce a new arbitrary number, and reports this tension explicitly
rather than picking one policy's preferred Δ and silently favoring it.

Each policy's other, genuinely policy-owned tuned parameters
(`matching_radius_seconds`, `value_weight`, `lookahead_seconds`) are used
as tuned.

Usage: uv run python analysis/tuned_ranking_flip_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models, indifference_set, run_ranking_flip_experiment
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "tuned_ranking_flip_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 40
N_REPLICATIONS = 4
SEED = 23


def build_tuned_policies(tuned: dict, value_function, arrival_model) -> dict:
    b1 = tuned["B1_batched_hungarian"]["best_params"]
    b2 = tuned["B2_value_corrected"]["best_params"]
    b3 = tuned["B3_fluid_balancing"]["best_params"]
    b4 = tuned["B4_sampling_lookahead"]["best_params"]

    b2_inner = ValueCorrectedHungarianPolicy(
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
        "B2_value_corrected": b2_inner,
        "B3_fluid_balancing": FluidZoneBalancingPolicy(
            dispatch_policy=b3_inner, arrival_model=arrival_model, day_type="all"
        ),
        "B4_sampling_lookahead": SamplingLookaheadPolicy(
            dispatch_policy=b4_inner, arrival_model=arrival_model, day_type="all",
            lookahead_seconds=b4["lookahead"],
        ),
    }


def main() -> None:
    tuned = json.loads(TUNING_RESULTS_PATH.read_text())

    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    policies = build_tuned_policies(tuned, value_function, models.arrival)

    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)  # default delta=5.0s
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running rigorous B0-B4 comparison: B={N_BOOTSTRAP}, R={N_REPLICATIONS}...")
    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED,
    )

    indifferent = indifference_set(result, alpha=0.05)
    probs = result.probability_ranked_first()

    output = {
        "nominal_ranking": result.nominal_ranking,
        "probability_ranked_first": probs,
        "indifference_set_alpha_0.05": sorted(indifferent),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    print(f"Nominal ranking (best first): {result.nominal_ranking}")
    print(f"P(ranked first): {probs}")
    print(f"Indifference set (alpha=0.05): {sorted(indifferent)}")


if __name__ == "__main__":
    main()
