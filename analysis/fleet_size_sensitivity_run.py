"""Does a real indifference set emerge at a different fleet size?

`tuned_ranking_flip_run.py` found a clean, fully-separated ranking at
fleet_size=30 (B2 wins every bootstrap draw; the indifference set at
alpha=0.05 contains only the winner itself). That's a legitimate result
for that one scenario, not evidence against the plan's central thesis —
the plan explicitly expects indifference to show up in *some* regime, and
one clean-separation result doesn't rule that out elsewhere. Queueing
intuition suggests two natural directions to check: a tighter fleet (where
every efficient match matters more, plausibly *sharpening* separation) and
a more generous one (where any reasonable policy nearly always has an idle
vehicle nearby, plausibly *compressing* differences toward a floor) — both
are hypotheses to check empirically here, not assumed.

Reuses `tuned_ranking_flip_run.py`'s exact tuned policy configs and data —
only `fleet_size` varies across runs. `n_bootstrap` is reduced from that
script's 40 to 25 to keep a 3-point sweep tractable (~5 minutes total);
this is only looking for *qualitative* change (does the indifference set
gain a second member, does P(ranked first) stop being 1.0), which a
smaller budget can still detect clearly if it's really there.

Usage: uv run python analysis/fleet_size_sensitivity_run.py
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
OUTPUT_PATH = Path(__file__).parent / "cache" / "fleet_size_sensitivity_results.json"

ZONES = ["A", "B", "C", "D", "E"]
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 25
N_REPLICATIONS = 4
SEED = 23
FLEET_SIZES = [15, 30, 60]  # tight / the already-run baseline / generous


def build_tuned_policies(tuned: dict, value_function, arrival_model) -> dict:
    """Same tuned-config construction as `tuned_ranking_flip_run.py` — kept
    as a standalone copy rather than a cross-script import, matching every
    other script in this directory (each is self-contained; there's no
    shared `analysis` package to import from).
    """
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
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)

    results = {}
    for fleet_size in FLEET_SIZES:
        policies = build_tuned_policies(tuned, value_function, models.arrival)
        print(f"\nRunning fleet_size={fleet_size} (B={N_BOOTSTRAP}, R={N_REPLICATIONS})...")
        result = run_ranking_flip_experiment(
            trips_df, ZONES, policies, fleet_size, abandonment_model, config,
            N_BOOTSTRAP, N_REPLICATIONS, SEED,
        )
        indifferent = indifference_set(result, alpha=0.05)
        probs = result.probability_ranked_first()
        results[fleet_size] = {
            "nominal_ranking": result.nominal_ranking,
            "probability_ranked_first": probs,
            "indifference_set_alpha_0.05": sorted(indifferent),
        }
        print(f"  nominal ranking: {result.nominal_ranking}")
        print(f"  P(ranked first): {probs}")
        print(f"  indifference set: {sorted(indifferent)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
