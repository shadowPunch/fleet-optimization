"""A within-mechanism sweep: B2 (value-corrected Hungarian) at several
`value_weight` settings, compared against *each other* — not against a
different dispatch mechanism, unlike every comparison in
`TECHNICAL_REPORT.md`, all of which found no indifference.

The hypothesis this checks: indifference might not show up between
different *mechanisms* (already checked, four times, always no), but
could plausibly show up between close settings of the *same* mechanism's
own strength dial — e.g. is `value_weight=0.75` really distinguishable
from `value_weight=1.0`, or does the correction saturate once it's doing
*something*? `matching_radius_seconds` is held fixed at B2's own tuned
value throughout, so `value_weight` is the only thing that varies —
a clean, single-factor comparison.

Usage: uv run python analysis/value_weight_sweep_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models, indifference_set, run_ranking_flip_experiment
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "value_weight_sweep_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 30
N_REPLICATIONS = 4
SEED = 23
VALUE_WEIGHTS = [0.0, 0.5, 0.75, 1.0, 1.5]  # 0.0 = no value correction at all


def main() -> None:
    tuned = json.loads(TUNING_RESULTS_PATH.read_text())
    radius = tuned["B2_value_corrected"]["best_params"]["radius"]  # held fixed throughout

    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    policies = {
        f"weight_{w}": ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=w, matching_radius_seconds=radius
        )
        for w in VALUE_WEIGHTS
    }

    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running value_weight sweep {VALUE_WEIGHTS}: B={N_BOOTSTRAP}, R={N_REPLICATIONS}...")
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

    print(f"Nominal ranking: {result.nominal_ranking}")
    print(f"P(ranked first): {probs}")
    print(f"Indifference set (alpha=0.05): {sorted(indifferent)}")


if __name__ == "__main__":
    main()
