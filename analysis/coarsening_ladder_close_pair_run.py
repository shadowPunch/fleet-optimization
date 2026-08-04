"""C6's sharper follow-up, now that one exists: does data coarsening flip
a *genuinely close* pair, not just a large-effect-size one?

`coarsening_ladder_sweep.py` (the first C6 run) compared B0 vs. B1 and
found the ranking survived every rung — expected, since that pair sits
well outside C2's own indifference zone even at full resolution
(`docs/coarsening_ladder.md` flagged this as not a strong test).
`value_weight_sweep_run.py` then found a real indifference set:
`ValueCorrectedHungarianPolicy` at `value_weight` in
`{0.5, 0.75, 1.0, 1.5}` are all statistically indistinguishable from each
other (`docs/indifference_search.md`). This script re-runs the exact same
coarsening ladder on `weight_0.5` vs. `weight_1.5` — the two most distant
members of that indifference set, giving coarsening the best chance to
actually separate them (in either direction) if it's going to.

Usage: uv run python analysis/coarsening_ladder_close_pair_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.coarsening_ladder import ranking_shift_summary, run_coarsening_ladder
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "coarsening_ladder_close_pair_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 20
N_REPLICATIONS = 4
SEED = 23


def main() -> None:
    tuned = json.loads(TUNING_RESULTS_PATH.read_text())
    radius = tuned["B2_value_corrected"]["best_params"]["radius"]

    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    policies = {
        "weight_0.5": ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=0.5, matching_radius_seconds=radius
        ),
        "weight_1.5": ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=1.5, matching_radius_seconds=radius
        ),
    }

    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print("Running the coarsening ladder on a genuinely close pair (weight_0.5 vs weight_1.5)...")
    results = run_coarsening_ladder(
        trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED,
    )
    summary = ranking_shift_summary(results, baseline_rung="rung0_fine_grained")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    for rung_name, s in summary.items():
        print(f"=== {rung_name} ===")
        print(f"  nominal ranking: {s['nominal_ranking']}")
        print(f"  Kendall tau to rung0: {s['kendall_tau_to_baseline']:.3f}")
        print(f"  P(ranked first): {s['probability_ranked_first']}")
        print()


if __name__ == "__main__":
    main()
