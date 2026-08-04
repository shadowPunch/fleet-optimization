"""C6 — real coarsening-ladder run: how much of the B0-vs-B1 ranking
survives realistic data loss?

No real data needed (see README) — this re-runs C2's exact bootstrap-
ranking-flip experiment on the project's own synthetic data, progressively
coarsened toward what Bengaluru's real published data actually offers (see
`coarsening_ladder.py`'s docstring for the exact rung definitions).

Limited to B0 (greedy nearest-idle) vs. B1 (batched Hungarian) rather than
the full B0-B4 ladder: B2-B4 each carry a value function or arrival-model
reference fitted once, outside this experiment, and reused unchanged
across every rung — consistent with how `ranking_flip.py` already treats
fleet size, but a second scope decision worth its own dedicated pass
rather than folding into this one. B0 vs. B1 needs no such object and
isolates the question this script asks: does *input-model* coarsening
alone flip a ranking.

Usage: uv run python analysis/coarsening_ladder_sweep.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.coarsening_ladder import ranking_shift_summary, run_coarsening_ladder
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

OUTPUT_PATH = Path(__file__).parent / "cache" / "coarsening_ladder_results.json"

ZONES = ["A", "B", "C", "D", "E"]


def main() -> None:
    trips_df = generate_synthetic_trips(
        n_days=14, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=7
    )
    policies = {"B0_nearest_idle": NearestIdlePolicy(), "B1_batched_hungarian": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print("Running the coarsening ladder (4 rungs x bootstrap-CRN experiment each)...")
    results = run_coarsening_ladder(
        trips_df,
        ZONES,
        policies,
        fleet_size=30,
        abandonment_model=abandonment_model,
        config=config,
        n_bootstrap=20,
        n_replications=4,
        seed=13,
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
