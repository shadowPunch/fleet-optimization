"""Does the B0-B4 ladder's clean separation survive giving policies a
realistic (imperfect) forecast instead of an oracle one?

Every prior rigorous comparison in this project (`TECHNICAL_DOCUMENTATION.md`)
found B2 cleanly winning, no indifference between mechanisms — but every
one of those runs let B2-B4 forecast the world through models fit the
*same way*, on the *same kind of data*, as the model that generated the
true world. This sweep is the check: hold the true world fixed at a
correctly-specified, bootstrap-varying fit (as every other P3 run in this
project already does) and progressively degrade what the *policies*
believe, via `forecast_degradation.run_degradation_sweep`. If separation
collapses as the forecast degrades, the earlier "no indifference" finding
was (at least partly) an oracle-forecast artifact, not evidence that these
mechanisms genuinely differ this much.

Levels: the true (undegraded) forecast as a baseline; three data-limited
levels (fit the policy's belief on only 1, 3, or 5 of the dataset's days);
each of the four structural degradations in isolation (homogeneous
arrival — flattens time-of-day shape only; spatially uniform arrival —
strictly more severe, flattens *which zone is busier* too; marginal OD;
flat travel time), to see which one matters most; and a "fully degraded"
level combining data-limiting with every structural degradation — the
closest analog to a real deployment's crude, hand-built forecast.
`spatially_uniform_arrival` was added after the first pass at this sweep
found zero erosion from the other three in isolation: none of them touch
*spatial* signal (which zone is relatively busier), only temporal/
conditional/variance detail, leaving open whether that's what the
ladder's separation actually runs on. See TECHNICAL_DOCUMENTATION.md.

Usage: uv run python analysis/forecast_degradation_sweep.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.forecast_degradation import DegradationLevel, run_degradation_sweep
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.ranking_flip import indifference_set
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "forecast_degradation_sweep_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 25
N_REPLICATIONS = 4
SEED = 23
N_DAYS_TOTAL = 10

LEVELS = [
    DegradationLevel("true_forecast"),
    DegradationLevel("n_days_1", n_days=1),
    DegradationLevel("n_days_3", n_days=3),
    DegradationLevel("n_days_5", n_days=5),
    DegradationLevel("homogeneous_arrival_only", homogenize_arrival=True),
    DegradationLevel("spatially_uniform_arrival_only", spatially_uniform_arrival=True),
    DegradationLevel("marginal_od_only", marginalize_od=True),
    DegradationLevel("flat_travel_time_only", flatten_travel_time=True),
    DegradationLevel(
        "fully_degraded", n_days=1, spatially_uniform_arrival=True, marginalize_od=True,
        flatten_travel_time=True,
    ),
]


def main() -> None:
    tuned_params = json.loads(TUNING_RESULTS_PATH.read_text())
    trips_df = generate_synthetic_trips(
        n_days=N_DAYS_TOTAL, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running the forecast-degradation sweep ({len(LEVELS)} levels)...")
    results = run_degradation_sweep(
        trips_df, ZONES, tuned_params, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED, LEVELS,
    )

    summary = {}
    for name, result in results.items():
        indifferent = indifference_set(result, alpha=0.05)
        probs = result.probability_ranked_first()
        summary[name] = {
            "nominal_ranking": result.nominal_ranking,
            "probability_ranked_first": probs,
            "indifference_set_alpha_0.05": sorted(indifferent),
            "indifference_set_size": len(indifferent),
        }
        print(f"\n=== {name} ===")
        print(f"  nominal ranking: {result.nominal_ranking}")
        print(f"  P(ranked first): {probs}")
        print(f"  indifference set: {sorted(indifferent)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
