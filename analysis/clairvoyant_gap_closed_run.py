"""B5 wired in for real: every tuned B0-B4 policy's performance expressed
as fraction of the clairvoyant gap closed (the plan's own normalization,
B5 section: "lets you report every policy as fraction of the clairvoyant
gap closed, which normalises across simulators and cities in a way that
'3.2% better than greedy' does not").

Reuses the same tuned policy configs as `tuned_ranking_flip_run.py`, but
with `compute_clairvoyant=True` and a smaller bootstrap budget (B5's
min-cost-flow solve, once per (b, r), is real added cost on top of the
usual B x R x |policies| simulation runs).

`clairvoyant_bin_minutes=1.0` here, not the 15.0 default: at 15 minutes,
this project's mean_patience_seconds=300s (5 minutes, a third of one bin)
caused the solver to drop the large majority of requests as "unservable"
purely from rounding, before ever seeing them — see clairvoyant.py's
module docstring for the real, measured exclusion rate this caused.
1.0 is tractable here specifically because this project's analysis
scripts use 5 zones; the reposition-edge count scales as
zones^2 * horizon_seconds/bin_seconds, so this would need revisiting at
realistic ward counts. `clairvoyant_fraction_excluded` is reported below
so this isn't silently trusted either.

Usage: uv run python analysis/clairvoyant_gap_closed_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models, run_ranking_flip_experiment
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "clairvoyant_gap_closed_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 15
N_REPLICATIONS = 3
SEED = 23
BASELINE_POLICY = "B0_nearest_idle"


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
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running B0-B4 with compute_clairvoyant=True: B={N_BOOTSTRAP}, R={N_REPLICATIONS}...")
    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED, compute_clairvoyant=True,
        clairvoyant_bin_minutes=1.0,
    )

    gap_closed = result.fraction_of_gap_closed(BASELINE_POLICY)
    summary = {
        name: {
            "mean_fraction_of_gap_closed": float(arr.mean()),
            "median_fraction_of_gap_closed": float(np.median(arr)),
        }
        for name, arr in gap_closed.items()
    }
    output = {
        "baseline_policy": BASELINE_POLICY,
        "clairvoyant_mean_wait_seconds": float(result.clairvoyant_metric_by_draw.mean()),
        "clairvoyant_fraction_excluded_by_discretization": float(
            result.clairvoyant_fraction_excluded.mean()
        ),
        "per_policy_mean_wait_seconds": {
            name: float(arr.mean()) for name, arr in result.metric_by_policy.items()
        },
        "fraction_of_gap_closed": summary,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    print(f"Clairvoyant mean wait: {output['clairvoyant_mean_wait_seconds']:.1f}s")
    print(
        "Clairvoyant fraction excluded by discretization (should be near zero): "
        f"{output['clairvoyant_fraction_excluded_by_discretization']:.3f}"
    )
    for name, s in summary.items():
        print(
            f"{name}: mean_wait={output['per_policy_mean_wait_seconds'][name]:.1f}s, "
            f"fraction_of_gap_closed={s['mean_fraction_of_gap_closed']:.3f}"
        )


if __name__ == "__main__":
    main()
