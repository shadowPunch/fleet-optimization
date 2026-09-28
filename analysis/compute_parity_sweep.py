"""C4 — empirical decision-latency sweep across the B0-B4 baseline ladder.

No real data needed — this
uses synthetic-but-representative fitted models (same pattern as every
other test in this project) purely to build well-formed policy instances
(a real `ValueFunction`, a real `NHPPArrivalModel` with nonzero rates for
B4's sampled lookahead to actually do something), then measures wall-clock
`dispatch()`/`reposition()` time on synthetic Request/Vehicle objects sized
to a chosen sweep of problem sizes — exactly the measurement
`compute_parity.py` is built for.

See that module's docstring for what this script does *not* attempt: a
real "truncate the candidate graph" degradation mechanism doesn't exist in
this codebase yet (B1/B2's `matching_radius_seconds` masks costs rather
than shrinking the solved matrix, confirmed in
`tests/test_compute_parity.py`), so this is the measurement half of C4
only, not the "degrade + re-measure quality" half.

Usage: uv run python analysis/compute_parity_sweep.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.compute_parity import feasible_within_budget, latency_frontier
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.sources.synthetic import generate_synthetic_trips

OUTPUT_PATH = Path(__file__).parent / "cache" / "compute_parity_results.json"

ZONES = ["A", "B", "C", "D", "E"]
PROBLEM_SIZES = [(10, 10), (50, 50), (100, 100), (300, 300), (600, 600)]
BUDGETS_MS = (100, 500, 1000, 2000)


def build_policies() -> dict:
    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=100, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    b2 = ValueCorrectedHungarianPolicy(
        value_function=value_function, value_weight=1.0, matching_radius_seconds=900.0
    )
    policies = {
        "B0_nearest_idle": NearestIdlePolicy(),
        "B1_batched_hungarian": BatchedHungarianPolicy(matching_radius_seconds=900.0),
        "B2_value_corrected": b2,
        "B3_fluid_balancing": FluidZoneBalancingPolicy(
            dispatch_policy=b2, arrival_model=models.arrival, day_type="all"
        ),
        "B4_sampling_lookahead": SamplingLookaheadPolicy(
            dispatch_policy=b2, arrival_model=models.arrival, day_type="all", lookahead_seconds=300.0
        ),
    }
    return policies, models.travel_time


def main() -> None:
    policies, travel_time_model = build_policies()
    frontier = latency_frontier(
        policies, PROBLEM_SIZES, ZONES, travel_time_model, seed=0, n_repeats=5
    )

    results = {
        "problem_sizes": PROBLEM_SIZES,
        "latency_ms": {
            name: [m.total_seconds * 1000.0 for m in ms] for name, ms in frontier.items()
        },
        "feasible_within_budget": {
            f"{budget_ms}ms": feasible_within_budget(frontier, budget_seconds=budget_ms / 1000.0)
            for budget_ms in BUDGETS_MS
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUTPUT_PATH}")

    header = f"{'policy':<24}" + "".join(f"{f'{n}x{v}':>12}" for n, v in PROBLEM_SIZES)
    print(header)
    for name, ms in frontier.items():
        row = f"{name:<24}" + "".join(f"{m.total_seconds * 1000:>10.2f}ms" for m in ms)
        print(row)

    print()
    for budget_ms in BUDGETS_MS:
        feasible = feasible_within_budget(frontier, budget_seconds=budget_ms / 1000.0)
        first_infeasible = {
            name: PROBLEM_SIZES[flags.index(False)] if False in flags else None
            for name, flags in feasible.items()
        }
        print(f"budget={budget_ms}ms, first infeasible problem size per policy: {first_infeasible}")


if __name__ == "__main__":
    main()
