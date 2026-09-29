"""C3 end to end, against the real confirmatory bootstrap study.

Runs the exact same pipeline as `run_study.py --n-bootstrap 200
--n-replications 5` (the plan's target-scale confirmatory run, recorded in
`results/study_results_at_scale_v2.parquet`) in-memory via
`run_study.run_pipeline`, then converts every policy's bootstrap-draw wait
time into "worth N vehicles relative to B0" via `decision_currency` —
`run_study.py`'s own parquet only stores per-cell wait times, not the live
`RankingFlipResult`/`FittedModels` C3 needs to invert a fleet-wait curve,
so this reruns the pipeline rather than reading the parquet back.

This has never been run against real data before (`TECHNICAL_DOCUMENTATION.md`:
the NYC twin fails its own pre-registered wait-time KS threshold), so this
is a synthetic-data confirmatory result, not a real-Manhattan one — stated
here rather than implied.

Usage: uv run python analysis/decision_currency_run.py (~26 min, B=200/R=5)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dispatch_eval.decision_currency import (
    build_fleet_wait_curve,
    decision_currency,
    mean_fare_per_trip,
    trips_per_vehicle_per_day,
)
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

# run_study.py is a root-level CLI entrypoint, not part of the installed
# dispatch_eval package -- its directory isn't on sys.path when this script
# is invoked as `python analysis/decision_currency_run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_study import BASELINE_POLICY, HORIZON_SECONDS, ZONES, run_pipeline

N_BOOTSTRAP = 200
N_REPLICATIONS = 5
SEED = 2026
# Brackets every policy's wait time without heavy extrapolation: B0 is the
# worst policy in the ladder, so matching a *better* policy's wait time
# means asking "how many more vehicles would B0 need," well above its own
# calibrated fleet size -- the sweep must extend past it, not center on it.
REFERENCE_SWEEP_FLEET_SIZES = [20, 30, 40, 50, 60, 75, 90, 110, 130, 160, 200, 250, 300, 350]
OUTPUT_PATH = Path(__file__).parent / "cache" / "decision_currency_results.json"


def main() -> None:
    print("Running the confirmatory pipeline (B=200, R=5 -- matches "
          "results/study_results_at_scale_v2.parquet)...")
    result, models, abandonment_model, fleet_calibration, trips_df = run_pipeline(
        N_BOOTSTRAP, N_REPLICATIONS, SEED
    )

    print(f"\nBuilding B0's fleet-size -> wait-time curve ({len(REFERENCE_SWEEP_FLEET_SIZES)} points)...")
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    reference_curve = build_fleet_wait_curve(
        REFERENCE_SWEEP_FLEET_SIZES, models, abandonment_model, NearestIdlePolicy(), config,
        n_replications=N_REPLICATIONS, seed=SEED,
    )

    # B0's own completed-trip rate at the *actual* calibrated fleet size,
    # for the Delta$/day approximation (module docstring: no fare on
    # simulated Request objects, so this falls back to a flat rate).
    completed_counts = []
    for r in range(N_REPLICATIONS):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, fleet_calibration.fleet_size, r]))
        sim = run_simulation(
            fleet_calibration.fleet_size, models.arrival, models.od, models.travel_time,
            abandonment_model, NearestIdlePolicy(), config, rng,
        )
        completed_counts.append(len(sim.completed_requests))
    horizon_hours = HORIZON_SECONDS / 3600.0
    ref_trips_per_vehicle_per_day = trips_per_vehicle_per_day(
        int(np.mean(completed_counts)), fleet_calibration.fleet_size, horizon_hours
    )
    dollars_per_trip = mean_fare_per_trip(trips_df)

    print(f"  reference (B0) trips/vehicle/day: {ref_trips_per_vehicle_per_day:.2f}, "
          f"$/trip: {dollars_per_trip:.2f}")

    currency = decision_currency(
        result, BASELINE_POLICY, reference_curve, fleet_calibration.fleet_size, horizon_hours,
        dollars_per_trip, ref_trips_per_vehicle_per_day,
    )

    print(f"\n{'='*70}")
    print(f"DECISION CURRENCY (reference: {BASELINE_POLICY}, actual fleet_size={fleet_calibration.fleet_size})")
    print(f"{'='*70}")
    summaries = {}
    for name, dc in currency.items():
        s = dc.summary()
        summaries[name] = s
        print(f"\n{name}:")
        print(f"  worth {s['vehicles_worth_mean']:+.1f} vehicles, 95% CI "
              f"[{s['vehicles_worth_ci'][0]:+.1f}, {s['vehicles_worth_ci'][1]:+.1f}]")
        print(f"  Delta driver-hours/day: {s['delta_driver_hours_mean']:+.1f}")
        print(f"  Delta $/day: {s['delta_dollars_per_day_mean']:+.2f}")
        print(f"  fraction of draws extrapolated: {s['fraction_draws_extrapolated']:.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "reference_policy": BASELINE_POLICY,
        "actual_fleet_size": fleet_calibration.fleet_size,
        "reference_sweep_fleet_sizes": REFERENCE_SWEEP_FLEET_SIZES,
        "reference_curve_wait_seconds": reference_curve.mean_wait_seconds,
        "data_source": "synthetic -- see TECHNICAL_DOCUMENTATION.md for why not real NYC data",
        "summaries": summaries,
    }, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
