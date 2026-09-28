"""Follow-up flagged in `TECHNICAL_REPORT.md`, not a redo of P1 itself:
does calibrating fleet size to the real P90/median *ratio* (shape) instead
of absolute median+p90 close any of the KS gap?

The pre-registered P1 result (`analysis/nyc_p1_validation.py`) stands
exactly as reported — this script doesn't touch it, doesn't re-run it
against the held-out thresholds, and isn't a second attempt at "passing."
It answers a narrower, already-flagged question: was the FAIL partly an
artifact of `absolute_moment_loss` asking the fleet-size search to match
two things (scale and shape) that can't both be matched at once, when
matching only the shape might have gotten closer? Reuses the exact same
cached data and fitted models `nyc_p1_validation.py` produced.

Usage: uv run python analysis/nyc_p1_ratio_calibration_check.py
(requires analysis/cache/nyc_manhattan_hv0003_jan2024.parquet already
fetched by analysis/nyc_p1_validation.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from dispatch_eval.calibration.fleet_size import calibrate_fleet_size, ks_distance, ratio_loss
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.nyc_tlc import adapt_nyc_tlc_trips

CACHE_DIR = Path(__file__).parent / "cache"
TRIP_DATA_CACHE = CACHE_DIR / "nyc_manhattan_hv0003_jan2024.parquet"
ZONE_LOOKUP_CACHE = CACHE_DIR / "nyc_taxi_zone_lookup.csv"
OUTPUT_PATH = CACHE_DIR / "nyc_p1_ratio_calibration_check_results.json"

CALIBRATION_DAYS = [2, 3, 4, 5, 8, 9, 10, 11, 12, 16]
HELD_OUT_DAYS = [17, 18, 19, 22]
WINDOW_START_HOUR = 12
WINDOW_END_HOUR = 18

CANDIDATE_FLEET_SIZES = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6500, 9000, 12500]
MEAN_PATIENCE_SECONDS = 300.0
N_VALIDATION_REPLICATIONS = 4
KS_THRESHOLD = 0.10


def main() -> None:
    if not TRIP_DATA_CACHE.exists() or not ZONE_LOOKUP_CACHE.exists():
        raise SystemExit(
            f"{TRIP_DATA_CACHE} / {ZONE_LOOKUP_CACHE} not found -- "
            "run analysis/nyc_p1_validation.py first to fetch them."
        )
    raw = pl.read_parquet(TRIP_DATA_CACHE)
    trips = adapt_nyc_tlc_trips(raw)
    # Same 69-zone Manhattan list nyc_p1_validation.py used -- not just the
    # zones that happen to appear in this slice, so fleet placement and
    # simulation dynamics match exactly, not a subtly smaller geography.
    zone_lookup = pl.read_csv(ZONE_LOOKUP_CACHE)
    zones = [str(z) for z in zone_lookup.filter(pl.col("Borough") == "Manhattan")["LocationID"].to_list()]

    calibration = trips.filter(pl.col("request_ts").dt.day().is_in(CALIBRATION_DAYS))
    held_out = trips.filter(pl.col("request_ts").dt.day().is_in(HELD_OUT_DAYS))

    window_offset = pl.duration(hours=WINDOW_START_HOUR)
    calibration_for_fit = calibration.with_columns(
        (pl.col("request_ts") - window_offset).alias("request_ts"),
        (pl.col("pickup_ts") - window_offset).alias("pickup_ts"),
        (pl.col("dropoff_ts") - window_offset).alias("dropoff_ts"),
    )
    models = fit_all_models(calibration_for_fit, bin_minutes=15, od_time_bin_minutes=60)
    abandonment_model = AbandonmentModel(mean_patience_seconds=MEAN_PATIENCE_SECONDS)

    calibration_wait_seconds = (
        (calibration["pickup_ts"] - calibration["request_ts"]).dt.total_seconds().to_numpy()
    )
    calibration_wait_seconds = calibration_wait_seconds[calibration_wait_seconds >= 0]
    obs_median = float(np.median(calibration_wait_seconds))
    obs_p90 = float(np.percentile(calibration_wait_seconds, 90))
    print(f"Calibration set: median={obs_median:.1f}s, p90={obs_p90:.1f}s, ratio={obs_p90 / obs_median:.3f}")

    horizon_seconds = (WINDOW_END_HOUR - WINDOW_START_HOUR) * 3600.0
    config = StudyConfig(zones=zones, day_type="all", horizon_seconds=horizon_seconds)

    def simulate_fn(fleet_size: int) -> np.ndarray:
        result = run_simulation(
            fleet_size, models.arrival, models.od, models.travel_time, abandonment_model,
            NearestIdlePolicy(), config, np.random.default_rng(0),
        )
        return result.wait_times

    print(f"\nRecalibrating with ratio_loss (candidates: {CANDIDATE_FLEET_SIZES})...")
    ratio_result = calibrate_fleet_size(
        calibration_wait_seconds, simulate_fn, CANDIDATE_FLEET_SIZES, loss_fn=ratio_loss
    )
    print(f"  ratio-calibrated fleet_size={ratio_result.fleet_size}, loss={ratio_result.loss:.6f}, "
          f"hit_boundary={ratio_result.hit_boundary}")
    print("  (for comparison, the absolute-loss calibration in TECHNICAL_REPORT.md §5.5 picked fleet_size=4500)")

    held_out_wait_seconds = (
        (held_out["pickup_ts"] - held_out["request_ts"]).dt.total_seconds().to_numpy()
    )
    held_out_wait_seconds = held_out_wait_seconds[held_out_wait_seconds >= 0]

    print(f"\nSimulating {N_VALIDATION_REPLICATIONS} replications at the ratio-calibrated fleet size...")
    sim_wait_times = [
        run_simulation(
            ratio_result.fleet_size, models.arrival, models.od, models.travel_time,
            abandonment_model, NearestIdlePolicy(), config, np.random.default_rng(100 + r),
        ).wait_times
        for r in range(N_VALIDATION_REPLICATIONS)
    ]
    simulated_wait_seconds = np.concatenate(sim_wait_times)
    sim_median = float(np.median(simulated_wait_seconds)) if simulated_wait_seconds.size else float("nan")
    sim_p90 = float(np.percentile(simulated_wait_seconds, 90)) if simulated_wait_seconds.size else float("nan")

    ks = ks_distance(held_out_wait_seconds, simulated_wait_seconds)
    ks_pass = ks <= KS_THRESHOLD

    output = {
        "ratio_calibrated_fleet_size": ratio_result.fleet_size,
        "ratio_calibrated_hit_boundary": ratio_result.hit_boundary,
        "ratio_calibration_candidates": ratio_result.candidates,
        "held_out_wait_median_seconds": float(np.median(held_out_wait_seconds)),
        "held_out_wait_p90_seconds": float(np.percentile(held_out_wait_seconds, 90)),
        "simulated_wait_median_seconds": sim_median,
        "simulated_wait_p90_seconds": sim_p90,
        "wait_time_ks_distance": ks,
        "wait_time_ks_threshold": KS_THRESHOLD,
        "wait_time_ks_pass": ks_pass,
        "prior_absolute_loss_ks_distance": 0.7355,  # TECHNICAL_REPORT.md, for direct comparison
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*70}")
    print("RATIO-LOSS CALIBRATION CHECK (follow-up, not a P1 re-attempt)")
    print(f"{'='*70}")
    print(f"  simulated: median={sim_median:.1f}s, p90={sim_p90:.1f}s, "
          f"ratio={(sim_p90 / sim_median if sim_median else float('nan')):.3f}")
    print(f"  held-out real: median={np.median(held_out_wait_seconds):.1f}s, "
          f"p90={np.percentile(held_out_wait_seconds, 90):.1f}s, "
          f"ratio={np.percentile(held_out_wait_seconds, 90) / np.median(held_out_wait_seconds):.3f}")
    print(f"  KS distance: {ks:.4f} (threshold <= {KS_THRESHOLD}) -> {'PASS' if ks_pass else 'FAIL'}")
    print("  (prior absolute-loss KS distance was 0.7355)")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
