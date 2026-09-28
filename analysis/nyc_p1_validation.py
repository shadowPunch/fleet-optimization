"""P1 validation against real data — the actually falsifiable version.

**Superseded (V1).** Kept as the record of the failed first validation; the
current validation is `dispatch-eval validate` (`studies/validation.py`),
which fixes the same-zone pickup and boarding defects this run exposed
(TECHNICAL_REPORT.md §5.5).

Per the 2026-08-04 data-plan amendment (project plan §2,
`TECHNICAL_REPORT.md`): NYC TLC data is this project's primary
methodological study, since it's the only source here with real trip-level
timestamps to validate a wait-time prediction against at all. This script
is that validation, not another shape comparison.

**Scope, chosen and stated up front, not discovered after the fact**:

- Operator: HV0003 (Uber) only, matching `analysis/nyc_reference_comparison.py`.
- Geography: Manhattan only (both pickup *and* dropoff), via the TLC's own
  zone lookup table (69 zones) — real Manhattan-only Uber demand on one
  representative day (2024-01-16) is ~143k trips/24h, so this project's
  Python event-driven engine cannot simulate a full day at this volume in
  reasonable time. Restricted to a 6-hour window (12:00-18:00), matching
  the plan's own established "6-hour weekday peak" convention rather than
  inventing a new one.
- Day type: weekdays only, excluding two US federal holidays that fall in
  the window (New Year's Day, MLK Day) since they're not "typical weekday"
  demand. 14 remaining weekdays in January 2024: the first 10
  (2026-01-02..16) are the *calibration* set; the last 4 (01-17..22) are
  *held out* and never touched during fitting — the actual falsification
  target.
- Abandonment hazard: fixed at the pre-registered default
  (`mean_patience_seconds=300.0`), per `TECHNICAL_REPORT.md`'s sweep
  — not fit (fleet size and the abandonment hazard are never jointly
  identified from matched-trip data alone, and NYC HVFHS data only
  contains *matched* trips at all — no cancellation/abandonment record
  exists in this source, a real, structural limit on what it can validate,
  stated here rather than glossed over).

**Pre-registered thresholds this checks against** (`TECHNICAL_REPORT.md`):
KS distance on wait time ≤ 0.10; hour-of-day cosine similarity ≥ 0.90.
Cancellation rate is NOT checked here — NYC data cannot supply it (see
above); that threshold belongs to the separate Bengaluru applicability
study. **Thresholds are not adjusted after seeing the result below.**

Usage: uv run python analysis/nyc_p1_validation.py
(fetches and caches ~14 days of NYC TLC data on first run, no auth needed)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from dispatch_eval.calibration.fleet_size import calibrate_fleet_size, ks_distance
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.nyc_tlc import adapt_nyc_tlc_trips

CACHE_DIR = Path(__file__).parent / "cache"
TRIP_DATA_CACHE = CACHE_DIR / "nyc_manhattan_hv0003_jan2024.parquet"
ZONE_LOOKUP_CACHE = CACHE_DIR / "nyc_taxi_zone_lookup.csv"
OUTPUT_PATH = CACHE_DIR / "nyc_p1_validation_results.json"

TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# 14 weekdays, Jan 2024, excluding New Year's Day (01-01) and MLK Day (01-15).
CALIBRATION_DAYS = [2, 3, 4, 5, 8, 9, 10, 11, 12, 16]
HELD_OUT_DAYS = [17, 18, 19, 22]

WINDOW_START_HOUR = 12
WINDOW_END_HOUR = 18  # 6-hour weekday window, matching the plan's own convention

CANDIDATE_FLEET_SIZES = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6500, 9000, 12500]
MEAN_PATIENCE_SECONDS = 300.0  # pre-registered default, not fit (see module docstring)
N_VALIDATION_REPLICATIONS = 4  # matches len(HELD_OUT_DAYS), for a comparably-sized pooled sample

KS_THRESHOLD = 0.10
COSINE_THRESHOLD = 0.90


def fetch_manhattan_zones() -> list[int]:
    if not ZONE_LOOKUP_CACHE.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zones = pl.read_csv(ZONE_LOOKUP_URL)
        zones.write_csv(ZONE_LOOKUP_CACHE)
    else:
        zones = pl.read_csv(ZONE_LOOKUP_CACHE)
    return zones.filter(pl.col("Borough") == "Manhattan")["LocationID"].to_list()


def fetch_trip_data(manhattan_zones: list[int]) -> pl.DataFrame:
    if TRIP_DATA_CACHE.exists():
        return pl.read_parquet(TRIP_DATA_CACHE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_days = CALIBRATION_DAYS + HELD_OUT_DAYS
    # Window boundary is defined on request_datetime (when the rider asked),
    # not pickup_datetime -- that's the event the arrival model fits and the
    # one the [WINDOW_START_HOUR, WINDOW_END_HOUR) horizon is meant to bound.
    lf = pl.scan_parquet(TRIP_DATA_URL).filter(
        (pl.col("hvfhs_license_num") == "HV0003")
        & (pl.col("PULocationID").is_in(manhattan_zones))
        & (pl.col("DOLocationID").is_in(manhattan_zones))
        & (pl.col("request_datetime").dt.day().is_in(all_days))
        & (pl.col("request_datetime").dt.month() == 1)
        & (pl.col("request_datetime").dt.year() == 2024)
        & (pl.col("request_datetime").dt.hour() >= WINDOW_START_HOUR)
        & (pl.col("request_datetime").dt.hour() < WINDOW_END_HOUR)
    )
    df = lf.collect()
    df.write_parquet(TRIP_DATA_CACHE)
    return df


def hourly_shape(timestamps: pl.Series) -> np.ndarray:
    hours = timestamps.dt.hour().to_numpy() - WINDOW_START_HOUR
    counts = np.bincount(hours, minlength=WINDOW_END_HOUR - WINDOW_START_HOUR)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def main() -> None:
    print("Fetching Manhattan zone list and trip data (cached after first run)...")
    manhattan_zones = fetch_manhattan_zones()
    zones = [str(z) for z in manhattan_zones]
    raw = fetch_trip_data(manhattan_zones)
    trips = adapt_nyc_tlc_trips(raw)
    print(f"  {trips.height} trips, {len(zones)} zones, window {WINDOW_START_HOUR}:00-{WINDOW_END_HOUR}:00")

    calibration = trips.filter(pl.col("request_ts").dt.day().is_in(CALIBRATION_DAYS))
    held_out = trips.filter(pl.col("request_ts").dt.day().is_in(HELD_OUT_DAYS))
    print(f"  calibration: {calibration.height} trips ({len(CALIBRATION_DAYS)} days), "
          f"held-out: {held_out.height} trips ({len(HELD_OUT_DAYS)} days)")

    print("\nFitting input models on the calibration set only...")
    # The engine's simulated clock always starts at midnight (hour 0) --
    # `generate_scenario` samples arrivals over [0, horizon_minutes) and
    # `SimulationEngine._hour_of` wraps to `(time // 3600) % 24` -- but
    # `fit_arrival_model`/`fit_od_model`/`fit_travel_time_model` all bin by
    # each timestamp's *absolute* hour-of-day. Fit directly against real
    # 12:00-18:00 timestamps and every fitted rate/probability/travel-time
    # lands in hour-of-day bins 12-17, which the simulated clock (0-5) never
    # visits -- arrivals would silently generate at rate 0 everywhere. Shift
    # timestamps so window-start (12:00) maps to simulated midnight before
    # fitting; this only affects which bin index a row is fit into, not any
    # duration (wait time, trip time) computed from timestamp differences.
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

    horizon_seconds = (WINDOW_END_HOUR - WINDOW_START_HOUR) * 3600.0
    config = StudyConfig(zones=zones, day_type="all", horizon_seconds=horizon_seconds)

    def simulate_fn(fleet_size: int) -> np.ndarray:
        result = run_simulation(
            fleet_size, models.arrival, models.od, models.travel_time, abandonment_model,
            NearestIdlePolicy(), config, np.random.default_rng(0),
        )
        return result.wait_times

    print(f"\nCalibrating fleet size against the calibration set's own wait-time "
          f"distribution (candidates: {CANDIDATE_FLEET_SIZES})...")
    fleet_result = calibrate_fleet_size(calibration_wait_seconds, simulate_fn, CANDIDATE_FLEET_SIZES)
    print(f"  fleet_size={fleet_result.fleet_size}, loss={fleet_result.loss:.1f}, "
          f"hit_boundary={fleet_result.hit_boundary}")
    if fleet_result.hit_boundary:
        print("  WARNING: calibration hit the edge of the candidate range -- not a converged optimum.")

    print(f"\nSimulating {N_VALIDATION_REPLICATIONS} replications at the calibrated fleet size, "
          f"pooling against the held-out days...")
    sim_wait_times = []
    sim_hourly_shapes = []
    for r in range(N_VALIDATION_REPLICATIONS):
        result = run_simulation(
            fleet_result.fleet_size, models.arrival, models.od, models.travel_time,
            abandonment_model, NearestIdlePolicy(), config, np.random.default_rng(100 + r),
        )
        sim_wait_times.append(result.wait_times)
        pickup_hours = np.array([
            (WINDOW_START_HOUR + req.request_time / 3600.0) for req in result.completed_requests
        ])
        counts = np.bincount(
            (pickup_hours - WINDOW_START_HOUR).astype(int), minlength=WINDOW_END_HOUR - WINDOW_START_HOUR
        )
        sim_hourly_shapes.append(counts)
    simulated_wait_seconds = np.concatenate(sim_wait_times)
    simulated_hourly_counts = np.sum(sim_hourly_shapes, axis=0)
    simulated_hourly_shape = simulated_hourly_counts / simulated_hourly_counts.sum()

    held_out_wait_seconds = (
        (held_out["pickup_ts"] - held_out["request_ts"]).dt.total_seconds().to_numpy()
    )
    held_out_wait_seconds = held_out_wait_seconds[held_out_wait_seconds >= 0]
    held_out_hourly_shape = hourly_shape(held_out["request_ts"])

    ks = ks_distance(held_out_wait_seconds, simulated_wait_seconds)
    cos_sim = cosine_similarity(held_out_hourly_shape, simulated_hourly_shape)

    ks_pass = ks <= KS_THRESHOLD
    cosine_pass = cos_sim >= COSINE_THRESHOLD

    output = {
        "scope": {
            "operator": "HV0003",
            "geography": "Manhattan-only (pickup and dropoff)",
            "window": f"{WINDOW_START_HOUR}:00-{WINDOW_END_HOUR}:00",
            "calibration_days": CALIBRATION_DAYS,
            "held_out_days": HELD_OUT_DAYS,
            "mean_patience_seconds": MEAN_PATIENCE_SECONDS,
        },
        "fleet_calibration": {
            "fleet_size": fleet_result.fleet_size,
            "loss": fleet_result.loss,
            "hit_boundary": fleet_result.hit_boundary,
            "candidates": fleet_result.candidates,
        },
        "wait_time_ks_distance": ks,
        "wait_time_ks_threshold": KS_THRESHOLD,
        "wait_time_ks_pass": ks_pass,
        "hour_of_day_cosine_similarity": cos_sim,
        "hour_of_day_cosine_threshold": COSINE_THRESHOLD,
        "hour_of_day_cosine_pass": cosine_pass,
        "held_out_wait_median_seconds": float(np.median(held_out_wait_seconds)),
        "simulated_wait_median_seconds": float(np.median(simulated_wait_seconds)) if simulated_wait_seconds.size else None,
        "held_out_n_trips": len(held_out_wait_seconds),
        "simulated_n_completed": len(simulated_wait_seconds),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print("P1 VALIDATION RESULT — against pre-registered thresholds, not adjusted after seeing this")
    print(f"{'='*60}")
    print(f"Wait-time KS distance: {ks:.4f} (threshold <= {KS_THRESHOLD}) -> {'PASS' if ks_pass else 'FAIL'}")
    print(f"Hour-of-day cosine similarity: {cos_sim:.4f} (threshold >= {COSINE_THRESHOLD}) -> {'PASS' if cosine_pass else 'FAIL'}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
