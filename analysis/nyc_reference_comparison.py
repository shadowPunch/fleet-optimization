"""Compare the synthetic trip generator against real NYC TLC data.

Not a data-source change for the project — there's still no public,
downloadable, trip-level dataset for any Indian city (see the project plan
and README). This is a validation exercise: NYC TLC data is real, richly
fielded, and immediately downloadable with no credentials, which makes it
useful as a reference for asking "is our synthetic generator's *shape* — not
its absolute numbers, which are arbitrary and unitless by construction —
comparable to a real ride-hailing market, and can the differences be
explained?"

Requires network access to
https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet
(no auth). Run once to fetch+cache the day of data this script uses, then it
reads from the cache on subsequent runs.

Usage: uv run python analysis/nyc_reference_comparison.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import skew

from dispatch_eval.calibration.fare import fit_fare_model
from dispatch_eval.calibration.travel_time import fit_travel_time_model
from dispatch_eval.sources.nyc_tlc import adapt_nyc_tlc_trips
from dispatch_eval.sources.synthetic import generate_synthetic_trips

CACHE_PATH = Path(__file__).parent / "cache" / "nyc_uber_2024-01-16.parquet"
NYC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet"
OUTPUT_PATH = Path(__file__).parent / "cache" / "comparison_results.json"


def fetch_nyc_sample() -> pl.DataFrame:
    if CACHE_PATH.exists():
        return pl.read_parquet(CACHE_PATH)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(NYC_URL).filter(
        (pl.col("pickup_datetime") >= pl.datetime(2024, 1, 16))
        & (pl.col("pickup_datetime") < pl.datetime(2024, 1, 17))
        & (pl.col("hvfhs_license_num") == "HV0003")
    )
    df = lf.collect()
    df.write_parquet(CACHE_PATH)
    return df


def hourly_shape(timestamps: pl.Series) -> list[float]:
    hours = timestamps.dt.hour()
    counts = np.bincount(hours.to_numpy(), minlength=24)
    return (counts / counts.sum()).tolist()


def describe_trips(trips: pl.DataFrame, label: str) -> dict:
    wait_s = (trips["pickup_ts"] - trips["request_ts"]).dt.total_seconds()
    wait_s_valid = wait_s.filter(wait_s >= 0)  # see note on negative waits below
    dur_s = (trips["dropoff_ts"] - trips["pickup_ts"]).dt.total_seconds()

    fare_model = fit_fare_model(trips)
    work = trips.select(["trip_distance_km", "pickup_ts", "dropoff_ts", "fare"]).drop_nulls()
    dur_min = (work["dropoff_ts"] - work["pickup_ts"]).dt.total_seconds() / 60.0
    pred = (
        fare_model.intercept
        + fare_model.distance_coef * work["trip_distance_km"]
        + fare_model.duration_coef * dur_min
    )
    actual = work["fare"].to_numpy()
    ss_res = float(np.sum((actual - pred.to_numpy()) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    fare_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    tt_model = fit_travel_time_model(trips)
    sigmas = [s for (_, s) in tt_model.params.values()]

    # Cell sparsity: fit_travel_time_model falls back to a pooled global
    # sigma for any (origin, dest, hour) cell with fewer than 5 samples —
    # and pooling across cells with different means inflates the apparent
    # sigma above any single cell's true value. Worth knowing how much of
    # the "travel_time_sigma" figure above is actually the fallback talking.
    cell_counts = (
        trips.select(["origin_zone", "dest_zone", "pickup_ts"])
        .with_columns(pl.col("pickup_ts").dt.hour().alias("hour"))
        .group_by(["origin_zone", "dest_zone", "hour"])
        .agg(pl.len().alias("n"))
    )
    well_sampled = cell_counts.filter(pl.col("n") >= 5)

    zone_counts = trips.group_by("origin_zone").agg(pl.len().alias("n")).sort("n", descending=True)
    shares = zone_counts["n"].to_numpy() / zone_counts["n"].sum()
    hhi = float(np.sum(shares**2))

    valid_ratio = trips.filter(pl.col("fare") > 1.0).select(
        (pl.col("driver_pay") / pl.col("fare")).alias("ratio")
    )["ratio"]

    distance = trips["trip_distance_km"].drop_nulls().to_numpy()

    return {
        "label": label,
        "n_trips": trips.height,
        "n_zones": trips["origin_zone"].n_unique(),
        "hourly_shape": hourly_shape(trips["request_ts"]),
        "wait_seconds": {
            "mean": float(wait_s_valid.mean()),
            "median": float(wait_s_valid.median()),
            "p10": float(wait_s_valid.quantile(0.10)),
            "p90": float(wait_s_valid.quantile(0.90)),
            "p99": float(wait_s_valid.quantile(0.99)),
            "fraction_negative": float((wait_s < 0).sum() / wait_s.len()),
        },
        "trip_distance_km": {
            "mean": float(trips["trip_distance_km"].mean()),
            "median": float(trips["trip_distance_km"].median()),
            "p90": float(trips["trip_distance_km"].quantile(0.90)),
            "max": float(distance.max()),
            "skewness": float(skew(distance)),
            "histogram": np.histogram(distance, bins=30, range=(0, 30))[0].tolist(),
        },
        "trip_duration_seconds": {
            "mean": float(dur_s.mean()),
            "median": float(dur_s.median()),
        },
        "fare_model": {
            "intercept": fare_model.intercept,
            "distance_coef": fare_model.distance_coef,
            "duration_coef": fare_model.duration_coef,
            "r_squared": fare_r2,
        },
        "driver_pay_fraction": {
            "mean": float(valid_ratio.mean()),
            "median": float(valid_ratio.median()),
        },
        "travel_time_sigma": {
            "mean": float(np.mean(sigmas)),
            "median": float(np.median(sigmas)),
            "n_cells": len(sigmas),
            "n_cells_well_sampled": well_sampled.height,
            "fraction_trips_well_sampled": float(well_sampled["n"].sum() / cell_counts["n"].sum()),
        },
        "od_concentration_hhi": hhi,
    }


def main() -> None:
    print("Fetching/loading NYC reference sample...")
    raw = fetch_nyc_sample()
    nyc_trips = adapt_nyc_tlc_trips(raw)
    nyc_stats = describe_trips(nyc_trips, "NYC (Uber HVFHS, 2024-01-16)")

    print("Generating comparable synthetic data...")
    zones = [f"Z{i}" for i in range(15)]
    synthetic_df = generate_synthetic_trips(
        n_days=1, zones=zones, start_date=datetime(2024, 1, 16), fleet_size=500, seed=1
    )
    synthetic_trips = synthetic_df.filter(pl.col("status") == "completed")
    synthetic_stats = describe_trips(synthetic_trips, "Synthetic (15 zones, fleet=500)")

    nyc_h = np.array(nyc_stats["hourly_shape"])
    syn_h = np.array(synthetic_stats["hourly_shape"])
    hourly_cosine_similarity = float(
        np.dot(nyc_h, syn_h) / (np.linalg.norm(nyc_h) * np.linalg.norm(syn_h))
    )

    results = {
        "nyc": nyc_stats,
        "synthetic": synthetic_stats,
        "hourly_cosine_similarity": hourly_cosine_similarity,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote comparison results to {OUTPUT_PATH}")

    for key in ("nyc", "synthetic"):
        s = results[key]
        print(f"\n=== {s['label']} ===")
        print(f"  n_trips={s['n_trips']}, n_zones={s['n_zones']}")
        print(f"  wait (s): {s['wait_seconds']}")
        print(f"  trip_distance_km: mean={s['trip_distance_km']['mean']:.2f}")
        print(f"  fare_model: {s['fare_model']}")
        print(f"  driver_pay_fraction: {s['driver_pay_fraction']}")
        print(f"  travel_time_sigma: {s['travel_time_sigma']}")
        print(f"  od_concentration_hhi: {s['od_concentration_hhi']:.5f}")


if __name__ == "__main__":
    main()
