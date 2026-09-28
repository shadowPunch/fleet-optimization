"""Exploratory analysis of the NYC study window (all 14 weekdays).

Produces the tables behind the report's data section; charts are drawn from
them by `studies.report`. Everything here describes the real data, not the
simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from dispatch_eval.studies.nyc_data import NYCScope, load_nyc_data

SECONDS = pl.Float64


def _durations(trips: pl.DataFrame) -> pl.DataFrame:
    return trips.with_columns(
        (pl.col("pickup_ts") - pl.col("request_ts")).dt.total_seconds().cast(SECONDS).alias("wait_s"),
        (pl.col("on_scene_ts") - pl.col("request_ts")).dt.total_seconds().cast(SECONDS).alias("approach_s"),
        (pl.col("pickup_ts") - pl.col("on_scene_ts")).dt.total_seconds().cast(SECONDS).alias("boarding_s"),
        (pl.col("dropoff_ts") - pl.col("pickup_ts")).dt.total_seconds().cast(SECONDS).alias("trip_s"),
        pl.col("request_ts").dt.hour().alias("hour"),
        pl.col("request_ts").dt.truncate("15m").dt.time().alias("slot"),
        pl.col("request_ts").dt.date().alias("date"),
    ).filter(pl.col("wait_s") >= 0)


def demand_by_slot(trips: pl.DataFrame) -> pl.DataFrame:
    """Mean requests per 15-minute slot across days, with day-to-day spread."""
    per_day = trips.group_by("date", "slot").len("requests")
    return (
        per_day.group_by("slot")
        .agg(
            pl.col("requests").mean().alias("mean_requests"),
            pl.col("requests").min().alias("min_requests"),
            pl.col("requests").max().alias("max_requests"),
        )
        .sort("slot")
    )


def wait_components_by_hour(trips: pl.DataFrame) -> pl.DataFrame:
    """Median wait split into its two observable parts, per hour."""
    return (
        trips.group_by("hour")
        .agg(
            pl.len().alias("trips"),
            pl.col("wait_s").median().alias("median_wait_s"),
            pl.col("wait_s").quantile(0.9).alias("p90_wait_s"),
            pl.col("approach_s").median().alias("median_approach_s"),
            pl.col("boarding_s").median().alias("median_boarding_s"),
            # Means add up (approach + boarding = wait); medians don't.
            pl.col("wait_s").mean().alias("mean_wait_s"),
            pl.col("approach_s").mean().alias("mean_approach_s"),
            pl.col("boarding_s").mean().alias("mean_boarding_s"),
        )
        .sort("hour")
    )


def zone_service(trips: pl.DataFrame, zone_names: pl.DataFrame, n_days: int) -> pl.DataFrame:
    """Per pickup zone: daily demand and how long riders wait there."""
    return (
        trips.group_by("origin_zone")
        .agg(
            (pl.len() / n_days).alias("requests_per_day"),
            pl.col("wait_s").median().alias("median_wait_s"),
            pl.col("wait_s").quantile(0.9).alias("p90_wait_s"),
            pl.col("trip_s").median().alias("median_trip_s"),
        )
        .join(zone_names, left_on="origin_zone", right_on="zone", how="left")
        .sort("requests_per_day", descending=True)
    )


def top_flows(trips: pl.DataFrame, zone_names: pl.DataFrame, n_days: int, k: int = 15) -> pl.DataFrame:
    names = zone_names.select("zone", "name")
    return (
        trips.group_by("origin_zone", "dest_zone")
        .agg((pl.len() / n_days).alias("trips_per_day"), pl.col("trip_s").median().alias("median_trip_s"))
        .sort("trips_per_day", descending=True)
        .head(k)
        .join(names.rename({"name": "origin_name"}), left_on="origin_zone", right_on="zone", how="left")
        .join(names.rename({"name": "dest_name"}), left_on="dest_zone", right_on="zone", how="left")
    )


def busy_vehicles_by_slot(trips: pl.DataFrame) -> pl.DataFrame:
    """Lower bound on cars in use, per 15-minute slot, by Little's law:
    requests per second x mean time a car is committed (approach + boarding
    + trip). Idle cars are invisible in trip data, so this bounds the fleet
    from below — a sanity check on the calibrated fleet size."""
    committed = pl.col("wait_s") + pl.col("trip_s")
    per_day = trips.group_by("date", "slot").agg(
        (pl.len() / (15 * 60) * committed.mean()).alias("busy_vehicles")
    )
    return per_day.group_by("slot").agg(pl.col("busy_vehicles").mean()).sort("slot")


def run_eda(cfg: dict, output_dir: Path) -> dict:
    scope = NYCScope.from_config(cfg["data"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    data = load_nyc_data(scope, cache_dir)
    trips = _durations(pl.concat([data.calibration, data.held_out]))
    n_days = trips["date"].n_unique()
    zone_names = pl.read_csv(cache_dir / "nyc_taxi_zone_lookup.csv").select(
        pl.col("LocationID").cast(pl.Utf8).alias("zone"), pl.col("Zone").alias("name")
    )

    tables = {
        "demand_by_slot": demand_by_slot(trips),
        "wait_components_by_hour": wait_components_by_hour(trips),
        "zone_service": zone_service(trips, zone_names, n_days),
        "top_flows": top_flows(trips, zone_names, n_days),
        "busy_vehicles_by_slot": busy_vehicles_by_slot(trips),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(output_dir / f"{name}.parquet")

    summary = {
        "trips": trips.height,
        "days": n_days,
        "zones": len(data.zones),
        "trips_per_day": trips.height / n_days,
        "median_wait_s": trips["wait_s"].median(),
        "p90_wait_s": trips["wait_s"].quantile(0.9),
        "median_approach_s": trips["approach_s"].median(),
        "median_boarding_s": trips["boarding_s"].median(),
        "median_trip_s": trips["trip_s"].median(),
        "same_zone_trip_share": trips.filter(pl.col("origin_zone") == pl.col("dest_zone")).height
        / trips.height,
        "peak_busy_vehicles": tables["busy_vehicles_by_slot"]["busy_vehicles"].max(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
