"""Fit a lognormal travel-time model per (origin, dest, hour) from trip records."""

from __future__ import annotations

import numpy as np
import polars as pl

from dispatch_eval.models import TravelTimeModel


def fit_travel_time_model(
    df: pl.DataFrame,
    origin_col: str = "origin_zone",
    dest_col: str = "dest_zone",
    pickup_ts_col: str = "pickup_ts",
    dropoff_ts_col: str = "dropoff_ts",
    min_samples_per_cell: int = 5,
) -> TravelTimeModel:
    """MLE lognormal fit per (origin, dest, hour): mean/std of log(duration).

    Cells with fewer than `min_samples_per_cell` observations fall back to a
    pooled global fit rather than an unstable per-cell estimate from a
    handful of trips. Keeping variance (not just the mean) is the point —
    see P1 in the project plan.
    """
    working = df.select([origin_col, dest_col, pickup_ts_col, dropoff_ts_col]).drop_nulls()
    duration_seconds = (pl.col(dropoff_ts_col) - pl.col(pickup_ts_col)).dt.total_seconds()
    working = working.with_columns(
        duration_seconds.alias("duration_seconds"),
        pl.col(pickup_ts_col).dt.hour().alias("hour"),
    ).filter(pl.col("duration_seconds") > 0)
    working = working.with_columns(pl.col("duration_seconds").log().alias("log_duration"))

    cell_stats = working.group_by([origin_col, dest_col, "hour"]).agg(
        pl.len().alias("n"),
        pl.col("log_duration").mean().alias("mu"),
        pl.col("log_duration").std(ddof=1).alias("sigma"),
    )

    global_mu = float(working["log_duration"].mean())
    global_sigma = float(working["log_duration"].std(ddof=1))
    fallback_params = (
        global_mu,
        global_sigma if global_sigma and not np.isnan(global_sigma) else 0.3,
    )

    params: dict[tuple[str, str, int], tuple[float, float]] = {}
    for row in cell_stats.iter_rows(named=True):
        key = (row[origin_col], row[dest_col], row["hour"])
        if row["n"] >= min_samples_per_cell and row["sigma"] and not np.isnan(row["sigma"]):
            params[key] = (row["mu"], row["sigma"])
        else:
            params[key] = fallback_params

    return TravelTimeModel(params=params, fallback_params=fallback_params)
