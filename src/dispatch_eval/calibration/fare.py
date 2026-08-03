"""Fit a linear fare model: fare ~ distance + duration, via ordinary least squares."""

from __future__ import annotations

import numpy as np
import polars as pl

from dispatch_eval.models import FareModel


def fit_fare_model(
    df: pl.DataFrame,
    distance_col: str = "trip_distance_km",
    pickup_ts_col: str = "pickup_ts",
    dropoff_ts_col: str = "dropoff_ts",
    fare_col: str = "fare",
    driver_pay_fraction: float = 0.75,
) -> FareModel:
    working = df.select([distance_col, pickup_ts_col, dropoff_ts_col, fare_col]).drop_nulls()
    duration_minutes = (pl.col(dropoff_ts_col) - pl.col(pickup_ts_col)).dt.total_seconds() / 60.0
    working = working.with_columns(duration_minutes.alias("duration_minutes"))

    x = working.select([distance_col, "duration_minutes"]).to_numpy()
    y = working[fare_col].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, distance_coef, duration_coef = coefs

    return FareModel(
        intercept=float(intercept),
        distance_coef=float(distance_coef),
        duration_coef=float(duration_coef),
        driver_pay_fraction=driver_pay_fraction,
    )
