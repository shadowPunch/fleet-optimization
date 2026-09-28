"""Fit the boarding-time model (driver on scene -> rider aboard)."""

from __future__ import annotations

import numpy as np
import polars as pl

from dispatch_eval.models import BoardingModel

N_QUANTILES = 201


def fit_boarding_model(df: pl.DataFrame) -> BoardingModel | None:
    """Empirical boarding-time distribution from `pickup_ts - on_scene_ts`.

    Returns None when the source doesn't report `on_scene_ts` (every source
    except NYC TLC), in which case the simulator models no boarding delay.
    Negative gaps (clock noise in the raw data) are dropped.
    """
    if "on_scene_ts" not in df.columns:
        return None
    seconds = (
        df.select((pl.col("pickup_ts") - pl.col("on_scene_ts")).dt.total_seconds())
        .to_series()
        .drop_nulls()
        .to_numpy()
    )
    seconds = seconds[seconds >= 0]
    if seconds.size == 0:
        return None
    return BoardingModel(quantiles=np.quantile(seconds, np.linspace(0.0, 1.0, N_QUANTILES)))
