"""Fit a piecewise-constant NHPP arrival-rate model from trip-level request timestamps."""

from __future__ import annotations

import polars as pl

from dispatch_eval.models import NHPPArrivalModel


def fit_arrival_model(
    df: pl.DataFrame,
    zone_col: str = "origin_zone",
    ts_col: str = "request_ts",
    day_type_col: str | None = None,
    bin_minutes: int = 15,
) -> NHPPArrivalModel:
    """Fit rate[(zone, day_type, bin_index)] as mean requests per minute.

    If `day_type_col` is None, every row is treated as day_type="all" — use
    this for a single day-type study window before extending to weekday /
    weekend splits.
    """
    working = df.select([zone_col, ts_col] + ([day_type_col] if day_type_col else []))
    if day_type_col is None:
        working = working.with_columns(pl.lit("all").alias("day_type"))
        day_type_col = "day_type"

    minute_of_day = working[ts_col].dt.hour().cast(pl.Int32) * 60 + working[ts_col].dt.minute()
    working = working.with_columns(
        (minute_of_day // bin_minutes).alias("bin_index"),
        working[ts_col].dt.date().alias("_date"),
    )

    n_days = working["_date"].n_unique()
    counts = working.group_by([zone_col, day_type_col, "bin_index"]).agg(pl.len().alias("n_total"))

    rates = {
        (row[zone_col], row[day_type_col], row["bin_index"]): row["n_total"] / n_days / bin_minutes
        for row in counts.iter_rows(named=True)
    }
    return NHPPArrivalModel(rates=rates, bin_minutes=bin_minutes)


def dispersion_by_zone(
    df: pl.DataFrame,
    zone_col: str = "origin_zone",
    ts_col: str = "request_ts",
    bin_minutes: int = 15,
) -> pl.DataFrame:
    """Index of dispersion (variance/mean of per-bin counts), per zone.

    Near 1 is consistent with the Poisson assumption; well above 1 flags
    overdispersion, which the plan says belongs in the structural ambiguity
    set (C1) rather than being silently absorbed into the rate fit.
    """
    minute_of_day = df[ts_col].dt.hour().cast(pl.Int32) * 60 + df[ts_col].dt.minute()
    working = df.select([zone_col]).with_columns(
        (minute_of_day // bin_minutes).alias("bin_index"),
        df[ts_col].dt.date().alias("_date"),
    )

    observed = working.group_by([zone_col, "_date", "bin_index"]).agg(pl.len().alias("n"))

    zones = working[zone_col].unique().to_frame()
    dates = working["_date"].unique().to_frame()
    bin_lo, bin_hi = int(working["bin_index"].min()), int(working["bin_index"].max())
    bins = pl.DataFrame({"bin_index": list(range(bin_lo, bin_hi + 1))})

    full_index = zones.join(dates, how="cross").join(bins, how="cross")
    filled = full_index.join(
        observed, on=[zone_col, "_date", "bin_index"], how="left"
    ).with_columns(pl.col("n").fill_null(0))

    return (
        filled.group_by(zone_col)
        .agg(pl.col("n").mean().alias("mean_count"), pl.col("n").var().alias("var_count"))
        .with_columns((pl.col("var_count") / pl.col("mean_count")).alias("index_of_dispersion"))
    )
