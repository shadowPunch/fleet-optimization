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

    **A (zone, day_type, bin_index) cell with zero observed requests is
    silently absent from `rates`** — `NHPPArrivalModel.rate_per_minute`
    then falls back to a rate of exactly 0.0 for it. For a cell genuinely
    covered by the fitting window, rate=0 is the correct MLE given zero
    observed events — not a bug. But unlike `fit_od_model`'s destination
    distribution (which Dirichlet-smooths every zone into nonzero mass
    specifically so sparse-data zero counts aren't mistaken for structural
    impossibility), this fit applies **no smoothing at all**: a zone/bin
    that happened to see zero arrivals purely by chance in a short fitting
    window is treated with the same full confidence as one that structurally
    never has demand there. `arrival_sparsity_report` makes how much of the
    fitted grid rests on this fallback visible; a real follow-up (not
    currently in scope) is a shrinkage/backoff estimator matching the OD
    model's treatment. This asymmetry is one plausible explanation for
    `TECHNICAL_DOCUMENTATION.md`'s finding that across-theta variance
    shrinks *faster* than 1/n_days at small n — an unsmoothed zero-fallback
    is a **lower**-variance (more confidently wrong, not more uncertain)
    estimator at small n than a smoothed one would be, not yet confirmed
    but consistent with the direction measured there.
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


def arrival_sparsity_report(
    model: NHPPArrivalModel, zones: list[str], day_types: list[str]
) -> dict[str, float | int]:
    """How much of the full (zone, day_type, bin_index) grid a fitted
    `NHPPArrivalModel` actually observed, vs. how much silently rests on
    the zero-rate fallback — see `fit_arrival_model`'s docstring for why
    that fallback isn't smoothed the way the OD model's is. A high
    `fraction_zero_fallback` means a lot of the model's "this zone/bin has
    no demand" claims are absence-of-evidence from a short fitting window,
    not evidence of absence — worth widening the fitting window or adding
    smoothing before trusting simulation output driven by those cells.
    """
    n_bins_per_day = -(-1440 // model.bin_minutes)  # ceil(1440 / bin_minutes)
    full_grid_size = len(zones) * len(day_types) * n_bins_per_day
    observed_cells = sum(
        1
        for zone in zones
        for day_type in day_types
        for bin_index in range(n_bins_per_day)
        if (zone, day_type, bin_index) in model.rates
    )
    zero_fallback_cells = full_grid_size - observed_cells
    return {
        "full_grid_size": full_grid_size,
        "observed_cells": observed_cells,
        "zero_fallback_cells": zero_fallback_cells,
        "fraction_zero_fallback": zero_fallback_cells / full_grid_size if full_grid_size else 0.0,
    }


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
