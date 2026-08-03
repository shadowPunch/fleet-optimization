"""Fit a row-normalised, smoothed OD (destination-choice) distribution."""

from __future__ import annotations

import polars as pl

from dispatch_eval.models import ODModel


def fit_od_model(
    df: pl.DataFrame,
    origin_col: str = "origin_zone",
    dest_col: str = "dest_zone",
    ts_col: str = "request_ts",
    time_bin_minutes: int = 60,
    smoothing_alpha: float = 1.0,
) -> ODModel:
    """Fit destination | (origin, time bin), with additive (Dirichlet) smoothing.

    Smoothing matters here specifically: origin-destination pairs are sparse
    at the zone level, and an unsmoothed row-normalised matrix would assign
    exactly zero probability to any (origin, dest) pair never observed in the
    fitting window, which is usually just a sparse-data artefact, not a
    structural impossibility.
    """
    minute_of_day = df[ts_col].dt.hour().cast(pl.Int32) * 60 + df[ts_col].dt.minute()
    working = df.select([origin_col, dest_col]).with_columns(
        (minute_of_day // time_bin_minutes).alias("time_bin")
    )

    all_zones = sorted(
        set(working[origin_col].unique().to_list()) | set(working[dest_col].unique().to_list())
    )

    counts = working.group_by([origin_col, "time_bin", dest_col]).agg(pl.len().alias("n"))

    dest_probs: dict[tuple[str, int], dict[str, float]] = {}
    for group_key, group in counts.group_by([origin_col, "time_bin"]):
        origin, time_bin = group_key
        smoothed = {z: smoothing_alpha for z in all_zones}
        for row in group.iter_rows(named=True):
            smoothed[row[dest_col]] += row["n"]
        total = sum(smoothed.values())
        dest_probs[(origin, int(time_bin))] = {z: c / total for z, c in smoothed.items()}

    global_counts = working.group_by(dest_col).agg(pl.len().alias("n"))
    fallback = {z: smoothing_alpha for z in all_zones}
    for row in global_counts.iter_rows(named=True):
        fallback[row[dest_col]] += row["n"]
    fallback_total = sum(fallback.values())
    fallback_probs = {z: c / fallback_total for z, c in fallback.items()}

    return ODModel(dest_probs=dest_probs, fallback_probs=fallback_probs)
