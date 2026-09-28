"""Adapter for Bengaluru's Namma Yatri ward-wise open data.

Source: nammayatri.in/open — real, live, operational data from an actual
running dispatch platform, aggregated at the ward level. Mirrored as a
scraped snapshot at `arshdkhan/namma-yatri-bengaluru-ward-wise-ride-open-data`
on Kaggle. See TECHNICAL_REPORT.md: this source has no trip-level
timestamps or OD structure, only per-ward, per-time-window counts — it is
used for aggregate validation and the C5 demographic join, never as
simulator input.

`DEFAULT_COLUMN_MAP` is an unverified best guess at the export's column
names; confirm against the actual file before trusting a validation run.
"""

from __future__ import annotations

import polars as pl

from dispatch_eval.schema import validate_ward_aggregates

DEFAULT_COLUMN_MAP = {
    "ward": "Ward",
    "window_start": "Date",
    "requests": "Ride Requests",
    "completed": "Completed Trips",
    "cancelled": "Cancelled Trips",
    "avg_fare_estimate": "Avg Fare",
    "total_earnings": "Total Earnings",
}

REQUIRED_CANONICAL_COLUMNS = ("ward", "window_start", "requests")
OPTIONAL_CANONICAL_COLUMNS = ("completed", "cancelled", "avg_fare_estimate", "total_earnings")


def adapt_bengaluru_ward_aggregates(
    raw: pl.DataFrame, column_map: dict[str, str] | None = None
) -> pl.DataFrame:
    """Adapt an already-loaded raw ward-aggregate DataFrame into WARD_AGGREGATE_SCHEMA.

    `raw[column_map["window_start"]]` must already be a Datetime column — date
    parsing is left to the caller, same rationale as `sources.delhi_ncr`.
    Columns in `column_map` that aren't present in `raw` are simply skipped if
    optional, or raise if required — the real export's exact column set
    hasn't been confirmed in this environment.
    """
    column_map = column_map or DEFAULT_COLUMN_MAP
    present = {canon: raw_col for canon, raw_col in column_map.items() if raw_col in raw.columns}

    missing_required = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in present]
    if missing_required:
        raise ValueError(
            f"Missing required mapped column(s) {missing_required}. "
            f"Columns present in raw data: {raw.columns}. Pass an explicit column_map — "
            f"DEFAULT_COLUMN_MAP is an unverified best guess, not a confirmed schema."
        )

    aggregates = raw.select(
        [pl.col(raw_col).alias(canon) for canon, raw_col in present.items()]
    ).with_columns(pl.lit("bengaluru_namma_yatri").alias("source"))

    for optional_col in OPTIONAL_CANONICAL_COLUMNS:
        if optional_col not in aggregates.columns:
            aggregates = aggregates.with_columns(pl.lit(None).alias(optional_col))

    validate_ward_aggregates(aggregates)
    return aggregates
