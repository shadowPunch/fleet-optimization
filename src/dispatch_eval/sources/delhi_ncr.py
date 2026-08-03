"""Adapter for the Kaggle Delhi NCR ride-booking dataset.

Source: `shayanzk/ola-ride-bookings-dataset` on Kaggle. Treated as
**synthetic, not real operational data** — see section 2 of
dispatch-evaluation-project-plan.md for why (no disclosed data-generating
process, Ola/Uber don't publish real trip microdata for India, and the shape
matches generic Kaggle BI-practice datasets). Use it as a structural
test-bed for the harness, never as evidence about real Delhi ride-hailing.

`DEFAULT_COLUMN_MAP` is a best-effort guess at the raw column names from the
dataset's public Kaggle description as of 2026-08 — it has not been verified
against the actual downloaded CSV in this environment. Confirm it against the
real file header before trusting a calibration run on it; `adapt_delhi_ncr_trips`
raises a clear error if the map doesn't match instead of silently emitting
nulls.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dispatch_eval.schema import validate_trip_records

DEFAULT_COLUMN_MAP = {
    "trip_id": "Booking ID",
    "origin_zone": "Pickup Location",
    "dest_zone": "Drop Location",
    "request_ts": "Date",
    "status": "Booking Status",
    "trip_distance_km": "Ride Distance",
    "fare": "Booking Value",
    "vehicle_type": "Vehicle Type",
}

DEFAULT_STATUS_MAP = {
    "Completed": "completed",
    "Cancelled by Customer": "cancelled_customer",
    "Cancelled by Driver": "cancelled_driver",
    "Incomplete": "incomplete",
}


def adapt_delhi_ncr_trips(
    raw: pl.DataFrame,
    column_map: dict[str, str] | None = None,
    status_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Adapt an already-loaded raw Delhi NCR DataFrame into TRIP_RECORD_SCHEMA.

    Takes a DataFrame rather than a file path: date/time parsing in the raw
    CSV needs to be checked against the actual downloaded file, so that step
    is left to the caller (see `load_delhi_ncr_trips`) instead of guessed
    here. `raw[column_map["request_ts"]]` must already be a Datetime column.
    """
    column_map = column_map or DEFAULT_COLUMN_MAP
    status_map = status_map or DEFAULT_STATUS_MAP

    missing_raw_cols = [c for c in column_map.values() if c not in raw.columns]
    if missing_raw_cols:
        raise ValueError(
            f"Expected raw column(s) {missing_raw_cols} not found. "
            f"Columns present: {raw.columns}. Pass an explicit column_map — "
            f"DEFAULT_COLUMN_MAP is an unverified best guess, not a confirmed schema."
        )

    trips = raw.select([pl.col(raw_col).alias(canon) for canon, raw_col in column_map.items()])
    trips = trips.with_columns(
        pl.col("status").replace_strict(status_map, default="incomplete"),
        pl.lit("delhi_ncr_synthetic").alias("source"),
    )
    for optional_col in ("pickup_ts", "dropoff_ts", "driver_pay"):
        if optional_col not in trips.columns:
            trips = trips.with_columns(pl.lit(None).alias(optional_col))

    validate_trip_records(trips)
    return trips


def load_delhi_ncr_trips(
    csv_path: str | Path,
    column_map: dict[str, str] | None = None,
    status_map: dict[str, str] | None = None,
    datetime_format: str | None = None,
) -> pl.DataFrame:
    """Read the raw CSV, parse `request_ts`, and adapt it into TRIP_RECORD_SCHEMA.

    `datetime_format` must match whatever the real file's date/time column(s)
    actually look like — not confirmed in this environment. If the column
    already parses as a Datetime, leave this as None.
    """
    column_map = column_map or DEFAULT_COLUMN_MAP
    raw = pl.read_csv(csv_path)
    ts_raw_col = column_map["request_ts"]
    if ts_raw_col in raw.columns and raw.schema.get(ts_raw_col) not in (
        pl.Datetime,
        pl.Datetime("us"),
        pl.Datetime("ns"),
    ):
        raw = raw.with_columns(
            pl.col(ts_raw_col)
            .str.strptime(pl.Datetime, datetime_format, strict=False)
            .alias(ts_raw_col)
        )
    return adapt_delhi_ncr_trips(raw, column_map, status_map)
