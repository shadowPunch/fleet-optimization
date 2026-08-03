"""Canonical data schemas.

Every data source (real or synthetic) is adapted into one of these two shapes
before anything else in the package touches it. Keeping one canonical shape
means the simulator, calibration routines, and tests never need to know which
city or which adapter produced the data.

Two shapes, not one, because the two real data sources are not interchangeable
(see dispatch-evaluation-project-plan.md, section 2, and
docs/observability_table.md):

- ``TRIP_RECORD_SCHEMA`` — one row per trip. Only the Delhi NCR synthetic
  source and the synthetic generator produce this.
- ``WARD_AGGREGATE_SCHEMA`` — one row per (ward, time window). This is what
  Bengaluru's real Namma Yatri open data actually offers; it is used for
  validation and demographic joins, never as simulator input.
"""

from __future__ import annotations

import polars as pl

TRIP_STATUSES = (
    "completed",
    "cancelled_customer",
    "cancelled_driver",
    "incomplete",
)

TRIP_RECORD_SCHEMA: dict[str, pl.PolarsDataType] = {
    "trip_id": pl.Utf8,
    "origin_zone": pl.Utf8,
    "dest_zone": pl.Utf8,
    "request_ts": pl.Datetime("us"),
    "pickup_ts": pl.Datetime("us"),
    "dropoff_ts": pl.Datetime("us"),
    "trip_distance_km": pl.Float64,
    "fare": pl.Float64,
    "driver_pay": pl.Float64,
    "status": pl.Utf8,
    "vehicle_type": pl.Utf8,
    "source": pl.Utf8,
}

# Nullable in practice: pickup_ts/dropoff_ts/trip_distance_km/fare/driver_pay
# are absent for cancelled or incomplete trips. request_ts, origin_zone,
# dest_zone, status, and source are required for every row.
TRIP_RECORD_REQUIRED_COLUMNS = (
    "trip_id",
    "origin_zone",
    "dest_zone",
    "request_ts",
    "status",
    "source",
)

WARD_AGGREGATE_SCHEMA: dict[str, pl.PolarsDataType] = {
    "ward": pl.Utf8,
    "window_start": pl.Datetime("us"),
    "requests": pl.Int64,
    "completed": pl.Int64,
    "cancelled": pl.Int64,
    "avg_fare_estimate": pl.Float64,
    "total_earnings": pl.Float64,
    "source": pl.Utf8,
}

WARD_AGGREGATE_REQUIRED_COLUMNS = (
    "ward",
    "window_start",
    "requests",
    "source",
)


class SchemaValidationError(ValueError):
    """Raised when a DataFrame doesn't conform to a canonical schema."""


def _validate(df: pl.DataFrame, required_columns: tuple[str, ...], schema_name: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{schema_name}: missing required column(s) {missing}. Columns present: {df.columns}"
        )


def validate_trip_records(df: pl.DataFrame) -> None:
    """Raise SchemaValidationError if df is missing required trip-record columns."""
    _validate(df, TRIP_RECORD_REQUIRED_COLUMNS, "TRIP_RECORD_SCHEMA")
    bad_status = df.filter(~pl.col("status").is_in(TRIP_STATUSES)).select("status").unique()
    if bad_status.height > 0:
        raise SchemaValidationError(
            f"TRIP_RECORD_SCHEMA: unrecognized status value(s) {bad_status['status'].to_list()}, "
            f"expected one of {TRIP_STATUSES}"
        )


def validate_ward_aggregates(df: pl.DataFrame) -> None:
    """Raise SchemaValidationError if df is missing required ward-aggregate columns."""
    _validate(df, WARD_AGGREGATE_REQUIRED_COLUMNS, "WARD_AGGREGATE_SCHEMA")
