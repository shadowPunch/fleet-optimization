"""Adapter for NYC TLC High Volume FHV trip records.

Source: `fhvhv_tripdata_YYYY-MM.parquet`, published monthly at
`https://d37ci6vzurychx.cloudfront.net/trip-data/`, no authentication
required. Unlike `sources.delhi_ncr` and `sources.bengaluru`, this schema is
**verified**, not a guess: confirmed directly against a live scan of
`fhvhv_tripdata_2024-01.parquet` (see `analysis/nyc_reference_comparison.py`).
This isn't this project's primary data source (see the project plan and
README for why — no public Indian-city equivalent exists), but it's real,
richly-fielded, and immediately downloadable, which makes it useful as a
reference for sanity-checking the synthetic generator's realism.

`hvfhs_license_num` distinguishes operators (HV0003 = Uber, HV0005 = Lyft,
as of the 2024 data dictionary) — always filter to one before treating this
as "a fleet," per the plan's data notes.
"""

from __future__ import annotations

import polars as pl

from dispatch_eval.schema import validate_trip_records

MILES_TO_KM = 1.609344


def adapt_nyc_tlc_trips(raw: pl.DataFrame) -> pl.DataFrame:
    """Adapt a raw NYC TLC HVFHS DataFrame (already loaded, ideally
    pre-filtered to one `hvfhs_license_num` and a manageable date range)
    into TRIP_RECORD_SCHEMA.

    Only matched trips appear in this dataset — abandoned/unserved requests
    are absent (see the project plan's data caveats) — so every row is
    `status="completed"`; there is no cancellation taxonomy to map, unlike
    the Delhi NCR / Bengaluru sources.
    """
    trips = raw.select(
        [
            (pl.int_range(pl.len()).cast(pl.Utf8)).alias("trip_id"),
            pl.col("PULocationID").cast(pl.Utf8).alias("origin_zone"),
            pl.col("DOLocationID").cast(pl.Utf8).alias("dest_zone"),
            pl.col("request_datetime").alias("request_ts"),
            pl.col("on_scene_datetime").alias("on_scene_ts"),
            pl.col("pickup_datetime").alias("pickup_ts"),
            pl.col("dropoff_datetime").alias("dropoff_ts"),
            (pl.col("trip_miles") * MILES_TO_KM).alias("trip_distance_km"),
            pl.col("base_passenger_fare").alias("fare"),
            pl.col("driver_pay").alias("driver_pay"),
            pl.lit("completed").alias("status"),
            pl.lit(None, dtype=pl.Utf8).alias("vehicle_type"),
            pl.lit("nyc_tlc").alias("source"),
        ]
    )
    validate_trip_records(trips)
    return trips
