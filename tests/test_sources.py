from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from dispatch_eval.schema import (
    SchemaValidationError,
    validate_trip_records,
    validate_ward_aggregates,
)
from dispatch_eval.sources.bengaluru import adapt_bengaluru_ward_aggregates
from dispatch_eval.sources.delhi_ncr import adapt_delhi_ncr_trips
from dispatch_eval.sources.nyc_tlc import adapt_nyc_tlc_trips
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C"]


def test_generate_synthetic_trips_produces_schema_valid_data():
    df = generate_synthetic_trips(
        n_days=2, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=20, seed=5
    )
    validate_trip_records(df)  # raises on failure
    assert df.height > 0
    assert set(df["status"].unique().to_list()) <= {
        "completed",
        "cancelled_customer",
        "cancelled_driver",
        "incomplete",
    }


def test_validate_trip_records_rejects_missing_columns():
    bad = pl.DataFrame({"trip_id": ["1"], "origin_zone": ["A"]})
    with pytest.raises(SchemaValidationError):
        validate_trip_records(bad)


def test_validate_trip_records_rejects_unknown_status():
    bad = pl.DataFrame(
        {
            "trip_id": ["1"],
            "origin_zone": ["A"],
            "dest_zone": ["B"],
            "request_ts": [datetime(2026, 1, 1)],
            "status": ["not_a_real_status"],
            "source": ["synthetic"],
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_trip_records(bad)


def test_adapt_delhi_ncr_trips_maps_columns_and_status():
    raw = pl.DataFrame(
        {
            "Booking ID": ["B1", "B2"],
            "Pickup Location": ["Okhla", "Saket"],
            "Drop Location": ["Cyber Hub", "Barakhamba Road"],
            "Date": [datetime(2026, 1, 1, 9, 0), datetime(2026, 1, 1, 10, 0)],
            "Booking Status": ["Completed", "Cancelled by Customer"],
            "Ride Distance": [5.2, None],
            "Booking Value": [150.0, None],
            "Vehicle Type": ["Auto", "Auto"],
        }
    )
    adapted = adapt_delhi_ncr_trips(raw)
    validate_trip_records(adapted)
    assert adapted["status"].to_list() == ["completed", "cancelled_customer"]
    assert adapted["source"].unique().to_list() == ["delhi_ncr_synthetic"]
    assert adapted["origin_zone"].to_list() == ["Okhla", "Saket"]


def test_adapt_delhi_ncr_trips_raises_on_unexpected_schema():
    raw = pl.DataFrame({"Some Other Column": [1, 2]})
    with pytest.raises(ValueError, match="not found"):
        adapt_delhi_ncr_trips(raw)


def test_adapt_bengaluru_ward_aggregates_maps_required_and_optional_columns():
    raw = pl.DataFrame(
        {
            "Ward": ["Indiranagar", "Koramangala"],
            "Date": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
            "Ride Requests": [120, 80],
            "Completed Trips": [100, 65],
        }
    )
    adapted = adapt_bengaluru_ward_aggregates(raw)
    validate_ward_aggregates(adapted)
    assert adapted["ward"].to_list() == ["Indiranagar", "Koramangala"]
    assert adapted["completed"].to_list() == [100, 65]
    assert adapted["cancelled"].is_null().all()  # not present in raw, left null


def test_adapt_bengaluru_ward_aggregates_raises_on_missing_required_column():
    raw = pl.DataFrame({"Ward": ["Indiranagar"]})
    with pytest.raises(ValueError, match="Missing required"):
        adapt_bengaluru_ward_aggregates(raw)


def test_adapt_nyc_tlc_trips_maps_columns():
    # Field names match the live fhvhv_tripdata schema exactly (verified by
    # scanning the real file), unlike the Delhi NCR / Bengaluru guesses.
    raw = pl.DataFrame(
        {
            "hvfhs_license_num": ["HV0003", "HV0003"],
            "PULocationID": [7, 130],
            "DOLocationID": [234, 122],
            "request_datetime": [datetime(2024, 1, 16, 0, 3), datetime(2024, 1, 16, 0, 4)],
            "on_scene_datetime": [datetime(2024, 1, 16, 0, 8), datetime(2024, 1, 16, 0, 9)],
            "pickup_datetime": [datetime(2024, 1, 16, 0, 9), datetime(2024, 1, 16, 0, 10)],
            "dropoff_datetime": [datetime(2024, 1, 16, 0, 25), datetime(2024, 1, 16, 0, 20)],
            "trip_miles": [5.0, 2.0],
            "base_passenger_fare": [22.5, 10.0],
            "driver_pay": [17.0, 7.5],
        }
    )
    adapted = adapt_nyc_tlc_trips(raw)
    validate_trip_records(adapted)
    assert adapted["origin_zone"].to_list() == ["7", "130"]
    assert adapted["on_scene_ts"][0] == datetime(2024, 1, 16, 0, 8)
    assert adapted["status"].unique().to_list() == ["completed"]
    assert adapted["trip_distance_km"][0] == pytest.approx(5.0 * 1.609344)
    assert adapted["source"].unique().to_list() == ["nyc_tlc"]
