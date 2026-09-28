from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dispatch_eval.calibration.pickup import calibrate_supply, same_zone_log_sigma
from dispatch_eval.studies.nyc_data import NYCData, NYCScope, approach_seconds, wait_seconds
from dispatch_eval.studies.validation import cosine_similarity, hourly_shape

SCOPE = NYCScope("2024-01", "HV0003", "Manhattan", 12, 18, (2, 3), (4,))


def _trips() -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, 12, 30)
    return pl.DataFrame(
        {
            "origin_zone": ["1", "1", "2"],
            "dest_zone": ["1", "1", "3"],
            "request_ts": [t0, t0, t0],
            "on_scene_ts": [t0 + timedelta(seconds=100), None, t0 + timedelta(seconds=50)],
            "pickup_ts": [t0 + timedelta(seconds=s) for s in (130, 200, 60)],
            "dropoff_ts": [t0 + timedelta(seconds=s) for s in (430, 1100, 660)],
        }
    )


def test_for_simulation_maps_window_start_to_midnight_and_keeps_durations():
    data = NYCData(SCOPE, ["1", "2", "3"], _trips(), _trips())
    shifted = data.for_simulation(data.calibration)
    assert shifted["request_ts"][0] == datetime(2024, 1, 2, 0, 30)
    assert np.array_equal(wait_seconds(shifted), wait_seconds(data.calibration))


def test_wait_and_approach_seconds():
    trips = _trips()
    assert wait_seconds(trips).tolist() == [130.0, 200.0, 60.0]
    assert approach_seconds(trips).tolist() == [100.0, 50.0]  # null on_scene dropped


def test_same_zone_log_sigma_uses_only_same_zone_trips():
    # same-zone durations: 300s and 900s -> std of their logs
    expected = float(np.std(np.log([300.0, 900.0]), ddof=1))
    assert same_zone_log_sigma(_trips()) == pytest.approx(expected)


def test_calibrate_supply_finds_grid_minimum_and_flags_boundary():
    observed = np.random.default_rng(0).normal(200.0, 20.0, 2000)

    def simulate(fleet_size, params):
        # Wait falls with fleet size and rises with same-zone pickup time.
        median = np.exp(params[0])
        shift = median + 40_000.0 / fleet_size
        return np.random.default_rng(1).normal(shift, 20.0, 2000)

    interior = calibrate_supply(observed, simulate, [200, 400, 800], [60.0, 100.0, 140.0], 0.5)
    assert (interior.fleet_size, interior.intra_zone_median_seconds) == (400, 100.0)
    assert not interior.hit_boundary
    assert len(interior.grid) == 9
    assert interior.intra_zone_params == pytest.approx((np.log(100.0), 0.5))

    edge = calibrate_supply(observed, simulate, [400, 800], [100.0, 140.0], 0.5)
    assert edge.hit_boundary


def test_hourly_shape_and_cosine():
    shape = hourly_shape(np.array([0, 0, 1, 5]), 6)
    assert shape.tolist() == [0.5, 0.25, 0, 0, 0, 0.25]
    assert cosine_similarity(shape, shape) == pytest.approx(1.0)
