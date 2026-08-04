from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from dispatch_eval.decision_currency import (
    DecisionCurrencyResult,
    FleetWaitCurve,
    build_fleet_wait_curve,
    decision_currency,
    mean_fare_per_trip,
    trips_per_vehicle_per_day,
)
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import RankingFlipResult, fit_all_models
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C"]


def _fake_ranking_result(policy_names: list[str], metric_arrays: list[np.ndarray]) -> RankingFlipResult:
    n_bootstrap = metric_arrays[0].shape[0]
    return RankingFlipResult(
        policy_names=policy_names,
        metric_by_policy=dict(zip(policy_names, metric_arrays, strict=True)),
        rankings=np.zeros((n_bootstrap, len(policy_names)), dtype=int),
        nominal_ranking=policy_names,
    )


# --- FleetWaitCurve.invert -------------------------------------------------


def test_invert_interpolates_within_range():
    curve = FleetWaitCurve(fleet_sizes=[10, 20, 30], mean_wait_seconds=[300.0, 200.0, 100.0])
    fleet, extrapolated = curve.invert(150.0)
    assert fleet == pytest.approx(25.0)
    assert extrapolated is False


def test_invert_extrapolates_below_range_when_target_better_than_any_swept_point():
    curve = FleetWaitCurve(fleet_sizes=[10, 20, 30], mean_wait_seconds=[300.0, 200.0, 100.0])
    fleet, extrapolated = curve.invert(50.0)  # better wait than even fleet=30 achieves
    # same trend (wait = 400 - 10*fleet) extended: fleet = (400-50)/10 = 35
    assert fleet == pytest.approx(35.0)
    assert extrapolated is True


def test_invert_extrapolates_above_range_when_target_worse_than_any_swept_point():
    curve = FleetWaitCurve(fleet_sizes=[10, 20, 30], mean_wait_seconds=[300.0, 200.0, 100.0])
    fleet, extrapolated = curve.invert(400.0)
    assert fleet == pytest.approx(0.0)
    assert extrapolated is True


def test_invert_requires_at_least_two_points():
    curve = FleetWaitCurve(fleet_sizes=[10], mean_wait_seconds=[300.0])
    with pytest.raises(ValueError, match="at least 2"):
        curve.invert(150.0)


# --- build_fleet_wait_curve --------------------------------------------------


@pytest.fixture(scope="module")
def trips_df():
    return generate_synthetic_trips(
        n_days=3, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=30, seed=3
    )


def test_build_fleet_wait_curve_more_vehicles_means_lower_or_equal_wait(trips_df):
    models = fit_all_models(trips_df)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    curve = build_fleet_wait_curve(
        fleet_sizes=[10, 20, 40],
        models=models,
        abandonment_model=abandonment_model,
        policy=NearestIdlePolicy(),
        config=config,
        n_replications=3,
        seed=11,
    )

    assert curve.fleet_sizes == [10, 20, 40]
    assert len(curve.mean_wait_seconds) == 3
    # not strictly monotonic under noise, but adding lots more vehicles should
    # not make the smallest fleet size look better than the largest
    assert curve.mean_wait_seconds[0] >= curve.mean_wait_seconds[-1]


# --- mean_fare_per_trip / trips_per_vehicle_per_day --------------------------


def test_mean_fare_per_trip_averages_only_completed_trips():
    df = pl.DataFrame(
        {
            "status": ["completed", "completed", "cancelled_customer"],
            "fare": [100.0, 200.0, 9999.0],
        }
    )
    assert mean_fare_per_trip(df) == pytest.approx(150.0)


def test_trips_per_vehicle_per_day_arithmetic():
    # 240 completed trips, 10 vehicles, 24-hour horizon -> 24 trips/vehicle/day
    assert trips_per_vehicle_per_day(n_completed=240, fleet_size=10, horizon_hours=24.0) == pytest.approx(24.0)


# --- decision_currency --------------------------------------------------------


def test_decision_currency_centers_near_zero_for_reference_policy_against_itself():
    # A curve where the reference policy's own nominal mean wait at
    # actual_fleet_size=20 is exactly 200s (matches the curve's own 20 -> 200
    # point), and a "ref" entry in the ranking result whose per-draw wait is
    # always 200s: vehicles_worth should be exactly zero for every draw.
    curve = FleetWaitCurve(fleet_sizes=[10, 20, 30], mean_wait_seconds=[300.0, 200.0, 100.0])
    n_bootstrap = 50
    metric = np.full((n_bootstrap, 4), 200.0)
    result = _fake_ranking_result(["ref"], [metric])

    out = decision_currency(
        result,
        reference_policy_name="ref",
        reference_curve=curve,
        actual_fleet_size=20,
        horizon_hours=24.0,
        dollars_per_trip=50.0,
        reference_trips_per_vehicle_per_day=10.0,
    )

    assert np.allclose(out["ref"].vehicles_worth_per_draw, 0.0)
    assert not out["ref"].extrapolated_per_draw.any()


def test_decision_currency_positive_when_policy_beats_reference():
    # A policy whose per-draw wait (150s) is better than the reference curve's
    # wait at the actual fleet size (200s at fleet=20) should be "worth"
    # positive additional vehicles: the reference policy would need fleet=25
    # (interpolated) to match 150s, i.e. +5 vehicles.
    curve = FleetWaitCurve(fleet_sizes=[10, 20, 30], mean_wait_seconds=[300.0, 200.0, 100.0])
    n_bootstrap = 10
    metric = np.full((n_bootstrap, 2), 150.0)
    result = _fake_ranking_result(["better"], [metric])

    out = decision_currency(
        result,
        reference_policy_name="ref",
        reference_curve=curve,
        actual_fleet_size=20,
        horizon_hours=24.0,
        dollars_per_trip=50.0,
        reference_trips_per_vehicle_per_day=10.0,
    )

    assert np.allclose(out["better"].vehicles_worth_per_draw, 5.0)


def test_decision_currency_summary_arithmetic():
    result = DecisionCurrencyResult(
        policy_name="better",
        reference_policy_name="ref",
        actual_fleet_size=20,
        vehicles_worth_per_draw=np.array([4.0, 5.0, 6.0]),
        extrapolated_per_draw=np.array([False, False, True]),
        horizon_hours=24.0,
        dollars_per_trip=50.0,
        reference_trips_per_vehicle_per_day=10.0,
    )
    summary = result.summary(confidence=0.95)

    assert summary["vehicles_worth_mean"] == pytest.approx(5.0)
    assert summary["fraction_draws_extrapolated"] == pytest.approx(1 / 3)
    # 5 vehicles * 24 hours each
    assert summary["delta_driver_hours_mean"] == pytest.approx(120.0)
    # 5 vehicles * 10 trips/vehicle/day * $50/trip
    assert summary["delta_dollars_per_day_mean"] == pytest.approx(2500.0)
    lo, hi = summary["vehicles_worth_ci"]
    assert lo <= summary["vehicles_worth_mean"] <= hi
