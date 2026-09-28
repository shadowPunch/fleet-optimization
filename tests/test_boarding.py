from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dispatch_eval.calibration.boarding import fit_boarding_model
from dispatch_eval.models import BoardingModel, TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from tests.test_engine import ZONES, _uniform_models


def _trips(boarding_seconds: list[float | None]) -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, 12)
    on_scene = [t0 if b is not None else None for b in boarding_seconds]
    pickup = [t0 + timedelta(seconds=b or 0) for b in boarding_seconds]
    return pl.DataFrame({"on_scene_ts": on_scene, "pickup_ts": pickup})


def test_fit_boarding_model_returns_none_without_on_scene_column():
    assert fit_boarding_model(pl.DataFrame({"pickup_ts": [datetime(2024, 1, 1)]})) is None


def test_fit_boarding_model_recovers_empirical_quantiles_and_drops_nulls():
    model = fit_boarding_model(_trips([10.0, 20.0, 30.0, None, 40.0, 50.0]))
    assert model.quantiles[0] == pytest.approx(10.0)
    assert model.quantiles[-1] == pytest.approx(50.0)
    assert np.median(model.quantiles) == pytest.approx(30.0)


def test_boarding_samples_stay_within_observed_range():
    model = BoardingModel(quantiles=np.linspace(5.0, 120.0, 11))
    rng = np.random.default_rng(0)
    draws = np.array([model.sample(rng) for _ in range(2000)])
    assert draws.min() >= 5.0 and draws.max() <= 120.0
    assert np.median(draws) == pytest.approx(62.5, rel=0.1)


def test_wait_includes_approach_and_boarding():
    arrival_model, od_model, tt_model, ab_model = _uniform_models(rate_per_minute=0.2)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=2 * 3600.0)
    constant_boarding = BoardingModel(quantiles=np.full(5, 45.0))

    result = run_simulation(
        fleet_size=30, arrival_model=arrival_model, od_model=od_model,
        travel_time_model=tt_model, abandonment_model=ab_model,
        policy=NearestIdlePolicy(), config=config, rng=np.random.default_rng(3),
        boarding_model=constant_boarding,
    )

    assert len(result.completed_requests) > 0
    for request in result.completed_requests:
        assert request.boarding_seconds == pytest.approx(45.0)
        assert request.pickup_time - request.on_scene_time == pytest.approx(45.0)
        assert request.on_scene_time >= request.request_time


def test_intra_zone_params_control_same_zone_travel():
    tt = TravelTimeModel(params={}, fallback_params=(np.log(600.0), 0.0),
                         intra_zone_params=(np.log(150.0), 0.0))
    assert tt.expected("A", "A", 0) == pytest.approx(150.0)
    assert tt.expected("A", "B", 0) == pytest.approx(600.0)
