from __future__ import annotations

import numpy as np
import pytest

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.scenario import Scenario
from dispatch_eval.simulator.engine import SimulationEngine
from dispatch_eval.simulator.entities import Vehicle, VehicleStatus
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

ZONES = ["A", "B", "C"]


def _uniform_models(rate_per_minute: float = 0.3, mean_patience: float = 180.0):
    rates = {(z, "all", b): rate_per_minute for z in ZONES for b in range(96)}
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=15)

    dest_probs = {(z, h): {zz: 1.0 / len(ZONES) for zz in ZONES} for z in ZONES for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(ZONES) for z in ZONES})

    params = {
        (o, d, h): (float(np.log(300.0)), 0.3) for o in ZONES for d in ZONES for h in range(24)
    }
    travel_time_model = TravelTimeModel(params=params, fallback_params=(float(np.log(300.0)), 0.3))

    abandonment_model = AbandonmentModel(mean_patience_seconds=mean_patience)
    return arrival_model, od_model, travel_time_model, abandonment_model


def test_run_produces_internally_consistent_completed_trips():
    arrival_model, od_model, tt_model, ab_model = _uniform_models()
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    result = run_simulation(
        fleet_size=20,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=tt_model,
        abandonment_model=ab_model,
        policy=NearestIdlePolicy(),
        config=config,
        rng=np.random.default_rng(0),
    )

    assert result.total_requests > 0
    assert len(result.completed_requests) > 0

    for request in result.completed_requests:
        assert request.pickup_time >= request.request_time
        assert request.dropoff_time >= request.pickup_time
        assert request.wait_time == pytest.approx(request.pickup_time - request.request_time)

    completed_and_abandoned = len(result.completed_requests) + len(result.abandoned_requests)
    assert completed_and_abandoned <= result.total_requests


def test_undersupplied_fleet_produces_abandonment():
    # High demand, tiny patience, one vehicle: most requests should abandon.
    arrival_model, od_model, tt_model, _ = _uniform_models(rate_per_minute=2.0)
    ab_model = AbandonmentModel(mean_patience_seconds=20.0)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=3 * 3600.0)
    result = run_simulation(
        fleet_size=1,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=tt_model,
        abandonment_model=ab_model,
        policy=NearestIdlePolicy(),
        config=config,
        rng=np.random.default_rng(1),
    )
    assert len(result.abandoned_requests) > len(result.completed_requests)


def test_reposition_vehicle_transitions_state_and_completes():
    _, _, tt_model, _ = _uniform_models(rate_per_minute=0.0)
    vehicle = Vehicle(vehicle_id="veh-0", zone="A")
    engine = SimulationEngine(
        vehicles=[vehicle],
        scenario=Scenario(requests=[]),
        travel_time_model=tt_model,
        policy=NearestIdlePolicy(),
        zones=ZONES,
        horizon_seconds=3600.0,
        rng=np.random.default_rng(2),
    )
    engine.reposition_vehicle("veh-0", "B", current_time=0.0)
    assert vehicle.status == VehicleStatus.REPOSITIONING

    engine.run()
    assert vehicle.status == VehicleStatus.IDLE
    assert vehicle.zone == "B"


def test_event_ordering_is_deterministic_for_a_fixed_seed():
    arrival_model, od_model, tt_model, ab_model = _uniform_models()
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=2 * 3600.0)

    def run_once():
        return run_simulation(
            fleet_size=10,
            arrival_model=arrival_model,
            od_model=od_model,
            travel_time_model=tt_model,
            abandonment_model=ab_model,
            policy=NearestIdlePolicy(),
            config=config,
            rng=np.random.default_rng(7),
        )

    first, second = run_once(), run_once()
    assert [r.request_id for r in first.completed_requests] == [
        r.request_id for r in second.completed_requests
    ]
    assert np.allclose(first.wait_times, second.wait_times)


class _SpyPolicy:
    """Records which travel_time_model object dispatch() is actually called
    with, so tests can assert on object identity rather than behavior."""

    def __init__(self):
        self.seen_travel_time_models: list[TravelTimeModel] = []

    def dispatch(self, waiting_requests, idle_vehicles, current_time, current_hour, travel_time_model):
        self.seen_travel_time_models.append(travel_time_model)
        return []


def test_policy_travel_time_model_defaults_to_the_true_model():
    arrival_model, od_model, true_tt_model, ab_model = _uniform_models()
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=3600.0)
    spy = _SpyPolicy()

    run_simulation(
        fleet_size=5, arrival_model=arrival_model, od_model=od_model,
        travel_time_model=true_tt_model, abandonment_model=ab_model,
        policy=spy, config=config, rng=np.random.default_rng(0),
    )

    assert len(spy.seen_travel_time_models) > 0
    assert all(m is true_tt_model for m in spy.seen_travel_time_models)


def test_policy_travel_time_model_overrides_what_the_policy_receives():
    arrival_model, od_model, true_tt_model, ab_model = _uniform_models()
    policy_tt_model = TravelTimeModel(params={}, fallback_params=(float(np.log(999.0)), 0.1))
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=3600.0)
    spy = _SpyPolicy()

    run_simulation(
        fleet_size=5, arrival_model=arrival_model, od_model=od_model,
        travel_time_model=true_tt_model, abandonment_model=ab_model,
        policy=spy, config=config, rng=np.random.default_rng(0),
        policy_travel_time_model=policy_tt_model,
    )

    assert len(spy.seen_travel_time_models) > 0
    assert all(m is policy_tt_model for m in spy.seen_travel_time_models)
    assert all(m is not true_tt_model for m in spy.seen_travel_time_models)


def test_realized_trip_durations_use_the_true_model_not_the_policy_belief():
    # True world: fast travel (~30s). Policy's belief: absurdly slow (~10000s).
    # Realized dropoff-pickup durations must reflect the true (fast) model,
    # never the policy's mistaken belief -- the belief only shapes decisions.
    arrival_model, od_model, _, ab_model = _uniform_models(rate_per_minute=0.5)
    fast_true_model = TravelTimeModel(params={}, fallback_params=(float(np.log(30.0)), 0.01))
    slow_policy_belief = TravelTimeModel(params={}, fallback_params=(float(np.log(9999.0)), 0.01))
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=3600.0)

    result = run_simulation(
        fleet_size=10, arrival_model=arrival_model, od_model=od_model,
        travel_time_model=fast_true_model, abandonment_model=ab_model,
        policy=NearestIdlePolicy(), config=config, rng=np.random.default_rng(1),
        policy_travel_time_model=slow_policy_belief,
    )

    assert len(result.completed_requests) > 0
    for request in result.completed_requests:
        duration = request.dropoff_time - request.pickup_time
        assert duration < 200.0  # nowhere near the policy's ~9999s belief
