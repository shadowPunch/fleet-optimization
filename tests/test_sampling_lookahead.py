from __future__ import annotations

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.scenario import generate_scenario
from dispatch_eval.simulator.engine import SimulationEngine
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle


def _flat_travel_time_model(zones, seconds: float = 60.0) -> TravelTimeModel:
    params = {
        (o, d, h): (float(np.log(seconds)), 0.0) for o in zones for d in zones for h in range(24)
    }
    return TravelTimeModel(params=params, fallback_params=(float(np.log(seconds)), 0.0))


def _skewed_arrival_model(zones, rates: dict[str, float]) -> NHPPArrivalModel:
    arrival_rates = {(z, "all", b): rates[z] for z in zones for b in range(96)}
    return NHPPArrivalModel(rates=arrival_rates, bin_minutes=15)


def test_reposition_moves_vehicles_toward_sampled_demand():
    zones = ["quiet", "busy"]
    arrival_model = _skewed_arrival_model(zones, {"quiet": 0.0, "busy": 5.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [Vehicle(vehicle_id=f"v{i}", zone="quiet") for i in range(4)]

    policy = SamplingLookaheadPolicy(
        dispatch_policy=NearestIdlePolicy(),
        arrival_model=arrival_model,
        day_type="all",
        lookahead_seconds=600.0,
    )
    moves = policy.reposition(
        idle_vehicles,
        current_time=0.0,
        current_hour=9,
        travel_time_model=tt_model,
        zones=zones,
        rng=np.random.default_rng(0),
    )

    assert len(moves) > 0
    assert all(target_zone == "busy" for _, target_zone in moves)


def test_reposition_returns_empty_with_no_idle_vehicles():
    zones = ["A", "B"]
    arrival_model = _skewed_arrival_model(zones, {"A": 1.0, "B": 1.0})
    tt_model = _flat_travel_time_model(zones)

    policy = SamplingLookaheadPolicy(
        dispatch_policy=NearestIdlePolicy(),
        arrival_model=arrival_model,
        day_type="all",
        lookahead_seconds=300.0,
    )
    moves = policy.reposition(
        [],
        current_time=0.0,
        current_hour=9,
        travel_time_model=tt_model,
        zones=zones,
        rng=np.random.default_rng(0),
    )
    assert moves == []


def test_reposition_returns_empty_when_nothing_is_sampled():
    zones = ["A", "B"]
    arrival_model = _skewed_arrival_model(zones, {"A": 0.0, "B": 0.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [Vehicle(vehicle_id="v0", zone="A")]

    policy = SamplingLookaheadPolicy(
        dispatch_policy=NearestIdlePolicy(),
        arrival_model=arrival_model,
        day_type="all",
        lookahead_seconds=300.0,
    )
    moves = policy.reposition(
        idle_vehicles,
        current_time=0.0,
        current_hour=9,
        travel_time_model=tt_model,
        zones=zones,
        rng=np.random.default_rng(0),
    )
    assert moves == []


def test_reposition_never_moves_a_vehicle_to_the_zone_it_is_already_in():
    zones = ["only"]
    arrival_model = _skewed_arrival_model(zones, {"only": 5.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [Vehicle(vehicle_id="v0", zone="only")]

    policy = SamplingLookaheadPolicy(
        dispatch_policy=NearestIdlePolicy(),
        arrival_model=arrival_model,
        day_type="all",
        lookahead_seconds=600.0,
    )
    moves = policy.reposition(
        idle_vehicles,
        current_time=0.0,
        current_hour=9,
        travel_time_model=tt_model,
        zones=zones,
        rng=np.random.default_rng(0),
    )
    assert moves == []


def test_dispatch_delegates_unchanged_to_the_wrapped_policy():
    zones = ["A", "B"]
    tt_model = _flat_travel_time_model(zones)
    v1 = Vehicle(vehicle_id="v1", zone="A")
    arrival_model = _skewed_arrival_model(zones, {"A": 1.0, "B": 1.0})

    inner = NearestIdlePolicy()
    request = Request(
        request_id="r1",
        origin_zone="A",
        dest_zone="B",
        request_time=0.0,
        abandon_at=1000.0,
        status=RequestStatus.WAITING,
    )
    direct = inner.dispatch([request], [v1], 0.0, 9, tt_model)
    wrapped = SamplingLookaheadPolicy(
        inner, arrival_model, "all", lookahead_seconds=300.0
    ).dispatch([request], [v1], 0.0, 9, tt_model)
    assert direct == wrapped


def test_sampling_lookahead_actually_moves_vehicles_in_the_engine():
    zones = ["quiet", "busy"]
    arrival_model = _skewed_arrival_model(zones, {"quiet": 0.0, "busy": 3.0})
    dest_probs = {(z, h): {zz: 1.0 / len(zones) for zz in zones} for z in zones for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(zones) for z in zones})
    tt_model = _flat_travel_time_model(zones)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    vehicles = [Vehicle(vehicle_id=f"v{i}", zone="quiet") for i in range(6)]
    policy = SamplingLookaheadPolicy(
        dispatch_policy=NearestIdlePolicy(),
        arrival_model=arrival_model,
        day_type="all",
        lookahead_seconds=300.0,
    )
    rng = np.random.default_rng(0)
    scenario = generate_scenario(
        zones, "all", arrival_model, od_model, abandonment_model, 1800.0, rng
    )

    engine = SimulationEngine(
        vehicles=vehicles,
        scenario=scenario,
        travel_time_model=tt_model,
        policy=policy,
        zones=zones,
        horizon_seconds=1800.0,
        rng=rng,
        dispatch_interval_seconds=30.0,
    )
    engine.run()

    assert any(v.zone == "busy" for v in vehicles)
