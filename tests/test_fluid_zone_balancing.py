from __future__ import annotations

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.fluid_zone_balancing import (
    FluidZoneBalancingPolicy,
    largest_remainder_allocation,
)
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.engine import SimulationEngine
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle


def test_largest_remainder_allocation_sums_to_total_exactly():
    shares = {"A": 0.34, "B": 0.33, "C": 0.33}
    result = largest_remainder_allocation(shares, total=10)
    assert sum(result.values()) == 10
    assert result["A"] == 4  # largest fractional part (0.4) gets the leftover unit
    assert result["B"] == 3
    assert result["C"] == 3


def test_largest_remainder_allocation_handles_exact_division():
    shares = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = largest_remainder_allocation(shares, total=10)
    assert result == {"A": 5, "B": 3, "C": 2}


def _flat_travel_time_model(zones, seconds: float = 60.0) -> TravelTimeModel:
    params = {
        (o, d, h): (float(np.log(seconds)), 0.0) for o in zones for d in zones for h in range(24)
    }
    return TravelTimeModel(params=params, fallback_params=(float(np.log(seconds)), 0.0))


def _skewed_arrival_model(zones, rates: dict[str, float]) -> NHPPArrivalModel:
    arrival_rates = {(z, "all", b): rates[z] for z in zones for b in range(96)}
    return NHPPArrivalModel(rates=arrival_rates, bin_minutes=15)


def test_reposition_moves_vehicles_from_quiet_zone_toward_busy_zone():
    zones = ["quiet", "busy"]
    arrival_model = _skewed_arrival_model(zones, {"quiet": 0.0, "busy": 3.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [Vehicle(vehicle_id=f"v{i}", zone="quiet") for i in range(4)]

    policy = FluidZoneBalancingPolicy(
        dispatch_policy=NearestIdlePolicy(), arrival_model=arrival_model, day_type="all"
    )
    moves = policy.reposition(
        idle_vehicles, current_time=0.0, current_hour=9, travel_time_model=tt_model, zones=zones
    )

    assert len(moves) > 0
    assert all(target_zone == "busy" for _, target_zone in moves)


def test_reposition_is_a_noop_when_already_balanced():
    zones = ["A", "B"]
    arrival_model = _skewed_arrival_model(zones, {"A": 1.0, "B": 1.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [
        Vehicle(vehicle_id="v0", zone="A"),
        Vehicle(vehicle_id="v1", zone="B"),
    ]

    policy = FluidZoneBalancingPolicy(
        dispatch_policy=NearestIdlePolicy(), arrival_model=arrival_model, day_type="all"
    )
    moves = policy.reposition(
        idle_vehicles, current_time=0.0, current_hour=9, travel_time_model=tt_model, zones=zones
    )
    assert moves == []


def test_reposition_returns_empty_when_total_rate_is_zero():
    zones = ["A", "B"]
    arrival_model = _skewed_arrival_model(zones, {"A": 0.0, "B": 0.0})
    tt_model = _flat_travel_time_model(zones)
    idle_vehicles = [Vehicle(vehicle_id="v0", zone="A")]

    policy = FluidZoneBalancingPolicy(
        dispatch_policy=NearestIdlePolicy(), arrival_model=arrival_model, day_type="all"
    )
    moves = policy.reposition(
        idle_vehicles, current_time=0.0, current_hour=9, travel_time_model=tt_model, zones=zones
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
    wrapped = FluidZoneBalancingPolicy(inner, arrival_model, "all").dispatch(
        [request], [v1], 0.0, 9, tt_model
    )
    assert direct == wrapped


def test_reposition_actually_moves_vehicles_in_the_engine():
    zones = ["quiet", "busy"]
    arrival_model = _skewed_arrival_model(zones, {"quiet": 0.0, "busy": 3.0})
    dest_probs = {(z, h): {zz: 1.0 / len(zones) for zz in zones} for z in zones for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(zones) for z in zones})
    tt_model = _flat_travel_time_model(zones)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    vehicles = [Vehicle(vehicle_id=f"v{i}", zone="quiet") for i in range(6)]
    policy = FluidZoneBalancingPolicy(
        dispatch_policy=NearestIdlePolicy(), arrival_model=arrival_model, day_type="all"
    )

    engine = SimulationEngine(
        vehicles=vehicles,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=tt_model,
        abandonment_model=abandonment_model,
        policy=policy,
        zones=zones,
        day_type="all",
        horizon_seconds=1800.0,
        rng=np.random.default_rng(0),
        dispatch_interval_seconds=30.0,
    )
    engine.run()

    assert any(v.zone == "busy" for v in vehicles)
