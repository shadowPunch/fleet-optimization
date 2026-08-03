from __future__ import annotations

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

ZONES = ["P", "Q", "X", "Y"]


def _deterministic_travel_time_model(
    travel_seconds: dict[tuple[str, str], float],
) -> TravelTimeModel:
    """A TravelTimeModel with sigma=0, so .expected() returns exact values."""
    params = {}
    for (o, d), seconds in travel_seconds.items():
        for hour in range(24):
            params[(o, d, hour)] = (float(np.log(seconds)), 0.0)
    return TravelTimeModel(params=params, fallback_params=(float(np.log(3600.0)), 0.0))


def _waiting_request(request_id: str, origin_zone: str, request_time: float = 0.0) -> Request:
    return Request(
        request_id=request_id,
        origin_zone=origin_zone,
        dest_zone=origin_zone,
        request_time=request_time,
        abandon_at=request_time + 10_000.0,
        status=RequestStatus.WAITING,
    )


def test_batched_hungarian_finds_the_globally_optimal_assignment_not_greedy():
    # V1 at P is far from X (10) but close to Y (1); V2 at Q is the mirror
    # image. Greedy nearest-first (processing R1 before R2) would grab V2 for
    # R1 (cost 1) and leave V1 stuck with R2 (cost 10) -> total 11. The
    # globally optimal pairing is V1-R2, V2-R1 -> total 2.
    tt_model = _deterministic_travel_time_model(
        {("P", "X"): 10.0, ("P", "Y"): 1.0, ("Q", "X"): 1.0, ("Q", "Y"): 10.0}
    )
    v1 = Vehicle(vehicle_id="V1", zone="P")
    v2 = Vehicle(vehicle_id="V2", zone="Q")
    r1 = _waiting_request("R1", "X", request_time=0.0)
    r2 = _waiting_request("R2", "Y", request_time=1.0)

    policy = BatchedHungarianPolicy()
    assignments = policy.dispatch(
        [r1, r2], [v1, v2], current_time=5.0, current_hour=9, travel_time_model=tt_model
    )

    assert set(assignments) == {("V1", "R2"), ("V2", "R1")}


def test_matching_radius_excludes_far_pairs_instead_of_forcing_them():
    tt_model = _deterministic_travel_time_model({("P", "X"): 5.0, ("P", "Y"): 500.0})
    v1 = Vehicle(vehicle_id="V1", zone="P")
    r_near = _waiting_request("near", "X")
    r_far = _waiting_request("far", "Y")

    policy = BatchedHungarianPolicy(matching_radius_seconds=60.0)
    assignments = policy.dispatch(
        [r_near, r_far], [v1], current_time=0.0, current_hour=9, travel_time_model=tt_model
    )

    assert assignments == [("V1", "near")]


def test_no_idle_vehicles_or_no_requests_returns_no_assignments():
    tt_model = _deterministic_travel_time_model({})
    policy = BatchedHungarianPolicy()
    assert policy.dispatch([], [Vehicle("V1", "P")], 0.0, 9, tt_model) == []
    assert policy.dispatch([_waiting_request("R1", "X")], [], 0.0, 9, tt_model) == []


def test_batched_hungarian_runs_end_to_end_in_the_engine():
    zones = ["A", "B", "C"]
    rates = {(z, "all", b): 0.4 for z in zones for b in range(96)}
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=15)
    dest_probs = {(z, h): {zz: 1.0 / len(zones) for zz in zones} for z in zones for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(zones) for z in zones})
    tt_model = _deterministic_travel_time_model(
        {(o, d): (60.0 if o != d else 30.0) for o in zones for d in zones}
    )
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)
    config = StudyConfig(
        zones=zones, day_type="all", horizon_seconds=3 * 3600.0, dispatch_interval_seconds=30.0
    )

    result = run_simulation(
        fleet_size=15,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=tt_model,
        abandonment_model=abandonment_model,
        policy=BatchedHungarianPolicy(matching_radius_seconds=600.0),
        config=config,
        rng=np.random.default_rng(0),
    )

    assert result.total_requests > 0
    assert len(result.completed_requests) > 0
    for request in result.completed_requests:
        assert request.pickup_time >= request.request_time
