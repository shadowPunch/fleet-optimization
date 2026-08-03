from __future__ import annotations

import numpy as np

from dispatch_eval.clairvoyant import solve_clairvoyant_schedule
from dispatch_eval.models import TravelTimeModel
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle


def _flat_travel_time_model(zones, seconds: float = 300.0) -> TravelTimeModel:
    params = {
        (o, d, h): (float(np.log(seconds)), 0.0) for o in zones for d in zones for h in range(24)
    }
    return TravelTimeModel(params=params, fallback_params=(float(np.log(seconds)), 0.0))


def _request(request_id, origin, dest, request_time, abandon_at):
    return Request(
        request_id=request_id,
        origin_zone=origin,
        dest_zone=dest,
        request_time=request_time,
        abandon_at=abandon_at,
        status=RequestStatus.WAITING,
    )


def test_trivial_same_zone_request_is_served_with_near_zero_wait():
    zones = ["A", "B"]
    tt_model = _flat_travel_time_model(zones)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "A", "B", request_time=0.0, abandon_at=1200.0)]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=3600.0, bin_minutes=5.0
    )

    assert result.served_request_ids == {"r0"}
    assert result.wait_seconds["r0"] == 0.0


def test_infeasible_request_is_not_served():
    # Patience (60s) is far shorter than the time it would take a vehicle to
    # reposition from A to B (300s), so this request cannot be served no
    # matter what.
    zones = ["A", "B"]
    tt_model = _flat_travel_time_model(zones, seconds=300.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "B", "A", request_time=0.0, abandon_at=60.0)]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=1200.0, bin_minutes=1.0
    )

    assert result.served_request_ids == set()
    assert result.fraction_served == 0.0


def test_single_vehicle_chains_two_sequential_requests():
    # Only one vehicle exists, so if request 2 gets served at all, it must
    # be by the same vehicle that served request 1 — this is exactly the
    # capability a one-shot bipartite match doesn't have.
    zones = ["A", "B"]
    tt_model = _flat_travel_time_model(zones, seconds=300.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [
        _request("r1", "A", "B", request_time=0.0, abandon_at=300.0),
        _request("r2", "B", "A", request_time=600.0, abandon_at=900.0),
    ]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=1200.0, bin_minutes=5.0
    )

    assert result.served_request_ids == {"r1", "r2"}


def test_maximizing_requests_served_beats_minimizing_wait():
    # Two requests both reachable, but only one vehicle: serving both
    # requires a longer total wait than serving just the closer one. The
    # bonus should still make serving both (or as many as feasible) the
    # priority over shaving off wait time. "near"'s own patience is kept
    # short deliberately: the pessimistic-exit approximation (see the module
    # docstring) ties up the serving vehicle until *near's own* abandon
    # deadline regardless of how quickly it's actually picked up, so a long
    # patience window on the *first* request in a chain would starve the
    # second one out of the horizon — this is that approximation's real
    # cost, not a bug, and the test numbers below account for it.
    zones = ["A", "B", "C"]
    tt_model = _flat_travel_time_model(zones, seconds=120.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [
        _request("near", "A", "A", request_time=0.0, abandon_at=120.0),
        _request("far", "C", "C", request_time=0.0, abandon_at=3600.0),
    ]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=4000.0, bin_minutes=2.0
    )

    # Only one vehicle: it serves "near" immediately, then must reposition
    # all the way to C and accept a substantial wait to also serve "far" —
    # accepting that wait should still beat leaving "far" unserved.
    assert result.served_request_ids == {"near", "far"}
    assert result.wait_seconds["far"] > 0


def test_capacity_limits_how_many_requests_one_vehicle_can_serve_at_once():
    zones = ["A"]
    tt_model = _flat_travel_time_model(zones, seconds=60.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [
        _request("r1", "A", "A", request_time=0.0, abandon_at=60.0),
        _request("r2", "A", "A", request_time=0.0, abandon_at=60.0),
    ]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=600.0, bin_minutes=1.0
    )

    # Both requests want a vehicle at the exact same place and time; only
    # one vehicle exists, so at most one can be served right then.
    assert len(result.served_request_ids) <= 1


def test_empty_requests_or_vehicles_returns_trivially():
    zones = ["A"]
    tt_model = _flat_travel_time_model(zones)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "A", "A", 0.0, 600.0)]

    assert solve_clairvoyant_schedule([], vehicles, tt_model, 600.0).total_requests == 0
    assert solve_clairvoyant_schedule(requests, [], tt_model, 600.0).fraction_served == 0.0
