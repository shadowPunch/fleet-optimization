from __future__ import annotations

import numpy as np
import pytest

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
    # priority over shaving off wait time.
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


def test_relaxation_can_over_serve_a_true_simultaneous_capacity_tie():
    # Two requests want the same vehicle, in the same zone, at the same
    # moment — true capacity allows at most one. The earliest-feasible-
    # release relaxation (module docstring: RELAXATION, not APPROXIMATION)
    # deliberately doesn't enforce this tightly: a served request's vehicle
    # re-enters the graph at its *earliest possible* pickup time + trip
    # duration, regardless of when the flow actually picks it up — trading
    # tightness for a genuine upper-bound guarantee (over-serve relative to
    # true capacity, never under-serve relative to true optimal). Both
    # requests being "served" here is that trade-off working as intended,
    # not a correctness bug.
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

    assert result.served_request_ids == {"r1", "r2"}


def test_long_patience_window_no_longer_starves_a_downstream_request():
    # Direct regression check for the relaxation's actual purpose. Under
    # the old pessimistic-pinning approximation, "near"'s long patience
    # window (3800s) tied up its vehicle until that deadline regardless of
    # how quickly it was really picked up, which pushed the vehicle's
    # reachable-by time for "far" past far's own deadline (3700s) —
    # confirmed by re-running this exact scenario against the old
    # t_abandon-pinned formula, which serves only {"near"}. The relaxed,
    # earliest-feasible-release version serves both: near's vehicle
    # re-enters the graph based on its *earliest* possible pickup, not its
    # patience window, so a long patience window on an earlier request no
    # longer costs a later one its capacity.
    zones = ["A", "B", "C"]
    tt_model = _flat_travel_time_model(zones, seconds=120.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [
        _request("near", "A", "B", request_time=0.0, abandon_at=3800.0),
        _request("far", "C", "C", request_time=0.0, abandon_at=3700.0),
    ]

    result = solve_clairvoyant_schedule(
        requests, vehicles, tt_model, horizon_seconds=4000.0, bin_minutes=2.0
    )

    assert result.served_request_ids == {"near", "far"}


def test_empty_requests_or_vehicles_returns_trivially():
    zones = ["A"]
    tt_model = _flat_travel_time_model(zones)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "A", "A", 0.0, 600.0)]

    assert solve_clairvoyant_schedule([], vehicles, tt_model, 600.0).total_requests == 0
    assert solve_clairvoyant_schedule(requests, [], tt_model, 600.0).fraction_served == 0.0


def test_coarse_bin_minutes_excludes_short_patience_requests_before_solving():
    # request_time=1s, abandon_at=899s: a ~15-minute patience window that
    # straddles a 900s (15-min) bin boundary awkwardly. At bin_minutes=15,
    # t_req=ceil(1/900)=1 (the *earliest* pickup bin starts at t=900, since
    # any request_time > 0 within bin 0 ceils up to bin 1) but
    # t_abandon=floor(899/900)=0 -- excluded (0 < 1) purely by rounding, not
    # because the window is actually too short: at bin_minutes=1, the same
    # request has t_req=1, t_abandon=14, comfortably servable.
    zones = ["A"]
    tt_model = _flat_travel_time_model(zones, seconds=10.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "A", "A", request_time=1.0, abandon_at=899.0)]

    coarse = solve_clairvoyant_schedule(requests, vehicles, tt_model, horizon_seconds=1800.0, bin_minutes=15.0)
    fine = solve_clairvoyant_schedule(requests, vehicles, tt_model, horizon_seconds=1800.0, bin_minutes=1.0)

    assert coarse.excluded_by_discretization == 1
    assert coarse.fraction_excluded_by_discretization == pytest.approx(1.0)
    assert "r0" not in coarse.served_request_ids

    assert fine.excluded_by_discretization == 0
    assert "r0" in fine.served_request_ids  # the same request is genuinely servable at a finer grid


def test_no_discretization_exclusion_when_windows_fit_comfortably():
    zones = ["A"]
    tt_model = _flat_travel_time_model(zones, seconds=10.0)
    vehicles = [Vehicle(vehicle_id="v0", zone="A")]
    requests = [_request("r0", "A", "A", request_time=0.0, abandon_at=600.0)]

    result = solve_clairvoyant_schedule(requests, vehicles, tt_model, horizon_seconds=1200.0, bin_minutes=1.0)

    assert result.excluded_by_discretization == 0
    assert result.fraction_excluded_by_discretization == 0.0
