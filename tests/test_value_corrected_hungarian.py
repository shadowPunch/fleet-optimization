from __future__ import annotations

import numpy as np

from dispatch_eval.calibration.value_function import ValueFunction, compute_value_function
from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

ZONES = ["P", "Good", "Bad"]


def _flat_travel_time_model(seconds: float) -> TravelTimeModel:
    params = {
        (o, d, h): (float(np.log(seconds)), 0.0) for o in ZONES for d in ZONES for h in range(24)
    }
    return TravelTimeModel(params=params, fallback_params=(float(np.log(seconds)), 0.0))


def _waiting_request(request_id: str, origin: str, dest: str, request_time: float = 0.0) -> Request:
    return Request(
        request_id=request_id,
        origin_zone=origin,
        dest_zone=dest,
        request_time=request_time,
        abandon_at=request_time + 10_000.0,
        status=RequestStatus.WAITING,
    )


def test_value_weight_zero_ignores_the_value_function_entirely():
    tt_model = _flat_travel_time_model(60.0)
    # A value function that would strongly prefer "Good" over "Bad" if it mattered.
    vf = ValueFunction(values={("Good", 0): 0.0, ("Bad", 0): 100_000.0}, bin_minutes=15, n_bins=10)
    v1 = Vehicle(vehicle_id="V1", zone="P")
    r_good = _waiting_request("r_good", "P", "Good")
    r_bad = _waiting_request("r_bad", "P", "Bad")

    policy = ValueCorrectedHungarianPolicy(value_function=vf, value_weight=0.0)
    assignments = policy.dispatch(
        [r_good, r_bad], [v1], current_time=0.0, current_hour=9, travel_time_model=tt_model
    )
    # tied on pure pickup time; with value_weight=0 either is a valid optimum,
    # but the assignment must actually happen (someone gets served).
    assert len(assignments) == 1
    assert assignments[0][0] == "V1"


def test_value_correction_prefers_the_higher_value_destination_when_tied_on_pickup():
    tt_model = _flat_travel_time_model(60.0)
    vf = ValueFunction(values={("Good", 0): 0.0, ("Bad", 0): 100_000.0}, bin_minutes=15, n_bins=10)
    v1 = Vehicle(vehicle_id="V1", zone="P")
    r_good = _waiting_request("r_good", "P", "Good")
    r_bad = _waiting_request("r_bad", "P", "Bad")

    policy = ValueCorrectedHungarianPolicy(value_function=vf, value_weight=1.0)
    assignments = policy.dispatch(
        [r_good, r_bad], [v1], current_time=0.0, current_hour=9, travel_time_model=tt_model
    )
    assert assignments == [("V1", "r_good")]


def test_matching_radius_still_excludes_far_pairs():
    # Origins differ from the vehicle's zone here (unlike the two tests
    # above), so pickup time comes from the real params lookup rather than
    # TravelTimeModel's same-zone special case, and the override actually
    # bites on pickup distance, not just the onward trip.
    tt_model = _flat_travel_time_model(60.0)
    tt_model.params[("P", "Bad", 9)] = (float(np.log(5000.0)), 0.0)
    vf = ValueFunction(values={}, bin_minutes=15, n_bins=10)
    v1 = Vehicle(vehicle_id="V1", zone="P")
    r_near = _waiting_request("r_near", "Good", "Good")
    r_far = _waiting_request("r_far", "Bad", "Good")

    policy = ValueCorrectedHungarianPolicy(
        value_function=vf, value_weight=1.0, matching_radius_seconds=200.0
    )
    assignments = policy.dispatch(
        [r_near, r_far], [v1], current_time=0.0, current_hour=9, travel_time_model=tt_model
    )
    assert assignments == [("V1", "r_near")]


def test_value_corrected_policy_runs_end_to_end_in_the_engine():
    zones = ["A", "B", "C"]
    rates = {(z, "all", b): 0.4 for z in zones for b in range(96)}
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=15)
    dest_probs = {(z, h): {zz: 1.0 / len(zones) for zz in zones} for z in zones for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(zones) for z in zones})
    params = {
        (o, d, h): (float(np.log(60.0)), 0.3) for o in zones for d in zones for h in range(24)
    }
    tt_model = TravelTimeModel(params=params, fallback_params=(float(np.log(60.0)), 0.3))
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    horizon = 3 * 3600.0
    value_function = compute_value_function(
        zones, arrival_model, od_model, tt_model, day_type="all", horizon_seconds=horizon
    )
    config = StudyConfig(
        zones=zones, day_type="all", horizon_seconds=horizon, dispatch_interval_seconds=30.0
    )

    result = run_simulation(
        fleet_size=15,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=tt_model,
        abandonment_model=abandonment_model,
        policy=ValueCorrectedHungarianPolicy(value_function=value_function, value_weight=0.5),
        config=config,
        rng=np.random.default_rng(0),
    )

    assert result.total_requests > 0
    assert len(result.completed_requests) > 0
    for request in result.completed_requests:
        assert request.pickup_time >= request.request_time
