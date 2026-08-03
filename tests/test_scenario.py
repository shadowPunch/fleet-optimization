from __future__ import annotations

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.scenario import generate_scenario
from dispatch_eval.simulator.engine import SimulationEngine
from dispatch_eval.simulator.entities import Vehicle

ZONES = ["A", "B", "C"]


def _models():
    rates = {(z, "all", b): 0.6 for z in ZONES for b in range(96)}
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=15)
    dest_probs = {(z, h): {zz: 1.0 / len(ZONES) for zz in ZONES} for z in ZONES for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(ZONES) for z in ZONES})
    params = {
        (o, d, h): (float(np.log(300.0)), 0.3) for o in ZONES for d in ZONES for h in range(24)
    }
    tt_model = TravelTimeModel(params=params, fallback_params=(float(np.log(300.0)), 0.3))
    abandonment_model = AbandonmentModel(mean_patience_seconds=180.0)
    return arrival_model, od_model, tt_model, abandonment_model


def _request_tuples(scenario):
    return [
        (r.request_id, r.origin_zone, r.dest_zone, r.request_time, r.abandon_at)
        for r in scenario.requests
    ]


def test_generate_scenario_is_deterministic_given_the_same_seed():
    arrival_model, od_model, _, abandonment_model = _models()

    scenario_a = generate_scenario(
        ZONES, "all", arrival_model, od_model, abandonment_model, 3600.0, np.random.default_rng(123)
    )
    scenario_b = generate_scenario(
        ZONES, "all", arrival_model, od_model, abandonment_model, 3600.0, np.random.default_rng(123)
    )

    assert _request_tuples(scenario_a) == _request_tuples(scenario_b)
    assert len(scenario_a.requests) > 0  # otherwise the test above is vacuous


def test_generate_scenario_does_not_depend_on_the_input_zones_list_order():
    arrival_model, od_model, _, abandonment_model = _models()

    forward = generate_scenario(
        ["A", "B", "C"],
        "all",
        arrival_model,
        od_model,
        abandonment_model,
        3600.0,
        np.random.default_rng(5),
    )
    reversed_order = generate_scenario(
        ["C", "B", "A"],
        "all",
        arrival_model,
        od_model,
        abandonment_model,
        3600.0,
        np.random.default_rng(5),
    )

    assert _request_tuples(forward) == _request_tuples(reversed_order)


def test_realized_request_trace_is_identical_across_different_policies_given_the_same_seed():
    # The actual regression test for the bug. Note that comparing just
    # `total_requests` would NOT catch it: arrival *timing* was already
    # generated upfront even before this fix, so the count was always
    # policy-independent — it was each request's *destination and patience*
    # (drawn reactively, interleaved with policy-dependent travel-time
    # samples from the same rng) that used to differ. Confirmed empirically
    # against the pre-fix engine: with an identical seed, 80 of 111 realized
    # requests had a different destination and/or abandon_at between
    # NearestIdlePolicy and BatchedHungarianPolicy. This test checks the full
    # tuple, not just the count, so it would have caught that.
    arrival_model, od_model, tt_model, abandonment_model = _models()

    def realized_requests(policy, seed):
        rng = np.random.default_rng(seed)
        scenario = generate_scenario(
            ZONES, "all", arrival_model, od_model, abandonment_model, 3600.0, rng
        )
        vehicles = [Vehicle(vehicle_id=f"veh-{i}", zone=ZONES[i % len(ZONES)]) for i in range(8)]
        engine = SimulationEngine(
            vehicles=vehicles,
            scenario=scenario,
            travel_time_model=tt_model,
            policy=policy,
            zones=ZONES,
            horizon_seconds=3600.0,
            rng=rng,
        )
        engine.run()
        return [
            (r.request_id, r.origin_zone, r.dest_zone, r.request_time, r.abandon_at)
            for r in sorted(engine.requests.values(), key=lambda r: r.request_id)
        ]

    nearest_trace = realized_requests(NearestIdlePolicy(), seed=77)
    batched_trace = realized_requests(BatchedHungarianPolicy(), seed=77)

    assert nearest_trace == batched_trace
    assert len(nearest_trace) > 0
