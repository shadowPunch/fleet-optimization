from __future__ import annotations

import numpy as np
import pytest

from dispatch_eval.compute_parity import (
    LatencyMeasurement,
    feasible_within_budget,
    latency_frontier,
    measure_decision_latency,
)
from dispatch_eval.models import TravelTimeModel
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.entities import Request, Vehicle

ZONES = ["A", "B", "C"]


def _travel_time_model() -> TravelTimeModel:
    return TravelTimeModel(params={}, fallback_params=(float(np.log(300.0)), 0.3))


def _flat_arrival_model():
    from dispatch_eval.models import NHPPArrivalModel

    rates = {(z, "all", b): 2.0 for z in ZONES for b in range(96)}
    return NHPPArrivalModel(rates=rates, bin_minutes=15)


# --- measure_decision_latency -------------------------------------------------


def test_measure_decision_latency_reports_correct_problem_size():
    m = measure_decision_latency(
        NearestIdlePolicy(),
        n_requests=10,
        n_idle_vehicles=8,
        zones=ZONES,
        travel_time_model=_travel_time_model(),
        rng=np.random.default_rng(0),
        n_repeats=3,
    )
    assert m.n_requests == 10
    assert m.n_idle_vehicles == 8
    assert m.dispatch_seconds >= 0.0
    assert m.total_seconds == pytest.approx(m.dispatch_seconds)


def test_measure_decision_latency_zero_reposition_for_policy_without_one():
    assert not hasattr(NearestIdlePolicy(), "reposition")
    m = measure_decision_latency(
        NearestIdlePolicy(),
        n_requests=5,
        n_idle_vehicles=5,
        zones=ZONES,
        travel_time_model=_travel_time_model(),
        rng=np.random.default_rng(0),
        n_repeats=3,
    )
    assert m.reposition_seconds == 0.0


def test_measure_decision_latency_measures_reposition_when_policy_has_one():
    wrapped = FluidZoneBalancingPolicy(
        dispatch_policy=NearestIdlePolicy(), arrival_model=_flat_arrival_model(), day_type="all"
    )
    assert hasattr(wrapped, "reposition")
    m = measure_decision_latency(
        wrapped,
        n_requests=0,
        n_idle_vehicles=6,
        zones=ZONES,
        travel_time_model=_travel_time_model(),
        rng=np.random.default_rng(0),
        n_repeats=3,
    )
    assert m.reposition_seconds >= 0.0
    assert m.total_seconds == pytest.approx(m.dispatch_seconds + m.reposition_seconds)


def test_measure_decision_latency_handles_empty_problem_sizes():
    m = measure_decision_latency(
        NearestIdlePolicy(),
        n_requests=0,
        n_idle_vehicles=0,
        zones=ZONES,
        travel_time_model=_travel_time_model(),
        rng=np.random.default_rng(0),
        n_repeats=2,
    )
    assert m.n_requests == 0
    assert m.n_idle_vehicles == 0
    assert m.dispatch_seconds >= 0.0


# --- latency_frontier / feasible_within_budget ---------------------------------


def test_latency_frontier_shapes():
    policies = {"B0": NearestIdlePolicy(), "B0_copy": NearestIdlePolicy()}
    problem_sizes = [(5, 5), (10, 10), (20, 15)]

    frontier = latency_frontier(
        policies, problem_sizes, ZONES, _travel_time_model(), seed=0, n_repeats=2
    )

    assert set(frontier.keys()) == {"B0", "B0_copy"}
    for name in frontier:
        assert len(frontier[name]) == len(problem_sizes)
        for measurement, (n_req, n_veh) in zip(frontier[name], problem_sizes, strict=True):
            assert measurement.n_requests == n_req
            assert measurement.n_idle_vehicles == n_veh


def test_feasible_within_budget_flags_over_budget_measurements():
    measurements = {
        "fast": [
            LatencyMeasurement(5, 5, dispatch_seconds=0.1, reposition_seconds=0.0),
            LatencyMeasurement(50, 50, dispatch_seconds=0.5, reposition_seconds=0.0),
        ],
        "slow": [
            LatencyMeasurement(5, 5, dispatch_seconds=1.0, reposition_seconds=0.5),
            LatencyMeasurement(50, 50, dispatch_seconds=5.0, reposition_seconds=1.0),
        ],
    }
    feasible = feasible_within_budget(measurements, budget_seconds=1.0)

    assert feasible["fast"] == [True, True]  # 0.1 and 0.5 both <= 1.0
    assert feasible["slow"] == [False, False]  # 1.0+0.5=1.5 and 5.0+1.0=6.0 both > 1.0


# --- empirical basis for the module docstring's compute-parity finding ---------


def test_matching_radius_does_not_shrink_the_solved_cost_matrix(monkeypatch):
    """B1's `matching_radius_seconds` masks unreachable pairs with a large
    cost rather than removing them from the problem — the assignment matrix
    `scipy.optimize.linear_sum_assignment` actually solves is always
    n_requests x candidate vehicles (see `zone_index`), regardless of the radius. So
    sweeping the radius changes assignment *quality*, not wall-clock
    *latency* — confirmed here directly (matrix shape), not inferred from
    timing, since timing comparisons are inherently noisier.
    """
    import dispatch_eval.policies.batched_hungarian as bh
    import dispatch_eval.policies.zone_index as zi

    seen_shapes = []
    real_lsa = zi.linear_sum_assignment

    def spy(cost):
        seen_shapes.append(cost.shape)
        return real_lsa(cost)

    monkeypatch.setattr(zi, "linear_sum_assignment", spy)

    requests = [
        Request(f"req-{i}", ZONES[i % 3], ZONES[(i + 1) % 3], 0.0, 600.0) for i in range(20)
    ]
    vehicles = [Vehicle(f"veh-{i}", ZONES[i % 3]) for i in range(20)]
    travel_time_model = _travel_time_model()

    bh.BatchedHungarianPolicy(matching_radius_seconds=float("inf")).dispatch(
        requests, vehicles, 0.0, 12, travel_time_model
    )
    bh.BatchedHungarianPolicy(matching_radius_seconds=1.0).dispatch(
        requests, vehicles, 0.0, 12, travel_time_model
    )

    assert seen_shapes == [(20, 20), (20, 20)]
