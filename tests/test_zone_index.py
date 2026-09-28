"""The zone-level reductions must be exact: same picks (nearest idle) or the
same optimal cost (assignment, transport) as brute force over individual
vehicles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.zone_index import (
    UNREACHABLE_COST,
    candidate_vehicles,
    solve_batched_assignment,
    solve_zone_transport,
)
from dispatch_eval.simulator.entities import Request, Vehicle

ZONES = [f"z{i}" for i in range(6)]


def _random_instance(seed: int, n_req: int, n_veh: int):
    rng = np.random.default_rng(seed)
    # Integer-valued expected times so exact ties between zones actually occur.
    params = {
        (o, d, 0): (float(np.log(rng.integers(1, 5) * 60)), 0.0) for o in ZONES for d in ZONES
    }
    tt = TravelTimeModel(params=params, fallback_params=(np.log(300.0), 0.0))
    requests = [
        Request(f"r{i}", str(rng.choice(ZONES)), str(rng.choice(ZONES)), float(i), 1e9)
        for i in range(n_req)
    ]
    vehicles = [Vehicle(f"v{j}", str(rng.choice(ZONES))) for j in range(n_veh)]
    return tt, requests, vehicles


def _brute_force_nearest_idle(requests, vehicles, tt):
    available = list(vehicles)
    out = []
    for request in sorted(requests, key=lambda r: r.request_time):
        if not available:
            break
        best = min(available, key=lambda v: tt.expected(v.zone, request.origin_zone, 0))
        out.append((best.vehicle_id, request.request_id))
        available.remove(best)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_nearest_idle_matches_brute_force_exactly(seed):
    tt, requests, vehicles = _random_instance(seed, n_req=8, n_veh=12)
    fast = NearestIdlePolicy().dispatch(requests, vehicles, 0.0, 0, tt)
    assert fast == _brute_force_nearest_idle(requests, vehicles, tt)


def _brute_force_cost(requests, vehicles, pair_cost):
    size = max(len(requests), len(vehicles))
    cost = np.full((size, size), UNREACHABLE_COST)
    for i, r in enumerate(requests):
        for j, v in enumerate(vehicles):
            c = pair_cost(v.zone, r)
            if c is not None:
                cost[i, j] = c
    rows, cols = linear_sum_assignment(cost)
    kept = [cost[i, j] for i, j in zip(rows, cols, strict=True) if cost[i, j] < UNREACHABLE_COST]
    return len(kept), sum(kept)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("radius", [float("inf"), 150.0])
def test_batched_assignment_reaches_brute_force_optimum(seed, radius):
    tt, requests, vehicles = _random_instance(seed, n_req=7, n_veh=15)

    def pair_cost(zone, request):
        t = tt.expected(zone, request.origin_zone, 0)
        return t if t <= radius else None

    pairs = solve_batched_assignment(requests, vehicles, pair_cost)
    by_id_v = {v.vehicle_id: v for v in vehicles}
    by_id_r = {r.request_id: r for r in requests}
    fast_cost = sum(pair_cost(by_id_v[v].zone, by_id_r[r]) for v, r in pairs)

    n_brute, brute_cost = _brute_force_cost(requests, vehicles, pair_cost)
    assert len(pairs) == n_brute
    assert fast_cost == pytest.approx(brute_cost)
    assert len({v for v, _ in pairs}) == len(pairs)  # no vehicle used twice


def test_candidate_vehicles_keeps_first_k_per_zone_in_order():
    vehicles = [Vehicle(f"v{i}", z) for i, z in enumerate("AABABBC")]
    kept = candidate_vehicles(vehicles, per_zone_limit=2)
    assert [v.vehicle_id for v in kept] == ["v0", "v1", "v2", "v4", "v6"]


@pytest.mark.parametrize("seed", range(20))
def test_zone_transport_reaches_expanded_assignment_optimum(seed):
    rng = np.random.default_rng(seed)
    cost_table = {(s, d): float(rng.integers(0, 10)) for s in ZONES for d in ZONES}
    supply = {z: int(rng.integers(0, 4)) for z in ZONES}
    demand = {z: int(rng.integers(0, 4)) for z in ZONES}

    flows = solve_zone_transport(supply, demand, lambda s, d: cost_table[(s, d)])

    units_src = [z for z, n in supply.items() for _ in range(n)]
    units_dst = [z for z, n in demand.items() for _ in range(n)]
    expected_units = min(len(units_src), len(units_dst))
    if expected_units == 0:
        assert flows == []
        return
    expanded = np.array([[cost_table[(s, d)] for d in units_dst] for s in units_src])
    rows, cols = linear_sum_assignment(expanded)

    assert sum(c for _, _, c in flows) == expected_units
    assert sum(cost_table[(s, d)] * c for s, d, c in flows) == pytest.approx(
        expanded[rows, cols].sum()
    )
    for zone in ZONES:
        assert sum(c for s, _, c in flows if s == zone) <= supply[zone]
        assert sum(c for _, d, c in flows if d == zone) <= demand[zone]
