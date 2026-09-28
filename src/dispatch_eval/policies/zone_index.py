"""Zone-level reductions shared by the dispatch and repositioning policies.

Every cost in this simulator depends on a vehicle only through its zone
(`TravelTimeModel.expected(vehicle.zone, ...)`), so idle vehicles in the same
zone are interchangeable. That lets each policy reason over at most
`n_zones` distinct vehicle "types" instead of every idle vehicle — at NYC
scale (~3,000 idle vehicles, 69 zones) the difference between an O(R·V)
and an O(R·Z) dispatch tick. Each reduction below is exact: it returns an
assignment of the same optimal cost as the unreduced problem.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

from dispatch_eval.simulator.entities import Request, Vehicle

UNREACHABLE_COST = 1e9


def group_by_zone(vehicles: list[Vehicle]) -> dict[str, deque[Vehicle]]:
    """Vehicles bucketed by zone, each bucket in the input's order."""
    buckets: dict[str, deque[Vehicle]] = {}
    for vehicle in vehicles:
        buckets.setdefault(vehicle.zone, deque()).append(vehicle)
    return buckets


def candidate_vehicles(vehicles: list[Vehicle], per_zone_limit: int) -> list[Vehicle]:
    """The first `per_zone_limit` vehicles of each zone, in input order.

    An assignment of `k` requests never uses more than `k` vehicles from any
    one zone, and same-zone vehicles cost the same, so dropping the rest
    cannot raise the optimal assignment cost.
    """
    taken: dict[str, int] = {}
    kept: list[Vehicle] = []
    for vehicle in vehicles:
        n = taken.get(vehicle.zone, 0)
        if n < per_zone_limit:
            taken[vehicle.zone] = n + 1
            kept.append(vehicle)
    return kept


def solve_batched_assignment(
    requests: list[Request],
    vehicles: list[Vehicle],
    pair_cost: Callable[[str, Request], float | None],
) -> list[tuple[str, str]]:
    """Least-cost request↔vehicle matching over one dispatch batch.

    `pair_cost(vehicle_zone, request)` returns the cost of serving `request`
    from `vehicle_zone`, or None if that pair is not allowed (e.g. outside
    a matching radius). It is evaluated once per (zone, request), not once
    per vehicle. Returns (vehicle_id, request_id) pairs.
    """
    if not requests or not vehicles:
        return []

    candidates = candidate_vehicles(vehicles, per_zone_limit=len(requests))
    zones = list(dict.fromkeys(v.zone for v in candidates))
    zone_col = {z: i for i, z in enumerate(zones)}

    zone_cost = np.full((len(requests), len(zones)), UNREACHABLE_COST)
    for i, request in enumerate(requests):
        for j, zone in enumerate(zones):
            cost = pair_cost(zone, request)
            if cost is not None:
                zone_cost[i, j] = cost

    cost = zone_cost[:, [zone_col[v.zone] for v in candidates]]
    rows, cols = linear_sum_assignment(cost)
    return [
        (candidates[j].vehicle_id, requests[i].request_id)
        for i, j in zip(rows, cols, strict=True)
        if cost[i, j] < UNREACHABLE_COST
    ]


def solve_zone_transport(
    supply: dict[str, int],
    demand: dict[str, int],
    zone_cost: Callable[[str, str], float],
) -> list[tuple[str, str, int]]:
    """Min-cost transportation of `min(total supply, total demand)` units
    from supply zones to demand zones.

    The zone-aggregated form of assigning individual vehicles to individual
    targets when both are interchangeable within a zone. Solved as an LP:
    the transportation constraint matrix is totally unimodular, so the
    simplex vertex HiGHS returns is integral. Returns (src, dst, count)
    flows with count > 0.
    """
    src = [z for z, n in supply.items() if n > 0]
    dst = [z for z, n in demand.items() if n > 0]
    if not src or not dst:
        return []

    n_src, n_dst = len(src), len(dst)
    cost = np.array([[zone_cost(s, d) for d in dst] for s in src]).ravel()

    # Row i: sum_j x_ij <= supply_i; column j: sum_i x_ij <= demand_j.
    a_rows = np.kron(np.eye(n_src), np.ones(n_dst))
    a_cols = np.kron(np.ones(n_src), np.eye(n_dst))
    a_ub = np.vstack([a_rows, a_cols])
    b_ub = np.array([supply[s] for s in src] + [demand[d] for d in dst], dtype=float)
    total = float(min(sum(supply[s] for s in src), sum(demand[d] for d in dst)))

    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n_src * n_dst)),
        b_eq=[total],
        bounds=(0, None),
        method="highs-ds",
    )
    if not result.success:
        raise RuntimeError(f"zone transport LP failed: {result.message}")

    flows = np.rint(result.x).astype(int).reshape(n_src, n_dst)
    return [
        (src[i], dst[j], int(flows[i, j]))
        for i in range(n_src)
        for j in range(n_dst)
        if flows[i, j] > 0
    ]
