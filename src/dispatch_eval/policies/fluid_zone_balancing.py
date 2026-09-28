"""B3 — B2's value-corrected dispatch + fluid zone-balancing repositioning.

Dispatch is delegated unchanged to whichever policy is wrapped (the plan's
B3 = B2 + repositioning, so that's a `ValueCorrectedHungarianPolicy` in
practice, but this class doesn't care which policy it wraps). The new piece
is `reposition`: after each dispatch tick's assignments are made, whatever
vehicles are *still* idle get proactively redirected using a fluid
(continuous-relaxation) target allocation — each zone's target share of idle
vehicles is proportional to its share of total expected arrival rate right
now, and vehicles move from over-supplied zones to under-supplied ones via
the same kind of least-cost bipartite assignment B1/B2 use for dispatch,
just matching movers to destination zones instead of vehicles to requests.

This is a genuinely different mechanism from B2's per-request value
correction: B2 decides which vehicle serves which request; B3 additionally
decides where a vehicle with nothing to do right now should go, based on the
aggregate demand shape rather than any one rider's destination.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from dispatch_eval.models import NHPPArrivalModel, TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.policies.zone_index import group_by_zone, solve_zone_transport
from dispatch_eval.simulator.entities import Request, Vehicle


def largest_remainder_allocation(target_shares: dict[str, float], total: int) -> dict[str, int]:
    """Round continuous per-zone target shares to integers summing to exactly `total`.

    Standard largest-remainder (Hamilton) apportionment: floor every raw
    target, then hand out the leftover units to whichever zones had the
    biggest fractional part, so the rounded targets don't drift from the
    fleet's actual idle count.
    """
    raw = {z: share * total for z, share in target_shares.items()}
    floors = {z: int(np.floor(v)) for z, v in raw.items()}
    remainder = total - sum(floors.values())
    by_fractional_part_desc = sorted(raw, key=lambda z: raw[z] - floors[z], reverse=True)
    for z in by_fractional_part_desc[:remainder]:
        floors[z] += 1
    return floors


class FluidZoneBalancingPolicy:
    def __init__(
        self,
        dispatch_policy: DispatchPolicy,
        arrival_model: NHPPArrivalModel,
        day_type: str,
    ):
        self.dispatch_policy = dispatch_policy
        self.arrival_model = arrival_model
        self.day_type = day_type

    def dispatch(
        self,
        waiting_requests: list[Request],
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
    ) -> list[tuple[str, str]]:
        return self.dispatch_policy.dispatch(
            waiting_requests, idle_vehicles, current_time, current_hour, travel_time_model
        )

    def reposition(
        self,
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
        zones: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, str]]:
        del rng  # fluid balancing is deterministic given current state; unused here
        if not idle_vehicles:
            return []

        minute_of_day = (current_time / 60.0) % (24 * 60)
        rates = {
            z: self.arrival_model.rate_per_minute(z, self.day_type, minute_of_day) for z in zones
        }
        total_rate = sum(rates.values())
        if total_rate <= 0:
            return []

        target_shares = {z: rates[z] / total_rate for z in zones}
        target_counts = largest_remainder_allocation(target_shares, len(idle_vehicles))

        current_counts = Counter(v.zone for v in idle_vehicles)
        surplus = {z: n - target_counts.get(z, 0) for z, n in current_counts.items()}
        deficit = {z: target_counts.get(z, 0) - current_counts[z] for z in zones}

        # Surplus vehicles move to deficit zones as a zone-level
        # transportation problem (see zone_index) — the same optimum as a
        # vehicle-by-slot assignment, without the O(V^2) matrix.
        flows = solve_zone_transport(
            surplus, deficit, lambda s, d: travel_time_model.expected(s, d, current_hour)
        )
        by_zone = group_by_zone(idle_vehicles)
        return [
            (by_zone[src].popleft().vehicle_id, dst)
            for src, dst, count in flows
            for _ in range(count)
        ]
