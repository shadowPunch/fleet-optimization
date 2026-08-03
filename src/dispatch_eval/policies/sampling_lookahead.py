"""B4 — B3's capabilities, but repositioning is driven by a sampled lookahead
instead of the fluid (expected-value) target allocation.

Dispatch is delegated unchanged to whatever policy is wrapped (B2 in
practice, same as B3). The difference from `FluidZoneBalancingPolicy` is
entirely in `reposition`: instead of computing a smooth, deterministic
target share of idle vehicles per zone from expected arrival rates, this
draws one Monte Carlo sample of the requests that might actually arrive over
the next `lookahead_seconds` (via the same arrival model the engine itself
uses to generate real arrivals — `models.NHPPArrivalModel.generate_arrival_minutes`)
and solves a direct least-cost assignment between currently idle vehicles
and those sampled future requests' origin zones — reusing the same Hungarian
machinery B1/B2 use for real dispatch, just with sampled phantom demand
standing in for real waiting requests.

This means B4 can react to a specific, lumpy realization of where demand
happens to land (e.g. several requests bunched in one zone this particular
draw), which the fluid model's smooth average can't represent — at the cost
of being noisier tick to tick, since it's driven by one sample rather than
an expectation. Only origin zones matter here (not destinations): the
question repositioning answers is "where will the *next pickup* be needed,"
not where any given rider is going.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from dispatch_eval.models import NHPPArrivalModel, TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.simulator.entities import Request, Vehicle


class SamplingLookaheadPolicy:
    def __init__(
        self,
        dispatch_policy: DispatchPolicy,
        arrival_model: NHPPArrivalModel,
        day_type: str,
        lookahead_seconds: float,
    ):
        self.dispatch_policy = dispatch_policy
        self.arrival_model = arrival_model
        self.day_type = day_type
        self.lookahead_seconds = lookahead_seconds

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
        if not idle_vehicles:
            return []

        start_minute = current_time / 60.0
        end_minute = (current_time + self.lookahead_seconds) / 60.0

        sampled_zones: list[str] = []
        for zone in zones:
            arrival_minutes = self.arrival_model.generate_arrival_minutes(
                zone, self.day_type, start_minute, end_minute, rng
            )
            sampled_zones.extend([zone] * len(arrival_minutes))

        if not sampled_zones:
            return []

        cost = np.array(
            [
                [travel_time_model.expected(v.zone, z, current_hour) for z in sampled_zones]
                for v in idle_vehicles
            ]
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        moves = []
        for i, j in zip(row_ind, col_ind, strict=True):
            vehicle, target_zone = idle_vehicles[i], sampled_zones[j]
            if target_zone != vehicle.zone:  # already there; nothing to do
                moves.append((vehicle.vehicle_id, target_zone))
        return moves
