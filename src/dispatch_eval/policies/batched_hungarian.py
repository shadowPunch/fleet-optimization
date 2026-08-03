"""B1 — batched bipartite matching (Hungarian), with a tunable batch window and matching radius.

Unlike B0's greedy immediate assignment, B1 solves one globally least-cost
assignment over everyone waiting at each dispatch tick, instead of matching
requests one at a time in arrival order. Two parameters control it, both
meant to be tuned under an identical budget (see `calibration.tuning`) rather
than hand-picked:

- **Batch window Δ** is the engine's `dispatch_interval_seconds`, not a
  parameter of this class — a bigger Δ lets more requests and vehicles
  accumulate before each batch is solved, trading dispatch latency for match
  quality.
- **Matching radius r** (`matching_radius_seconds`) excludes any
  (vehicle, request) pair whose expected travel time exceeds it. Better to
  leave a request waiting for the next tick than force a very long pickup
  just because the solver had to assign someone to every column.

The plan also describes a "B1'" variant using closed-form guidance for the
jointly-optimal Δ and r from the batching literature; that specific formula
hasn't been sourced yet, so B1 here is tuned empirically (random search over
Δ, r — see calibration.tuning) rather than set analytically.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.simulator.entities import Request, Vehicle

_UNREACHABLE_COST = 1e9


class BatchedHungarianPolicy:
    def __init__(self, matching_radius_seconds: float = float("inf")):
        self.matching_radius_seconds = matching_radius_seconds

    def dispatch(
        self,
        waiting_requests: list[Request],
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
    ) -> list[tuple[str, str]]:
        if not waiting_requests or not idle_vehicles:
            return []

        n_req, n_veh = len(waiting_requests), len(idle_vehicles)
        size = max(n_req, n_veh)
        # Square, padded with a large cost so linear_sum_assignment (which
        # requires a square matrix) never prefers a padding "ghost" match
        # over a real one, and so out-of-radius pairs are excluded in effect.
        cost = np.full((size, size), _UNREACHABLE_COST)
        for i, request in enumerate(waiting_requests):
            for j, vehicle in enumerate(idle_vehicles):
                travel = travel_time_model.expected(vehicle.zone, request.origin_zone, current_hour)
                if travel <= self.matching_radius_seconds:
                    cost[i, j] = travel

        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = []
        for i, j in zip(row_ind, col_ind, strict=True):
            if i < n_req and j < n_veh and cost[i, j] < _UNREACHABLE_COST:
                assignments.append((idle_vehicles[j].vehicle_id, waiting_requests[i].request_id))
        return assignments
