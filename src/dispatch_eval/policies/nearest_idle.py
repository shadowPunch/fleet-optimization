"""B0 — nearest idle vehicle (first dispatch).

The plan's simplest baseline: for each waiting request, oldest first, assign
whichever idle vehicle has the shortest expected travel time to the request's
origin zone. Greedy, no batching, no lookahead — exactly what generated the
observed wait-time distribution under the "first dispatch" assumption used to
calibrate fleet size (see P1 in the project plan).
"""

from __future__ import annotations

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.simulator.entities import Request, Vehicle


class NearestIdlePolicy:
    def dispatch(
        self,
        waiting_requests: list[Request],
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
    ) -> list[tuple[str, str]]:
        available = list(idle_vehicles)
        assignments: list[tuple[str, str]] = []

        for request in sorted(waiting_requests, key=lambda r: r.request_time):
            if not available:
                break
            best_vehicle = min(
                available,
                key=lambda v: travel_time_model.expected(v.zone, request.origin_zone, current_hour),
            )
            assignments.append((best_vehicle.vehicle_id, request.request_id))
            available.remove(best_vehicle)

        return assignments
