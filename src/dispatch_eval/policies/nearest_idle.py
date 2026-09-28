"""B0 — nearest idle vehicle (first dispatch).

The plan's simplest baseline: for each waiting request, oldest first, assign
whichever idle vehicle has the shortest expected travel time to the request's
origin zone. Greedy, no batching, no lookahead — exactly what generated the
observed wait-time distribution under the "first dispatch" assumption used to
calibrate fleet size (see P1 in the project plan).
"""

from __future__ import annotations

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.policies.zone_index import group_by_zone
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
        # Search over zones, not vehicles (see zone_index). Ties between
        # equally-near zones go to the zone whose next vehicle comes first
        # in `idle_vehicles` — the same vehicle a scan of the full list picks.
        position = {v.vehicle_id: i for i, v in enumerate(idle_vehicles)}
        by_zone = group_by_zone(idle_vehicles)
        assignments: list[tuple[str, str]] = []

        for request in sorted(waiting_requests, key=lambda r: r.request_time):
            if not by_zone:
                break
            best_zone = min(
                by_zone,
                key=lambda z: (
                    travel_time_model.expected(z, request.origin_zone, current_hour),
                    position[by_zone[z][0].vehicle_id],
                ),
            )
            vehicle = by_zone[best_zone].popleft()
            if not by_zone[best_zone]:
                del by_zone[best_zone]
            assignments.append((vehicle.vehicle_id, request.request_id))

        return assignments
