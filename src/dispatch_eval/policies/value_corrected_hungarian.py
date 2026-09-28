"""B2 — batched Hungarian assignment with a value-function correction.

Same batching/radius mechanics as B1 (`policies.batched_hungarian`), but the
cost matrix adds the future cost-to-go at the vehicle's *post-trip* position
(see `calibration.value_function`): a vehicle that will end up stranded in a
low-demand zone is charged for that, even if this particular pickup is
short. The vehicle's own cost-to-go at its *current* position would also
belong in this formula in principle, but it's the same value added to every
candidate in a vehicle's column and so is provably irrelevant to the
assignment — omitted rather than computed for nothing (see
`calibration.value_function`'s docstring).

`value_weight` (the plan's "value discount" tunable, alongside the value
function's own internal discount) controls how much the future term matters
relative to the immediate pickup time; `value_weight=0` recovers B1 exactly.
"""

from __future__ import annotations

from dispatch_eval.calibration.value_function import ValueFunction
from dispatch_eval.models import TravelTimeModel
from dispatch_eval.policies.zone_index import solve_batched_assignment
from dispatch_eval.simulator.entities import Request, Vehicle


class ValueCorrectedHungarianPolicy:
    def __init__(
        self,
        value_function: ValueFunction,
        value_weight: float = 1.0,
        matching_radius_seconds: float = float("inf"),
    ):
        self.value_function = value_function
        self.value_weight = value_weight
        self.matching_radius_seconds = matching_radius_seconds

    def dispatch(
        self,
        waiting_requests: list[Request],
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
    ) -> list[tuple[str, str]]:
        bin_seconds = self.value_function.bin_minutes * 60.0

        def pair_cost(vehicle_zone: str, request: Request) -> float | None:
            pickup_travel = travel_time_model.expected(
                vehicle_zone, request.origin_zone, current_hour
            )
            if pickup_travel > self.matching_radius_seconds:
                return None
            trip_travel = travel_time_model.expected(
                request.origin_zone, request.dest_zone, current_hour
            )
            dropoff_bin = int((current_time + pickup_travel + trip_travel) / bin_seconds)
            future_cost = self.value_function.cost_to_go(request.dest_zone, dropoff_bin)
            return pickup_travel + self.value_weight * future_cost

        return solve_batched_assignment(waiting_requests, idle_vehicles, pair_cost)
