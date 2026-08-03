"""Event-driven discrete-event simulator (heapq-based).

Deliberately minimal per the P1 spec in the project plan: ward/zone-level,
single operator, no road graph. All simulation time is seconds since the
start of the run.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

import numpy as np

from dispatch_eval.models import (
    AbandonmentModel,
    NHPPArrivalModel,
    ODModel,
    TravelTimeModel,
)
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.simulator.entities import Request, RequestStatus, Vehicle, VehicleStatus
from dispatch_eval.simulator.events import Event, EventType


@dataclass
class SimulationResult:
    completed_requests: list[Request]
    abandoned_requests: list[Request]
    total_requests: int
    vehicle_hours: float

    @property
    def wait_times(self) -> np.ndarray:
        return np.array([r.wait_time for r in self.completed_requests])

    @property
    def mean_wait_seconds(self) -> float:
        """Tuning/comparison objective. inf when nothing completed, so a
        policy that serves nobody never looks good under minimization."""
        wt = self.wait_times
        return float(wt.mean()) if wt.size else float("inf")

    @property
    def fraction_served(self) -> float:
        return len(self.completed_requests) / self.total_requests if self.total_requests else 0.0


class SimulationEngine:
    def __init__(
        self,
        vehicles: list[Vehicle],
        arrival_model: NHPPArrivalModel,
        od_model: ODModel,
        travel_time_model: TravelTimeModel,
        abandonment_model: AbandonmentModel,
        policy: DispatchPolicy,
        zones: list[str],
        day_type: str,
        horizon_seconds: float,
        rng: np.random.Generator,
        dispatch_interval_seconds: float = 5.0,
        od_bin_minutes: float = 60.0,
    ) -> None:
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.arrival_model = arrival_model
        self.od_model = od_model
        self.travel_time_model = travel_time_model
        self.abandonment_model = abandonment_model
        self.policy = policy
        self.zones = zones
        self.day_type = day_type
        self.horizon_seconds = horizon_seconds
        self.rng = rng
        self.dispatch_interval_seconds = dispatch_interval_seconds
        self.od_bin_minutes = od_bin_minutes

        self._queue: list[Event] = []
        self._seq = count()
        self._request_counter = count()
        self.requests: dict[str, Request] = {}
        self._completed: list[Request] = []
        self._abandoned: list[Request] = []

    def _schedule(self, time: float, event_type: EventType, **payload: object) -> None:
        heapq.heappush(
            self._queue,
            Event(time=time, seq=next(self._seq), event_type=event_type, payload=payload),
        )

    def _hour_of(self, time: float) -> int:
        return int((time // 3600.0) % 24)

    def reposition_vehicle(self, vehicle_id: str, target_zone: str, current_time: float) -> None:
        """Send an idle vehicle to reposition. Unused by B0; exists for B3+ policies."""
        vehicle = self.vehicles[vehicle_id]
        if vehicle.status != VehicleStatus.IDLE:
            return
        hour = self._hour_of(current_time)
        travel = self.travel_time_model.sample(vehicle.zone, target_zone, hour, self.rng)
        arrival_time = current_time + travel
        vehicle.status = VehicleStatus.REPOSITIONING
        vehicle.available_at = arrival_time
        self._schedule(
            arrival_time, EventType.REPOSITION_ARRIVAL, vehicle_id=vehicle_id, zone=target_zone
        )

    def _seed_arrivals(self) -> None:
        horizon_minutes = self.horizon_seconds / 60.0
        for zone in self.zones:
            arrival_minutes = self.arrival_model.generate_arrival_minutes(
                zone, self.day_type, 0.0, horizon_minutes, self.rng
            )
            for minute in arrival_minutes:
                self._schedule(minute * 60.0, EventType.REQUEST_ARRIVAL, zone=zone)

    def _seed_dispatch_ticks(self) -> None:
        t = 0.0
        while t <= self.horizon_seconds:
            self._schedule(t, EventType.DISPATCH_TICK)
            t += self.dispatch_interval_seconds

    def run(self) -> SimulationResult:
        self._seed_arrivals()
        self._seed_dispatch_ticks()

        while self._queue:
            event = heapq.heappop(self._queue)
            if event.time > self.horizon_seconds:
                break
            handler = getattr(self, f"_handle_{event.event_type.value}")
            handler(event)

        vehicle_hours = len(self.vehicles) * self.horizon_seconds / 3600.0
        return SimulationResult(
            completed_requests=self._completed,
            abandoned_requests=self._abandoned,
            total_requests=len(self.requests),
            vehicle_hours=vehicle_hours,
        )

    def _handle_request_arrival(self, event: Event) -> None:
        zone = event.payload["zone"]
        time_bin = int((event.time / 60.0) // self.od_bin_minutes)
        dest_zone = self.od_model.sample_destination(zone, time_bin, self.rng)
        request_id = f"req-{next(self._request_counter)}"
        patience = self.abandonment_model.sample_patience(self.rng)
        request = Request(
            request_id=request_id,
            origin_zone=zone,
            dest_zone=dest_zone,
            request_time=event.time,
            abandon_at=event.time + patience,
        )
        self.requests[request_id] = request
        self._schedule(request.abandon_at, EventType.ABANDONMENT, request_id=request_id)

    def _handle_dispatch_tick(self, event: Event) -> None:
        waiting = [r for r in self.requests.values() if r.status == RequestStatus.WAITING]
        idle = [v for v in self.vehicles.values() if v.status == VehicleStatus.IDLE]
        hour = self._hour_of(event.time)

        if waiting and idle:
            assignments = self.policy.dispatch(
                waiting, idle, event.time, hour, self.travel_time_model
            )
            for vehicle_id, request_id in assignments:
                self._assign(vehicle_id, request_id, event.time)

        # Repositioning is optional: only policies that define `reposition`
        # (e.g. B3, B4) get this call. Recomputed after dispatch, since
        # vehicles just assigned above are no longer idle and shouldn't be
        # reconsidered. `self.rng` is passed through rather than letting a
        # policy own its own stream — see RepositioningPolicy's docstring.
        reposition_fn = getattr(self.policy, "reposition", None)
        if reposition_fn is not None:
            still_idle = [v for v in self.vehicles.values() if v.status == VehicleStatus.IDLE]
            if still_idle:
                for vehicle_id, target_zone in reposition_fn(
                    still_idle, event.time, hour, self.travel_time_model, self.zones, self.rng
                ):
                    self.reposition_vehicle(vehicle_id, target_zone, event.time)

    def _assign(self, vehicle_id: str, request_id: str, current_time: float) -> None:
        vehicle = self.vehicles[vehicle_id]
        request = self.requests[request_id]
        if vehicle.status != VehicleStatus.IDLE or request.status != RequestStatus.WAITING:
            return  # stale pairing from a policy snapshot that's since moved on
        hour = self._hour_of(current_time)
        pickup_travel = self.travel_time_model.sample(
            vehicle.zone, request.origin_zone, hour, self.rng
        )
        pickup_time = current_time + pickup_travel
        vehicle.status = VehicleStatus.EN_ROUTE_PICKUP
        vehicle.assigned_request_id = request_id
        vehicle.available_at = pickup_time
        request.status = RequestStatus.DISPATCHED
        request.assigned_vehicle_id = vehicle_id
        self._schedule(pickup_time, EventType.PICKUP, vehicle_id=vehicle_id, request_id=request_id)

    def _handle_pickup(self, event: Event) -> None:
        vehicle_id = event.payload["vehicle_id"]
        request_id = event.payload["request_id"]
        vehicle = self.vehicles[vehicle_id]
        request = self.requests[request_id]
        if request.status != RequestStatus.DISPATCHED:
            return
        request.status = RequestStatus.IN_TRIP
        request.pickup_time = event.time
        hour = self._hour_of(event.time)
        trip_duration = self.travel_time_model.sample(
            request.origin_zone, request.dest_zone, hour, self.rng
        )
        dropoff_time = event.time + trip_duration
        vehicle.status = VehicleStatus.OCCUPIED
        vehicle.available_at = dropoff_time
        self._schedule(
            dropoff_time, EventType.DROPOFF, vehicle_id=vehicle_id, request_id=request_id
        )

    def _handle_dropoff(self, event: Event) -> None:
        vehicle_id = event.payload["vehicle_id"]
        request_id = event.payload["request_id"]
        vehicle = self.vehicles[vehicle_id]
        request = self.requests[request_id]
        request.status = RequestStatus.COMPLETED
        request.dropoff_time = event.time
        vehicle.zone = request.dest_zone
        vehicle.status = VehicleStatus.IDLE
        vehicle.assigned_request_id = None
        self._completed.append(request)

    def _handle_abandonment(self, event: Event) -> None:
        request = self.requests[event.payload["request_id"]]
        if request.status != RequestStatus.WAITING:
            return  # already dispatched before patience ran out
        request.status = RequestStatus.ABANDONED
        self._abandoned.append(request)

    def _handle_reposition_arrival(self, event: Event) -> None:
        vehicle = self.vehicles[event.payload["vehicle_id"]]
        vehicle.status = VehicleStatus.IDLE
        vehicle.zone = event.payload["zone"]
