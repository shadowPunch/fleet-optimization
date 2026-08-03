"""Simulator state: vehicles and requests, per the P1 spec in the project plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VehicleStatus(Enum):
    IDLE = "idle"
    EN_ROUTE_PICKUP = "en_route_pickup"
    OCCUPIED = "occupied"
    REPOSITIONING = "repositioning"


@dataclass
class Vehicle:
    vehicle_id: str
    zone: str
    status: VehicleStatus = VehicleStatus.IDLE
    available_at: float = 0.0
    assigned_request_id: str | None = None


class RequestStatus(Enum):
    WAITING = "waiting"
    DISPATCHED = "dispatched"
    IN_TRIP = "in_trip"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Request:
    request_id: str
    origin_zone: str
    dest_zone: str
    request_time: float
    abandon_at: float
    status: RequestStatus = RequestStatus.WAITING
    assigned_vehicle_id: str | None = None
    pickup_time: float | None = None
    dropoff_time: float | None = None

    @property
    def wait_time(self) -> float | None:
        if self.pickup_time is None:
            return None
        return self.pickup_time - self.request_time
