"""Dispatch policy interface.

A policy is called once per dispatch tick with the current set of waiting
requests and idle vehicles, and returns the assignments it wants to make. The
engine owns everything else (movement, state transitions, event scheduling) —
a policy only decides *who goes to whom*, so that B0-B5 in the project plan's
baseline ladder are all the same shape and can be swapped without touching
the engine.
"""

from __future__ import annotations

from typing import Protocol

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.simulator.entities import Request, Vehicle


class DispatchPolicy(Protocol):
    def dispatch(
        self,
        waiting_requests: list[Request],
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
    ) -> list[tuple[str, str]]:
        """Return a list of (vehicle_id, request_id) assignments to make now."""
        ...
