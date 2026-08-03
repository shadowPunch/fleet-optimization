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

import numpy as np

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


class RepositioningPolicy(Protocol):
    """Optional second capability (B3+): move idle vehicles with nothing to
    serve toward better zones. The engine checks for this via `getattr` at
    each dispatch tick — a plain `DispatchPolicy` without a `reposition`
    method (B0-B2) is unaffected and never has this called.

    `rng` is the engine's own random generator, not a policy-owned one —
    B4's sampled lookahead needs randomness, and threading the engine's rng
    through (rather than letting a policy carry its own separate stream)
    keeps every random draw in a simulation run traceable to one seed. That
    matters for P3's common-random-numbers requirement: a policy with a
    hidden, independently-seeded rng of its own would introduce variation
    across bootstrap replications that isn't controlled by the shared seed.
    """

    def reposition(
        self,
        idle_vehicles: list[Vehicle],
        current_time: float,
        current_hour: int,
        travel_time_model: TravelTimeModel,
        zones: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, str]]:
        """Return a list of (vehicle_id, target_zone) repositioning moves."""
        ...
