"""Event queue primitives for the heapq-based DES loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    REQUEST_ARRIVAL = "request_arrival"
    DISPATCH_TICK = "dispatch_tick"
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    REPOSITION_ARRIVAL = "reposition_arrival"
    ABANDONMENT = "abandonment"


@dataclass(order=True)
class Event:
    """Ordered by (time, seq) so heapq gives a deterministic FIFO tie-break.

    `seq` is assigned by the engine at schedule time, not here, so that a
    single monotonic counter is shared across everything pushed onto one
    engine's queue without relying on module-level global state.
    """

    time: float
    seq: int
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
