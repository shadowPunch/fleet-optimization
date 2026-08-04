"""C4 — Compute parity: wall-clock decision latency vs. problem size.

Production dispatch runs on a fixed real-time cycle (a couple of seconds).
A policy that can't finish its decision inside that cycle isn't actually
deployable at that fleet scale, no matter how good its assignments are in
principle — this module measures that directly instead of assuming every
policy in the ladder is free to compute.

`measure_decision_latency` / `latency_frontier`: wall-clock time for one
policy's `dispatch()` call (plus `reposition()`, if it has one — B3/B4),
as a function of the number of open requests and idle vehicles. Builds
fresh synthetic `Request`/`Vehicle` objects at each problem size (never
reuses instances across repeats) and reports the median over several
repeats, which damps OS/GC jitter without averaging it away the way a mean
would. `feasible_within_budget` turns a latency table into pass/fail
against a chosen real-time cycle length.

**Scope note, found while building this, not assumed going in:** the
plan's C4 description also calls for "degrade the ones that cannot
[complete a decision in budget] (truncate the candidate graph, coarsen the
radius) and re-measure quality." B1/B2 already expose a
`matching_radius_seconds` knob that looks like exactly this — but reading
`policies/batched_hungarian.py` shows it's a *cost-masking* mechanism
(out-of-radius pairs get an enormous cost, not removal), not a candidate-
graph-*size* reduction: the cost matrix `scipy.optimize.linear_sum_assignment`
solves is always `max(n_requests, n_vehicles)` square regardless of the
radius, so sweeping it changes assignment *quality*, not wall-clock
*latency*. Confirmed empirically, not just by reading the code — see
`tests/test_compute_parity.py::test_matching_radius_does_not_change_latency`.
A real "truncate the candidate graph" lever (actually shrinking the matrix
solved, e.g. only including each request's k nearest vehicles) doesn't
exist in this codebase yet — building it is a prerequisite for the
"degrade + re-measure quality" half of C4, and isn't attempted here: this
module answers the measurement half honestly, on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.simulator.entities import Request, Vehicle


def _synthetic_requests(n: int, zones: list[str], current_time: float) -> list[Request]:
    return [
        Request(
            request_id=f"req-{i}",
            origin_zone=zones[i % len(zones)],
            dest_zone=zones[(i + 1) % len(zones)],
            request_time=current_time,
            abandon_at=current_time + 600.0,
        )
        for i in range(n)
    ]


def _synthetic_vehicles(n: int, zones: list[str]) -> list[Vehicle]:
    return [Vehicle(vehicle_id=f"veh-{i}", zone=zones[i % len(zones)]) for i in range(n)]


@dataclass
class LatencyMeasurement:
    n_requests: int
    n_idle_vehicles: int
    dispatch_seconds: float
    reposition_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.dispatch_seconds + self.reposition_seconds


def measure_decision_latency(
    policy: DispatchPolicy,
    n_requests: int,
    n_idle_vehicles: int,
    zones: list[str],
    travel_time_model: TravelTimeModel,
    rng: np.random.Generator,
    current_time: float = 0.0,
    current_hour: int = 12,
    n_repeats: int = 5,
) -> LatencyMeasurement:
    """Median wall-clock time for one `dispatch()` call (+ `reposition()`,
    if `policy` has one) at a synthetic problem of this size, over
    `n_repeats` freshly-built problem instances.
    """
    dispatch_times = np.empty(n_repeats)
    has_reposition = hasattr(policy, "reposition")
    reposition_times = np.empty(n_repeats) if has_reposition else None

    for i in range(n_repeats):
        requests = _synthetic_requests(n_requests, zones, current_time)
        vehicles = _synthetic_vehicles(n_idle_vehicles, zones)

        start = time.perf_counter()
        policy.dispatch(requests, vehicles, current_time, current_hour, travel_time_model)
        dispatch_times[i] = time.perf_counter() - start

        if has_reposition:
            vehicles_for_reposition = _synthetic_vehicles(n_idle_vehicles, zones)
            start = time.perf_counter()
            policy.reposition(
                vehicles_for_reposition, current_time, current_hour, travel_time_model, zones, rng
            )
            reposition_times[i] = time.perf_counter() - start

    return LatencyMeasurement(
        n_requests=n_requests,
        n_idle_vehicles=n_idle_vehicles,
        dispatch_seconds=float(np.median(dispatch_times)),
        reposition_seconds=float(np.median(reposition_times)) if has_reposition else 0.0,
    )


def latency_frontier(
    policies: dict[str, DispatchPolicy],
    problem_sizes: list[tuple[int, int]],
    zones: list[str],
    travel_time_model: TravelTimeModel,
    seed: int = 0,
    n_repeats: int = 5,
) -> dict[str, list[LatencyMeasurement]]:
    """`measure_decision_latency` for every policy, at every
    `(n_requests, n_idle_vehicles)` pair in `problem_sizes`, in order."""
    out: dict[str, list[LatencyMeasurement]] = {}
    for name, policy in policies.items():
        rng = np.random.default_rng(seed)
        out[name] = [
            measure_decision_latency(
                policy, n_req, n_veh, zones, travel_time_model, rng, n_repeats=n_repeats
            )
            for n_req, n_veh in problem_sizes
        ]
    return out


def feasible_within_budget(
    measurements: dict[str, list[LatencyMeasurement]], budget_seconds: float
) -> dict[str, list[bool]]:
    """Which of each policy's measured problem sizes fit inside `budget_seconds`."""
    return {
        name: [m.total_seconds <= budget_seconds for m in ms] for name, ms in measurements.items()
    }
