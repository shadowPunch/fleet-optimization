"""B5 — clairvoyant offline upper bound (deliberately not a `DispatchPolicy`).

Given the full realized trace of requests for a horizon (as if a dispatcher
knew, in advance, every request that would ever arrive, its destination, and
its patience deadline) and the vehicles' starting positions, compute the
best achievable offline schedule: which vehicle serves which requests, in
what order, maximizing the number served and, subject to that, minimizing
total wait. There's no per-tick decision function here, unlike B0-B4 — it
needs the whole future at once, so it lives outside `policies/` entirely.

Why this needs more than a single bipartite match (unlike B1-B4): one
vehicle can serve many requests in sequence over the horizon, and a
feasible sequence of pickups is a *flow*, not a one-shot assignment — a
request can only be someone's "next pickup" if it was itself served by
someone first, and a plain bipartite match has no way to enforce that
precedence. Flow conservation on a time-expanded graph does.

Time-expanded graph: one node per (zone, time_bin). A "wait" edge connects
(zone, t) to (zone, t+1) at zero cost — a vehicle can always just sit still
— and a free "reposition" edge connects (origin, t) to (dest, t+k) for every
other zone, k bins later (however long that trip takes), so an idle vehicle
can always relocate for free, matching what B3/B4 can already do online; without
this, B5 would be a *looser* bound than policies that reposition, which
defeats the purpose. Each vehicle contributes one unit of supply at (its
start zone, 0); all supply drains to one sink at the horizon's end. Each
request gets its own pair of nodes, `in` and `out`, joined by a capacity-1
edge carrying a large per-request bonus (see `_SERVE_BONUS_MULTIPLIER`) — big
enough that the min-cost solution always prefers serving one more request
over saving any amount of wait time, so "maximize requests served" becomes
the primary objective and "minimize wait" the tie-break, without needing two
separate optimization passes.

APPROXIMATION — read this before trusting a headline number derived from it:
a request's true pickup time (and therefore exactly when the serving
vehicle becomes free again) is exactly the thing being solved for, which
would make edge costs solution-dependent in a naive version of this graph.
This implementation sidesteps that by pinning each request's "vehicle
becomes free" event to its patience deadline (the *latest* moment it could
still be served) rather than to whichever pickup time the solution actually
picks. Every candidate pickup time within [request_time, abandon_at] is
still correctly *costed* — the wait-time term reflects exactly when the
vehicle actually reaches the request — but the vehicle's re-entry into the
graph afterward is pessimistic. This turns out to cost more than "slightly":
serving a request ties up the vehicle until *that request's own* patience
deadline, no matter how quickly it was actually picked up — so a request
with a long patience window effectively parks its vehicle for that whole
window before anything downstream can use it again. Confirmed empirically
while writing this module's tests: a chain of two individually-easy requests
failed to both get served whenever the first one's patience window was long
relative to the horizon, purely from this pinning, not from any real
capacity limit (see `test_maximizing_requests_served_beats_minimizing_wait`
in `tests/test_clairvoyant.py`, which keeps the first request's patience
short specifically to keep this cost small). Net effect: this is a valid,
honestly-conservative achievable schedule, not a razor-tight bound, and it
gets *more* conservative as patience windows widen relative to trip
durations — worth keeping in mind when picking abandonment-hazard
specifications to run this against in P3.

**Confirmed in production use, not just in a unit test**: wired into
`ranking_flip.run_ranking_flip_experiment` via `compute_clairvoyant=True`
and run against the tuned B0-B4 ladder
(`analysis/clairvoyant_gap_closed_run.py`, `mean_patience_seconds=300.0`),
every policy from B1 up — not just an edge case — beat this "upper" bound
by 5-10% (`fraction_of_gap_closed` of 1.05-1.10, where 1.0 would mean
exactly matching it). At this patience setting the pinning approximation
is loose enough to be a real bound in name only; a report built on this
module should either use a much shorter patience window or present the
gap-closed numbers as "relative to a conservative reference schedule," not
as "the theoretical maximum."

This solver's input is a fixed, already-realized list of requests — how
that list gets generated (and whether it's the *same* realized trace across
different policies, which the "gap closed" comparison requires) is entirely
the caller's responsibility. See the README's note on the engine's current
random-number handling before wiring this into a cross-policy comparison.
`ranking_flip.run_ranking_flip_experiment(..., compute_clairvoyant=True)`
does exactly this correctly (same-seed scenario reproduction, see its own
docstring) — that's the intended way to use this module for a real
gap-closed report, not calling it standalone per policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dispatch_eval.models import TravelTimeModel
from dispatch_eval.simulator.entities import Request, Vehicle

_SERVE_BONUS_MULTIPLIER = 10  # per-request service bonus, as a multiple of horizon_seconds
_SINK = "__sink__"


@dataclass
class ClairvoyantResult:
    served_request_ids: set[str]
    wait_seconds: dict[str, float]
    total_requests: int

    @property
    def wait_times(self) -> np.ndarray:
        return np.array(list(self.wait_seconds.values()))

    @property
    def fraction_served(self) -> float:
        return len(self.served_request_ids) / self.total_requests if self.total_requests else 0.0


def _grid_node(zone: str, t: int, n_bins: int) -> str:
    return f"{zone}@{min(t, n_bins)}"


def solve_clairvoyant_schedule(
    requests: list[Request],
    vehicles: list[Vehicle],
    travel_time_model: TravelTimeModel,
    horizon_seconds: float,
    bin_minutes: float = 15.0,
) -> ClairvoyantResult:
    if not vehicles or not requests:
        return ClairvoyantResult(set(), {}, len(requests))

    bin_seconds = bin_minutes * 60.0
    n_bins = int(np.ceil(horizon_seconds / bin_seconds))
    bonus = _SERVE_BONUS_MULTIPLIER * horizon_seconds

    graph = nx.DiGraph()
    zones = sorted(
        {v.zone for v in vehicles}
        | {r.origin_zone for r in requests}
        | {r.dest_zone for r in requests}
    )
    total_fleet = len(vehicles)

    # Wait edges (same zone, one bin later) and empty-repositioning edges
    # (different zone, `k` bins later) — both free. Without repositioning, a
    # vehicle could only ever change zones by serving a request headed
    # there, which would make this a *looser* bound than online policies
    # that can reposition (B3, B4) — not a valid upper bound on them.
    for zone in zones:
        for t in range(n_bins):
            graph.add_edge(
                _grid_node(zone, t, n_bins),
                _grid_node(zone, t + 1, n_bins),
                capacity=total_fleet,
                weight=0,
            )
        graph.add_edge(_grid_node(zone, n_bins, n_bins), _SINK, capacity=total_fleet, weight=0)

    for origin in zones:
        for dest in zones:
            if origin == dest:
                continue
            for t in range(n_bins):
                hour = int((t * bin_seconds // 3600) % 24)
                travel_seconds = travel_time_model.expected(origin, dest, hour)
                k = max(1, round(travel_seconds / bin_seconds))
                graph.add_edge(
                    _grid_node(origin, t, n_bins),
                    _grid_node(dest, t + k, n_bins),
                    capacity=total_fleet,
                    weight=0,
                )

    supply_by_zone: dict[str, int] = {}
    for vehicle in vehicles:
        supply_by_zone[vehicle.zone] = supply_by_zone.get(vehicle.zone, 0) + 1
    for zone, count in supply_by_zone.items():
        graph.add_node(_grid_node(zone, 0, n_bins), demand=-count)
    graph.add_node(_SINK, demand=total_fleet)

    request_windows: dict[str, tuple[int, int]] = {}
    for req in requests:
        t_req = int(np.ceil(req.request_time / bin_seconds))
        t_abandon = min(int(np.floor(req.abandon_at / bin_seconds)), n_bins)
        if t_abandon < t_req or t_req > n_bins:
            continue  # unservable in this discretization: patience shorter than one bin
        request_windows[req.request_id] = (t_req, t_abandon)

        hour = int((t_abandon * bin_seconds // 3600) % 24)
        trip_seconds = travel_time_model.expected(req.origin_zone, req.dest_zone, hour)
        k = max(1, round(trip_seconds / bin_seconds))
        exit_t = min(t_abandon + k, n_bins)

        in_node, out_node = f"req_in::{req.request_id}", f"req_out::{req.request_id}"
        graph.add_edge(in_node, out_node, capacity=1, weight=round(-bonus))
        graph.add_edge(out_node, _grid_node(req.dest_zone, exit_t, n_bins), capacity=1, weight=0)

        for t in range(t_req, t_abandon + 1):
            wait_seconds = max(0.0, t * bin_seconds - req.request_time)
            graph.add_edge(
                _grid_node(req.origin_zone, t, n_bins),
                in_node,
                capacity=1,
                weight=round(wait_seconds),
            )

    flow_dict = nx.min_cost_flow(graph)

    served_ids: set[str] = set()
    wait_seconds_map: dict[str, float] = {}
    for req in requests:
        window = request_windows.get(req.request_id)
        if window is None:
            continue
        in_node, out_node = f"req_in::{req.request_id}", f"req_out::{req.request_id}"
        if flow_dict.get(in_node, {}).get(out_node, 0) <= 0:
            continue

        t_req, t_abandon = window
        for t in range(t_req, t_abandon + 1):
            origin_node = _grid_node(req.origin_zone, t, n_bins)
            if flow_dict.get(origin_node, {}).get(in_node, 0) > 0:
                wait_seconds_map[req.request_id] = max(0.0, t * bin_seconds - req.request_time)
                break
        served_ids.add(req.request_id)

    return ClairvoyantResult(served_ids, wait_seconds_map, len(requests))
