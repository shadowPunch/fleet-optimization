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

RELAXATION, not approximation — this distinction is the point, read it
before trusting a headline number derived from this module. A request's
true pickup time (and therefore exactly when the serving vehicle becomes
free again) is exactly the thing being solved for, which would make edge
costs solution-dependent in a naive version of this graph. Earlier
versions of this module sidestepped that by pinning each request's
"vehicle becomes free" event to its *patience deadline* — which
over-constrains capacity (ties a vehicle up longer than any real schedule
would) and can therefore make the "bound" tighter than true optimal in
either direction: not a valid bound at all, just an approximation that
happened to usually undershoot. Confirmed in practice, not hypothetically:
wired into `ranking_flip.run_ranking_flip_experiment` via
`compute_clairvoyant=True` and run against the tuned B0-B4 ladder
(`analysis/clairvoyant_gap_closed_run.py`, `mean_patience_seconds=300.0`),
every policy from B1 up beat that "upper" bound by 5-10% — proof the old
formulation wasn't a real bound, not just a theoretical worry.

**The fix**: pin each request's exit instead to its *earliest feasible*
release — earliest possible pickup time (`request_time`, rounded to the
grid) plus trip duration — regardless of which pickup time the flow
actually selects for the wait-cost term. Since every achievable pickup
time is `>= request_time`, this exit time is always `<=` the schedule's
true release time, for any pickup the flow could pick: the graph only
ever frees a vehicle *too early*, never too late. A relaxed capacity
constraint can only let the solver do as well as or better than the true
optimum — that is what makes this a genuine upper bound, not just a
plausible-sounding schedule. The cost is exactly the mirror image of the
old formulation's: this version can over-serve relative to true
simultaneous capacity in a tie (two requests wanting the same vehicle at
the same place and time can both come out "served" —
`test_relaxation_can_over_serve_a_true_simultaneous_capacity_tie` in
`tests/test_clairvoyant.py` documents this directly), which is the
expected, intended behavior of a relaxation, not a bug: erring toward
"too generous" is exactly the direction a valid upper bound is supposed
to err in.

**A second, separate bug found while re-validating this bound after the
relaxation fix — and the dominant one**: fixing the exit-pinning barely
moved the "policies beat the bound" numbers (still ~5-10% at
`mean_patience_seconds=300.0`). The real cause turned out to be
`bin_minutes` itself. A request is dropped before the solver ever sees it
whenever `ceil(request_time/bin_seconds) > floor(abandon_at/bin_seconds)`
— rounding, not real infeasibility — and at the default `bin_minutes=15.0`
against this project's typical `mean_patience_seconds=300.0` (5 minutes,
a third of one bin), **851 of 1261 requests (67.5%) in one representative
run were excluded this way**, none of them actually unservable. `bin_minutes`
must be small relative to the abandonment hazard's timescale, not just
"small enough to be tractable" — the two constraints trade off directly
(reposition-edge count scales as `zones^2 * horizon_seconds/bin_seconds`,
so shrinking bins to fix this gets expensive fast for many zones) and
picking one without checking the other silently produces a wrong answer,
not a slow one. `ClairvoyantResult.excluded_by_discretization` /
`.fraction_excluded_by_discretization` make this visible directly instead
of leaving it to be inferred from an implausible headline number — check
it before trusting any `fraction_served` or gap-closed report this module
produces, and shrink `bin_minutes` if it's not near zero.

**Both fixes together, re-measured for real**
(`analysis/clairvoyant_gap_closed_run.py`, `clairvoyant_bin_minutes=1.0`
instead of the 15.0 default): exclusion drops from 67.5% to 9.6%, and the
gap-closed numbers move from a suspicious 1.05-1.10 (every policy beating
the bound) to a believable 0.96-1.01 (every policy closing 96-101% of the
gap, B2 the only one fractionally over — plausibly residual discretization
noise, not evidence the bound is still broken). This is what a genuine
upper bound with a nearly-converged discretization looks like; the old
1.05-1.10 numbers were never a real finding about these policies, they
were this module's own bugs.

This solver's input is a fixed, already-realized list of requests — how
that list gets generated (and whether it's the *same* realized trace across
different policies, which the "gap closed" comparison requires) is entirely
the caller's responsibility. See TECHNICAL_DOCUMENTATION.md's note on the CRN fix
before wiring this into a cross-policy comparison.
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
    excluded_by_discretization: int = 0
    """Requests dropped before the solver ever saw them, purely because
    `bin_minutes` was too coarse relative to the request's own patience
    window (`ceil(request_time/bin) > floor(abandon_at/bin)`) — not
    because of any real capacity limit. Large relative to `total_requests`
    means `bin_minutes` needs to shrink, or `fraction_served`/`wait_times`
    understate what's truly achievable and this isn't a valid upper bound
    in practice. See this module's docstring."""

    @property
    def wait_times(self) -> np.ndarray:
        return np.array(list(self.wait_seconds.values()))

    @property
    def fraction_served(self) -> float:
        return len(self.served_request_ids) / self.total_requests if self.total_requests else 0.0

    @property
    def fraction_excluded_by_discretization(self) -> float:
        return self.excluded_by_discretization / self.total_requests if self.total_requests else 0.0


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
        return ClairvoyantResult(set(), {}, len(requests), excluded_by_discretization=0)

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
    excluded_by_discretization = 0
    for req in requests:
        t_req = int(np.ceil(req.request_time / bin_seconds))
        t_abandon = min(int(np.floor(req.abandon_at / bin_seconds)), n_bins)
        if t_abandon < t_req or t_req > n_bins:
            excluded_by_discretization += 1
            continue  # unservable in this discretization: patience shorter than one bin
        request_windows[req.request_id] = (t_req, t_abandon)

        hour = int((t_req * bin_seconds // 3600) % 24)
        trip_seconds = travel_time_model.expected(req.origin_zone, req.dest_zone, hour)
        k = max(1, round(trip_seconds / bin_seconds))
        exit_t = min(t_req + k, n_bins)

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
                # Boarding is exogenous: it adds to the rider's wait but can't
                # change which schedule is optimal, so it's applied here only.
                wait_seconds_map[req.request_id] = (
                    max(0.0, t * bin_seconds - req.request_time) + req.boarding_seconds
                )
                break
        served_ids.add(req.request_id)

    return ClairvoyantResult(
        served_ids, wait_seconds_map, len(requests), excluded_by_discretization=excluded_by_discretization
    )
