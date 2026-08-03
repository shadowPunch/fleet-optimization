"""Backward-induction value function for B2's value-corrected assignment cost.

Computes C(zone, time_bin): the expected discounted future *wasted idle time*
("cost-to-go") for a single vehicle idling at `zone` at the start of
`time_bin`, under a mean-field approximation — each vehicle acts as if it
alone serves whatever demand the fitted arrival model implies. This ignores
competition between vehicles (a full joint multi-vehicle MDP is intractable
at any realistic fleet size); it's the same simplification production
dispatch value functions in this literature typically make.

Lower C is better: it means a vehicle positioned there will, in expectation,
spend less future time sitting unmatched. `policies.value_corrected_hungarian`
adds C at a candidate destination to the pickup-time cost — but never C at
the vehicle's *current* position, because that term is constant across every
candidate in a given assignment column and is therefore provably irrelevant
to which row the Hungarian solver prefers (subtracting the same value from
every entry in a column never changes that column's preference order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch_eval.models import NHPPArrivalModel, ODModel, TravelTimeModel


@dataclass
class ValueFunction:
    """C[(zone, time_bin)], time_bin in [0, n_bins]. n_bins itself is the
    terminal boundary (always 0) added by `compute_value_function`."""

    values: dict[tuple[str, int], float]
    bin_minutes: int
    n_bins: int

    def cost_to_go(self, zone: str, time_bin: int) -> float:
        return self.values.get((zone, min(time_bin, self.n_bins)), 0.0)


def compute_value_function(
    zones: list[str],
    arrival_model: NHPPArrivalModel,
    od_model: ODModel,
    travel_time_model: TravelTimeModel,
    day_type: str,
    horizon_seconds: float,
    bin_minutes: int = 15,
    od_bin_minutes: float = 60.0,
    discount: float = 0.99,
) -> ValueFunction:
    """Backward-induction fit of C(zone, time_bin) over the fitted models.

    In each bin, a request departing `zone` arrives with probability
    ``1 - exp(-rate * bin_minutes)`` (Poisson). If so, the vehicle serves it
    at ~zero pickup cost (it's already there), travels to a destination drawn
    from the OD model, and continues from there `k >= 1` bins later — this
    branch adds no cost of its own, since being productively matched is the
    point. If not, the vehicle sits idle for the bin, which *does* cost:
    `bin_seconds` of wasted capacity gets added before continuing (discounted)
    to the next bin. Without that idle cost, a zone that's never served would
    look identical to one that's always served (nothing anywhere would ever
    push a value above zero) — the idle penalty is what makes "busy" and
    "quiet" zones separable at all. `discount` (the plan's "value discount")
    is a per-bin decay applied regardless of branch, and is one of B2's
    tunable parameters, not hand-picked.
    """
    n_bins = int(horizon_seconds / 60.0 / bin_minutes)
    values: dict[tuple[str, int], float] = {(z, n_bins): 0.0 for z in zones}
    bin_seconds = bin_minutes * 60.0

    for t in range(n_bins - 1, -1, -1):
        hour = int((t * bin_minutes // 60) % 24)
        od_time_bin = int((t * bin_minutes) // od_bin_minutes)
        for z in zones:
            rate = arrival_model.rate_per_minute(z, day_type, t * bin_minutes)
            p_serve = 1.0 - np.exp(-rate * bin_minutes)
            idle_next = values[(z, min(t + 1, n_bins))]

            dest_probs = od_model.dest_probs.get((z, od_time_bin), od_model.fallback_probs)
            if dest_probs:
                serve_continuation = 0.0
                for dest, prob in dest_probs.items():
                    travel_seconds = travel_time_model.expected(z, dest, hour)
                    k = max(1, round(travel_seconds / bin_seconds))
                    serve_continuation += prob * (discount**k) * values[(dest, min(t + k, n_bins))]
            else:
                serve_continuation = discount * idle_next

            idle_branch = bin_seconds + discount * idle_next
            values[(z, t)] = p_serve * serve_continuation + (1.0 - p_serve) * idle_branch

    return ValueFunction(values=values, bin_minutes=bin_minutes, n_bins=n_bins)
