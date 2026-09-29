"""Fleet-size calibration: latent, matched to the observed wait-time distribution.

Per the project plan: fleet size and the abandonment hazard are not
identified separately from matched-trip data alone. Calibrate fleet size to
match the observed wait-time distribution's median and 90th percentile under
the nearest-idle baseline; treat the abandonment hazard as a separate
structural axis to sweep (see `models.AbandonmentModel`), not something this
routine also tries to fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FleetSizeCalibrationResult:
    fleet_size: int
    loss: float
    candidates: dict[int, float]
    hit_boundary: bool = False
    """True when `fleet_size` is the smallest or largest value in
    `candidate_fleet_sizes` — a plain grid search has no way to tell "the
    true optimum is genuinely at this extreme" from "the true optimum is
    outside the range tried at all," and silently returns the same result
    either way. A `True` here means the candidate range should be widened
    and recalibrated before trusting this fleet size, not treated as
    converged."""


def absolute_moment_loss(simulated_wait: np.ndarray, observed_median: float, observed_p90: float) -> float:
    """Default loss: match median and p90 in absolute seconds.

    Asks the calibration to land on both the right *scale* and the right
    *shape* at once. When a simulator's dispatch mechanism produces a
    wait-time distribution with a fundamentally different shape than
    reality (see `TECHNICAL_DOCUMENTATION.md`: simulated P90/median ratio
    far from real, no fleet size closes it), no candidate can satisfy both
    halves of this loss simultaneously — the search still returns
    *something*, but it's a compromise between two things pulling in
    different directions, not evidence either one was actually matched.
    See `ratio_loss` for the alternative that asks for only one of them.
    """
    if simulated_wait.size == 0:
        return float("inf")
    sim_median = float(np.median(simulated_wait))
    sim_p90 = float(np.percentile(simulated_wait, 90))
    return (sim_median - observed_median) ** 2 + (sim_p90 - observed_p90) ** 2


def ratio_loss(simulated_wait: np.ndarray, observed_median: float, observed_p90: float) -> float:
    """Alternative loss: match the P90/median ratio (distributional
    *shape*) only, ignoring absolute scale.

    Useful specifically when `absolute_moment_loss` can't be satisfied at
    any fleet size because the simulator's own dispatch mechanism produces
    a structurally different-shaped wait-time distribution than reality —
    fleet size can shift the whole distribution's scale, but it can't
    change its shape, so asking it to match a shape it structurally can't
    produce (via the absolute loss) picks a fleet size that's a compromise
    between two demands, neither actually met. This asks only for the
    ratio to match, at whatever absolute scale the fleet size can offer,
    isolating "does *this* fleet size get the shape as close as
    possible" from "is the shape even matchable at all" (a separate
    question this loss can't answer either — if the true minimum ratio
    error is still large, that's a real finding about the dispatch
    mechanism, not something a different loss function can fix).
    """
    if simulated_wait.size == 0 or observed_median <= 0:
        return float("inf")
    sim_median = float(np.median(simulated_wait))
    if sim_median <= 0:
        return float("inf")
    sim_p90 = float(np.percentile(simulated_wait, 90))
    observed_ratio = observed_p90 / observed_median
    sim_ratio = sim_p90 / sim_median
    return (sim_ratio - observed_ratio) ** 2


def calibrate_fleet_size(
    observed_wait_seconds: np.ndarray,
    simulate_fn: Callable[[int], np.ndarray],
    candidate_fleet_sizes: list[int],
    loss_fn: Callable[[np.ndarray, float, float], float] = absolute_moment_loss,
) -> FleetSizeCalibrationResult:
    """Grid-search fleet size to match the observed wait-time distribution.

    `simulate_fn(fleet_size)` must return an array of simulated wait times
    (seconds) for that candidate fleet size. Kept as a callback rather than
    importing the engine directly so this is unit-testable against a cheap
    fake before being pointed at the real simulator (see `simulator.runner`).
    `loss_fn` defaults to `absolute_moment_loss` (median + p90 in seconds,
    this function's original and still-primary behavior); pass `ratio_loss`
    to match distributional shape only — see that function's docstring for
    when that's the more honest question to ask.
    """
    observed_median = float(np.median(observed_wait_seconds))
    observed_p90 = float(np.percentile(observed_wait_seconds, 90))

    losses = {
        n: loss_fn(simulate_fn(n), observed_median, observed_p90) for n in candidate_fleet_sizes
    }
    best_n = min(losses, key=losses.get)
    hit_boundary = best_n == min(candidate_fleet_sizes) or best_n == max(candidate_fleet_sizes)
    return FleetSizeCalibrationResult(
        fleet_size=best_n, loss=losses[best_n], candidates=losses, hit_boundary=hit_boundary
    )


def ks_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    """KS distance between observed and simulated wait-time distributions.

    Used for validation against a pre-registered acceptance threshold (P1),
    distinct from the median/p90 loss used to calibrate fleet size.
    """
    return float(stats.ks_2samp(observed, simulated).statistic)
