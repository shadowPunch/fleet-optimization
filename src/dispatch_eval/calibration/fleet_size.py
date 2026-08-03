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


def _loss(simulated_wait: np.ndarray, observed_median: float, observed_p90: float) -> float:
    if simulated_wait.size == 0:
        return float("inf")
    sim_median = float(np.median(simulated_wait))
    sim_p90 = float(np.percentile(simulated_wait, 90))
    return (sim_median - observed_median) ** 2 + (sim_p90 - observed_p90) ** 2


def calibrate_fleet_size(
    observed_wait_seconds: np.ndarray,
    simulate_fn: Callable[[int], np.ndarray],
    candidate_fleet_sizes: list[int],
) -> FleetSizeCalibrationResult:
    """Grid-search fleet size to match observed wait-time median + p90.

    `simulate_fn(fleet_size)` must return an array of simulated wait times
    (seconds) for that candidate fleet size. Kept as a callback rather than
    importing the engine directly so this is unit-testable against a cheap
    fake before being pointed at the real simulator (see `simulator.runner`).
    """
    observed_median = float(np.median(observed_wait_seconds))
    observed_p90 = float(np.percentile(observed_wait_seconds, 90))

    losses = {
        n: _loss(simulate_fn(n), observed_median, observed_p90) for n in candidate_fleet_sizes
    }
    best_n = min(losses, key=losses.get)
    return FleetSizeCalibrationResult(fleet_size=best_n, loss=losses[best_n], candidates=losses)


def ks_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    """KS distance between observed and simulated wait-time distributions.

    Used for validation against a pre-registered acceptance threshold (P1),
    distinct from the median/p90 loss used to calibrate fleet size.
    """
    return float(stats.ks_2samp(observed, simulated).statistic)
