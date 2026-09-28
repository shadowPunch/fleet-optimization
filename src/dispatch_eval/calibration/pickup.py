"""Joint calibration of the two latent supply parameters.

Neither the fleet size on the road nor how long a same-zone pickup takes is
observable in trip records, and both shift the wait-time distribution.
They are calibrated together, by grid search, against the calibration
days' observed wait times — never against held-out data.

The same-zone pickup spread (lognormal sigma) is not searched: it is taken
from the dispersion of real same-zone trip durations, the only observed
within-zone movement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from dispatch_eval.calibration.fleet_size import ks_distance


@dataclass
class SupplyCalibration:
    fleet_size: int
    intra_zone_median_seconds: float
    intra_zone_sigma: float
    loss: float  # KS distance to the observed wait distribution
    grid: list[dict]  # one row per (fleet_size, intra_zone_median_seconds)
    hit_boundary: bool  # best point on an edge of the grid: widen it before trusting

    @property
    def intra_zone_params(self) -> tuple[float, float]:
        return (float(np.log(self.intra_zone_median_seconds)), self.intra_zone_sigma)


def same_zone_log_sigma(trips: pl.DataFrame) -> float:
    """Std of log duration over trips that start and end in the same zone."""
    log_durations = (
        trips.filter(pl.col("origin_zone") == pl.col("dest_zone"))
        .select((pl.col("dropoff_ts") - pl.col("pickup_ts")).dt.total_seconds().alias("d"))
        .filter(pl.col("d") > 0)["d"]
        .log()
    )
    return float(log_durations.std())


def calibrate_supply(
    observed_wait_seconds: np.ndarray,
    simulate_fn: Callable[[int, tuple[float, float]], np.ndarray],
    fleet_sizes: list[int],
    intra_zone_medians: list[float],
    intra_zone_sigma: float,
    on_point: Callable[[dict], None] | None = None,
) -> SupplyCalibration:
    """Grid-search (fleet size, same-zone pickup median) minimizing the KS
    distance between simulated and observed wait times.

    `simulate_fn(fleet_size, intra_zone_params)` returns simulated waits.
    `on_point(row)` is called after each grid point (progress/tracking).
    """
    grid = []
    for fleet_size in fleet_sizes:
        for median in intra_zone_medians:
            params = (float(np.log(median)), intra_zone_sigma)
            simulated = simulate_fn(fleet_size, params)
            loss = ks_distance(observed_wait_seconds, simulated) if simulated.size else 1.0
            row = {
                "fleet_size": fleet_size,
                "intra_zone_median_seconds": median,
                "ks_distance": loss,
                "simulated_median_wait": float(np.median(simulated)) if simulated.size else None,
            }
            grid.append(row)
            if on_point is not None:
                on_point(row)

    best = min(grid, key=lambda row: row["ks_distance"])
    hit_boundary = best["fleet_size"] in (min(fleet_sizes), max(fleet_sizes)) or best[
        "intra_zone_median_seconds"
    ] in (min(intra_zone_medians), max(intra_zone_medians))
    return SupplyCalibration(
        fleet_size=best["fleet_size"],
        intra_zone_median_seconds=best["intra_zone_median_seconds"],
        intra_zone_sigma=intra_zone_sigma,
        loss=best["ks_distance"],
        grid=grid,
        hit_boundary=hit_boundary,
    )
