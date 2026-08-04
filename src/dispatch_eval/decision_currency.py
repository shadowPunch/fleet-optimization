"""C3 — Decision currency: express a policy's wait-time gain as "worth N
vehicles," with uncertainty propagated from C2's bootstrap draws.

The idea (project plan, C3): sweep a *reference* policy's fleet size to get
its wait-time-vs-fleet curve, then invert that curve to ask "how many more
(or fewer) vehicles would the reference policy need to match this other
policy's wait time?" That reframes an abstract wait-time delta into
something an operator can act on — a fleet-size-equivalent, plus its
knock-on Δdriver-hours and Δ$/day.

Uncertainty comes entirely from C2 (`ranking_flip.RankingFlipResult`, whose
`metric_by_policy` array already varies across bootstrap draws) — the
fleet-size curve itself is built once, on the nominal (un-resampled) data,
and reused to invert every draw. This mirrors the scope decision
`ranking_flip.py` already documents for fleet size itself: recalibrating a
whole curve per bootstrap draw would multiply the simulation budget by the
number of curve points, which the plan's own computational note flags as
exactly the cost to avoid. Bootstrap uncertainty in the curve's own shape
isn't propagated — only in the wait-time gap the curve is used to invert.

Δ$/day is a documented approximation, not a simulated quantity: the
simulator's `Request` doesn't carry a fare (fares are only ever fit against
real/synthetic *input* data for calibration, never assigned to simulated
trips — see `calibration/fare.py`), so this module falls back to a flat
average fare-per-trip from the input data times a simulated
trips-per-vehicle-per-day rate. Δdriver-hours needs no such approximation:
`vehicle_hours` scales exactly with fleet size and horizon in the engine
(`fleet_size * horizon_seconds / 3600`, see `simulator/engine.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.ranking_flip import FittedModels, RankingFlipResult
from dispatch_eval.simulator.runner import StudyConfig, run_simulation


@dataclass
class FleetWaitCurve:
    """Mean wait time at each of a swept set of fleet sizes, for one policy
    under one fitted-model draw (in practice: the nominal, un-resampled fit).
    """

    fleet_sizes: list[int]
    mean_wait_seconds: list[float]

    def invert(self, target_wait_seconds: float) -> tuple[float, bool]:
        """The fleet size at which this curve's wait time equals `target_wait_seconds`.

        Linearly interpolates within the swept range. If `target_wait_seconds`
        falls outside it — the reference policy at every swept fleet size is
        either always better or always worse than the target — linearly
        extrapolates from the nearest two points instead of silently
        clipping to the boundary, which would understate the answer for any
        policy whose performance falls outside the swept range. Returns
        `(fleet_size_estimate, extrapolated)` so callers can flag it.
        """
        order = np.argsort(self.mean_wait_seconds)
        waits = np.asarray(self.mean_wait_seconds, dtype=float)[order]
        sizes = np.asarray(self.fleet_sizes, dtype=float)[order]

        if len(waits) < 2:
            raise ValueError("need at least 2 swept fleet sizes to invert a curve")

        if target_wait_seconds < waits[0]:
            slope = (sizes[1] - sizes[0]) / (waits[1] - waits[0])
            return float(sizes[0] + slope * (target_wait_seconds - waits[0])), True
        if target_wait_seconds > waits[-1]:
            slope = (sizes[-1] - sizes[-2]) / (waits[-1] - waits[-2])
            return float(sizes[-1] + slope * (target_wait_seconds - waits[-1])), True
        return float(np.interp(target_wait_seconds, waits, sizes)), False


def build_fleet_wait_curve(
    fleet_sizes: list[int],
    models: FittedModels,
    abandonment_model: AbandonmentModel,
    policy: DispatchPolicy,
    config: StudyConfig,
    n_replications: int,
    seed: int,
) -> FleetWaitCurve:
    """Run `policy` at each candidate fleet size, averaging over `n_replications`
    CRN-seeded runs per size (same `SeedSequence` scheme `ranking_flip.py` uses)."""
    means = []
    for fleet_size in fleet_sizes:
        per_rep = []
        for r in range(n_replications):
            rng = np.random.default_rng(np.random.SeedSequence([seed, fleet_size, r]))
            result = run_simulation(
                fleet_size,
                models.arrival,
                models.od,
                models.travel_time,
                abandonment_model,
                policy,
                config,
                rng,
            )
            per_rep.append(result.mean_wait_seconds)
        means.append(float(np.mean(per_rep)))
    return FleetWaitCurve(fleet_sizes=list(fleet_sizes), mean_wait_seconds=means)


def mean_fare_per_trip(trips_df: pl.DataFrame) -> float:
    """Flat $/trip rate from completed input trips — the Δ$/day approximation's
    only ingredient the simulator itself can't supply (see module docstring)."""
    completed = trips_df.filter(pl.col("status") == "completed")
    return float(completed["fare"].mean())


def trips_per_vehicle_per_day(n_completed: int, fleet_size: int, horizon_hours: float) -> float:
    horizon_days = horizon_hours / 24.0
    return n_completed / fleet_size / horizon_days


@dataclass
class DecisionCurrencyResult:
    policy_name: str
    reference_policy_name: str
    actual_fleet_size: int
    vehicles_worth_per_draw: np.ndarray  # shape (n_bootstrap,)
    extrapolated_per_draw: np.ndarray  # shape (n_bootstrap,), bool
    horizon_hours: float
    dollars_per_trip: float
    reference_trips_per_vehicle_per_day: float

    def summary(self, confidence: float = 0.95) -> dict:
        alpha = 1.0 - confidence
        lo, hi = np.percentile(
            self.vehicles_worth_per_draw, [100 * alpha / 2, 100 * (1 - alpha / 2)]
        )
        mean_vehicles = float(self.vehicles_worth_per_draw.mean())
        dollars_per_vehicle_per_day = self.reference_trips_per_vehicle_per_day * self.dollars_per_trip
        return {
            "policy": self.policy_name,
            "reference_policy": self.reference_policy_name,
            "vehicles_worth_mean": mean_vehicles,
            "vehicles_worth_ci": (float(lo), float(hi)),
            "confidence": confidence,
            "fraction_draws_extrapolated": float(self.extrapolated_per_draw.mean()),
            "delta_driver_hours_mean": mean_vehicles * self.horizon_hours,
            "delta_dollars_per_day_mean": mean_vehicles * dollars_per_vehicle_per_day,
        }


def decision_currency(
    result: RankingFlipResult,
    reference_policy_name: str,
    reference_curve: FleetWaitCurve,
    actual_fleet_size: int,
    horizon_hours: float,
    dollars_per_trip: float,
    reference_trips_per_vehicle_per_day: float,
) -> dict[str, DecisionCurrencyResult]:
    """For every policy in `result`, the bootstrap-draw distribution of "how
    many more/fewer vehicles would `reference_policy_name` need, at
    `reference_curve`, to match this policy's per-draw mean wait time."

    `reference_policy_name` need not be a member of `result.policy_names` —
    `reference_curve` is the only thing this function actually reads for it
    (built separately via `build_fleet_wait_curve`, since it requires a
    fleet-size sweep, not just the single-fleet-size runs C2 already did).
    Applying this to `reference_policy_name` itself, if it is present in
    `result`, is a useful sanity check: the answer should center on zero.
    """
    n_bootstrap = next(iter(result.metric_by_policy.values())).shape[0]
    out: dict[str, DecisionCurrencyResult] = {}
    for name in result.policy_names:
        per_draw_wait = result.metric_by_policy[name].mean(axis=1)
        vehicles_worth = np.empty(n_bootstrap)
        extrapolated = np.empty(n_bootstrap, dtype=bool)
        for b in range(n_bootstrap):
            fleet_equiv, was_extrapolated = reference_curve.invert(per_draw_wait[b])
            vehicles_worth[b] = fleet_equiv - actual_fleet_size
            extrapolated[b] = was_extrapolated
        out[name] = DecisionCurrencyResult(
            policy_name=name,
            reference_policy_name=reference_policy_name,
            actual_fleet_size=actual_fleet_size,
            vehicles_worth_per_draw=vehicles_worth,
            extrapolated_per_draw=extrapolated,
            horizon_hours=horizon_hours,
            dollars_per_trip=dollars_per_trip,
            reference_trips_per_vehicle_per_day=reference_trips_per_vehicle_per_day,
        )
    return out
