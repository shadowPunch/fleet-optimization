"""P3 — the ranking-flip experiment. The project's centerpiece.

The claim under test: at the effect sizes reported in the ride-hailing
dispatch literature, policy rankings are not statistically identified once
input-model estimation error is propagated. The procedure (see the project
plan, §P3):

    for b in 1..B:                          # B ~= 200-500
        D*_b <- bootstrap resample of the trip data
        theta*_b <- refit all input models on D*_b
        for each policy pi in the ladder:
            for r in 1..R:                  # R replications, CRN across policies
                y[pi, b, r] <- simulate(pi, theta*_b, seed_r)
        rank_b <- ordering of policies by mean_r y[pi, b, r]

Common random numbers across policies within a given (b, r) is
non-negotiable — it's what makes the paired comparison sharp. This module
gets it for free from two things already fixed/built earlier in the
project: `scenario.generate_scenario` makes the realized request trace
depend only on the fitted models and an rng seed, never on which policy is
running (see the README's note on the CRN bug); and re-seeding a fresh
`np.random.Generator` with the same value for every policy at a given
(b, r) makes the *rest* of the randomness (travel-time realizations) shared
too, up to the point where different policies make different decisions.

Scope decisions, both deliberate:
- Fleet size is calibrated once, outside the bootstrap loop, and held fixed
  across all B draws — recalibrating it per draw would itself require a
  full simulator-based search 200-500 times, which the plan's own
  computational note flags as exactly the kind of cost to avoid. Bootstrap
  uncertainty in fleet size specifically isn't propagated yet.
- The abandonment hazard is a structural axis to sweep across a handful of
  specifications (per P1), not something this harness bootstraps — pass a
  fixed `AbandonmentModel` in, and run this whole experiment again under a
  different one to get the sweep.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import kendalltau

from dispatch_eval.calibration.arrivals import fit_arrival_model
from dispatch_eval.calibration.fare import fit_fare_model
from dispatch_eval.calibration.od import fit_od_model
from dispatch_eval.calibration.travel_time import fit_travel_time_model
from dispatch_eval.models import (
    AbandonmentModel,
    FareModel,
    NHPPArrivalModel,
    ODModel,
    TravelTimeModel,
)
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.simulator.engine import SimulationResult
from dispatch_eval.simulator.runner import StudyConfig, run_simulation


@dataclass
class FittedModels:
    arrival: NHPPArrivalModel
    od: ODModel
    travel_time: TravelTimeModel
    fare: FareModel


def bootstrap_resample_trips(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """A standard nonparametric bootstrap: resample rows with replacement,
    same size as the original. This is D*_b in the plan's P3 pseudocode."""
    n = df.height
    indices = rng.integers(0, n, size=n)
    return df[indices]


def fit_all_models(
    df: pl.DataFrame,
    day_type_col: str | None = None,
    bin_minutes: int = 15,
    od_time_bin_minutes: int = 60,
) -> FittedModels:
    """Refit every (non-latent, non-structural) input model on `df` in one
    call — the theta*_b step. See the module docstring for why fleet size
    and the abandonment hazard are excluded from this refit.
    """
    return FittedModels(
        arrival=fit_arrival_model(df, day_type_col=day_type_col, bin_minutes=bin_minutes),
        od=fit_od_model(df, time_bin_minutes=od_time_bin_minutes),
        travel_time=fit_travel_time_model(df),
        fare=fit_fare_model(df),
    )


@dataclass
class RankingFlipResult:
    policy_names: list[str]
    metric_by_policy: dict[str, np.ndarray]  # each: shape (n_bootstrap, n_replications)
    rankings: np.ndarray  # shape (n_bootstrap, n_policies): policy indices, best first
    nominal_ranking: list[str]  # ranking fit on the un-resampled data

    def probability_ranked_first(self) -> dict[str, float]:
        """P(pi ranked first) for each policy, across bootstrap draws."""
        n_bootstrap = self.rankings.shape[0]
        counts = dict.fromkeys(self.policy_names, 0)
        for b in range(n_bootstrap):
            winner = self.policy_names[self.rankings[b, 0]]
            counts[winner] += 1
        return {name: c / n_bootstrap for name, c in counts.items()}

    def kendall_tau_to_nominal(self) -> np.ndarray:
        """Kendall's tau between each bootstrap ranking and the nominal one.

        Near 1 for every draw means the ranking is stable under resampling;
        spread toward 0 (or negative) means it isn't — this is the
        distribution the plan's histogram output is built from.
        """
        nominal_order = [self.policy_names.index(name) for name in self.nominal_ranking]
        taus = np.empty(self.rankings.shape[0])
        for b in range(self.rankings.shape[0]):
            tau, _ = kendalltau(nominal_order, self.rankings[b])
            taus[b] = tau
        return taus


def _mean_wait_seconds(result: SimulationResult) -> float:
    return result.mean_wait_seconds


def run_ranking_flip_experiment(
    trips_df: pl.DataFrame,
    zones: list[str],
    policies: dict[str, DispatchPolicy],
    fleet_size: int,
    abandonment_model: AbandonmentModel,
    config: StudyConfig,
    n_bootstrap: int,
    n_replications: int,
    seed: int,
    metric_fn: Callable[[SimulationResult], float] = _mean_wait_seconds,
) -> RankingFlipResult:
    """Run the full B x R x |policies| bootstrap-CRN experiment.

    `metric_fn` maps one simulation's result to the scalar used for ranking
    (lower is better — the default is mean wait time). For every (b, r), all
    policies are run against the identical realized scenario: a fresh
    `np.random.Generator` seeded from `(seed, b, r)` via `SeedSequence` is
    handed to each policy's `run_simulation` call in turn, and because
    scenario generation only depends on that seed and the fitted models
    (never on the policy), the request trace — and therefore the comparison
    — is genuinely paired.
    """
    policy_names = list(policies.keys())

    # SeedSequence entropy must be plain non-negative ints (no strings) — these
    # sentinels just need to not collide with real (b, r) index values.
    _NOMINAL_SENTINEL = 0x4E4F4D31  # "NOM1"
    _RESAMPLE_SENTINEL = 0x52455331  # "RES1"

    nominal_models = fit_all_models(trips_df)
    nominal_metric: dict[str, float] = {}
    for name, policy in policies.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed, _NOMINAL_SENTINEL]))
        result = run_simulation(
            fleet_size,
            nominal_models.arrival,
            nominal_models.od,
            nominal_models.travel_time,
            abandonment_model,
            policy,
            config,
            rng,
        )
        nominal_metric[name] = metric_fn(result)
    nominal_ranking = sorted(policy_names, key=lambda name: nominal_metric[name])

    metric_by_policy = {name: np.zeros((n_bootstrap, n_replications)) for name in policy_names}
    resample_rng = np.random.default_rng(np.random.SeedSequence([seed, _RESAMPLE_SENTINEL]))

    for b in range(n_bootstrap):
        resampled = bootstrap_resample_trips(trips_df, resample_rng)
        models_b = fit_all_models(resampled)
        for r in range(n_replications):
            for name, policy in policies.items():
                rng = np.random.default_rng(np.random.SeedSequence([seed, b, r]))
                result = run_simulation(
                    fleet_size,
                    models_b.arrival,
                    models_b.od,
                    models_b.travel_time,
                    abandonment_model,
                    policy,
                    config,
                    rng,
                )
                metric_by_policy[name][b, r] = metric_fn(result)

    rankings = np.empty((n_bootstrap, len(policy_names)), dtype=int)
    for b in range(n_bootstrap):
        means = [metric_by_policy[name][b].mean() for name in policy_names]
        rankings[b] = np.argsort(means)

    return RankingFlipResult(
        policy_names=policy_names,
        metric_by_policy=metric_by_policy,
        rankings=rankings,
        nominal_ranking=nominal_ranking,
    )


def variance_decomposition(metric_a: np.ndarray, metric_b: np.ndarray) -> dict[str, float]:
    """C1: split Var(y_a - y_b) into within-theta (intrinsic simulation
    noise) and across-theta (input-model estimation error) components.

    `metric_a`/`metric_b` are two policies' `metric_by_policy` arrays (shape
    (n_bootstrap, n_replications)) from the *same* `RankingFlipResult`, so
    the paired difference at each (b, r) is meaningful (CRN). Uses the law
    of total variance over the nested (bootstrap draw, replication) design:
    within-theta is the mean, across draws, of the variance across
    replications within a draw; across-theta is the variance, across draws,
    of each draw's mean. `input_uncertainty_ratio` is the headline number
    the plan calls for: "input uncertainty contributes Nx the variance of
    simulation noise."
    """
    diff = metric_a - metric_b
    within_theta = float(diff.var(axis=1, ddof=1).mean())
    across_theta = float(diff.mean(axis=1).var(ddof=1))
    return {
        "within_theta_variance": within_theta,
        "across_theta_variance": across_theta,
        "total_variance": within_theta + across_theta,
        "input_uncertainty_ratio": across_theta / within_theta
        if within_theta > 0
        else float("inf"),
    }
