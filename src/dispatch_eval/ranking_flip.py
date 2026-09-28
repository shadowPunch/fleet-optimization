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
running (see TECHNICAL_REPORT.md's note on the CRN bug); and re-seeding a fresh
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

import multiprocessing
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import kendalltau, norm

from dispatch_eval.calibration.arrivals import fit_arrival_model
from dispatch_eval.calibration.boarding import fit_boarding_model
from dispatch_eval.calibration.fare import fit_fare_model
from dispatch_eval.calibration.od import fit_od_model
from dispatch_eval.calibration.travel_time import fit_travel_time_model
from dispatch_eval.clairvoyant import solve_clairvoyant_schedule
from dispatch_eval.models import (
    AbandonmentModel,
    BoardingModel,
    FareModel,
    NHPPArrivalModel,
    ODModel,
    TravelTimeModel,
)
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.scenario import generate_scenario
from dispatch_eval.simulator.engine import SimulationResult
from dispatch_eval.simulator.entities import Vehicle
from dispatch_eval.simulator.runner import StudyConfig, run_simulation


@dataclass
class FittedModels:
    arrival: NHPPArrivalModel
    od: ODModel
    travel_time: TravelTimeModel
    fare: FareModel
    boarding: BoardingModel | None = None


def bootstrap_resample_trips(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """A standard nonparametric bootstrap: resample rows with replacement,
    same size as the original. This is D*_b in the plan's P3 pseudocode."""
    return df[bootstrap_indices(df.height, rng)]


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def fit_all_models(
    df: pl.DataFrame,
    day_type_col: str | None = None,
    bin_minutes: int = 15,
    od_time_bin_minutes: int = 60,
    intra_zone_params: tuple[float, float] | None = None,
) -> FittedModels:
    """Refit every (non-latent, non-structural) input model on `df` in one
    call — the theta*_b step. See the module docstring for why fleet size
    and the abandonment hazard are excluded from this refit.

    `intra_zone_params` is the calibrated same-zone pickup time (latent,
    see `calibration.pickup`), carried into every refit unchanged; None
    keeps `TravelTimeModel`'s default.
    """
    travel_time = fit_travel_time_model(df)
    if intra_zone_params is not None:
        travel_time.intra_zone_params = intra_zone_params
    return FittedModels(
        arrival=fit_arrival_model(df, day_type_col=day_type_col, bin_minutes=bin_minutes),
        od=fit_od_model(df, time_bin_minutes=od_time_bin_minutes),
        travel_time=travel_time,
        fare=fit_fare_model(df),
        boarding=fit_boarding_model(df),
    )


@dataclass
class RankingFlipResult:
    policy_names: list[str]
    metric_by_policy: dict[str, np.ndarray]  # each: shape (n_bootstrap, n_replications)
    rankings: np.ndarray  # shape (n_bootstrap, n_policies): policy indices, best first
    nominal_ranking: list[str]  # ranking fit on the un-resampled data
    clairvoyant_metric_by_draw: np.ndarray | None = None  # shape (n_bootstrap, n_replications)
    clairvoyant_fraction_excluded: np.ndarray | None = None  # same shape; see clairvoyant.py
    # metric name -> policy -> (n_bootstrap, n_replications); recorded, not ranked on
    secondary_metrics: dict[str, dict[str, np.ndarray]] | None = None

    def fraction_of_gap_closed(self, baseline_policy: str) -> dict[str, np.ndarray]:
        """B5's own normalization (project plan, B5 section): every policy's
        metric expressed as the fraction of the gap between
        `baseline_policy`'s metric and the clairvoyant bound that it
        closes, per (bootstrap draw, replication) cell — 0.0 = no better
        than baseline, 1.0 = matches the clairvoyant bound. Values outside
        [0, 1] are possible: below 0 means worse than baseline; above 1
        means beating the clairvoyant bound, which is possible if
        `clairvoyant_fraction_excluded` isn't near zero (see
        `clairvoyant.py`'s module docstring — a `bin_minutes` too coarse
        for the abandonment hazard's timescale silently drops servable
        requests before the solver ever sees them, which is a real bug to
        fix, not the clairvoyant solve's own honest upper-bound property).
        Check `clairvoyant_fraction_excluded` before trusting values above
        1 as a real finding rather than this bug.

        Requires `run_ranking_flip_experiment(..., compute_clairvoyant=True)`.
        """
        if self.clairvoyant_metric_by_draw is None:
            raise ValueError(
                "clairvoyant bound was not computed for this result — "
                "call run_ranking_flip_experiment with compute_clairvoyant=True"
            )
        baseline = self.metric_by_policy[baseline_policy]
        denom = baseline - self.clairvoyant_metric_by_draw
        return {
            name: np.where(denom != 0, (baseline - self.metric_by_policy[name]) / denom, np.nan)
            for name in self.policy_names
        }

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


@dataclass
class _DrawOutcome:
    b: int
    metrics: dict[str, np.ndarray]  # policy -> shape (n_replications,)
    secondary: dict[str, dict[str, np.ndarray]]  # metric -> policy -> (n_replications,)
    clairvoyant_metric: np.ndarray | None
    clairvoyant_excluded: np.ndarray | None


@dataclass
class _DrawSpec:
    """Everything one bootstrap draw needs, bundled so a worker process can
    run draws without re-receiving the trip data each time."""

    trips_df: pl.DataFrame
    zones: list[str]
    policies: dict[str, DispatchPolicy]
    fleet_size: int
    abandonment_model: AbandonmentModel
    config: StudyConfig
    n_replications: int
    seed: int
    metric_fn: Callable[[SimulationResult], float]
    secondary_metric_fns: dict[str, Callable[[SimulationResult], float]]
    bin_minutes: int
    od_time_bin_minutes: int
    compute_clairvoyant: bool
    clairvoyant_bin_minutes: float
    policy_travel_time_model: TravelTimeModel | None
    intra_zone_params: tuple[float, float] | None

    def simulate(
        self, policy: DispatchPolicy, models: FittedModels, rng: np.random.Generator
    ) -> SimulationResult:
        return run_simulation(
            self.fleet_size,
            models.arrival,
            models.od,
            models.travel_time,
            self.abandonment_model,
            policy,
            self.config,
            rng,
            policy_travel_time_model=self.policy_travel_time_model,
            boarding_model=models.boarding,
        )

    def run_draw(self, b: int, resample_indices: np.ndarray) -> _DrawOutcome:
        models_b = fit_all_models(
            self.trips_df[resample_indices],
            bin_minutes=self.bin_minutes,
            od_time_bin_minutes=self.od_time_bin_minutes,
            intra_zone_params=self.intra_zone_params,
        )
        metrics = {name: np.zeros(self.n_replications) for name in self.policies}
        secondary = {
            m: {name: np.zeros(self.n_replications) for name in self.policies}
            for m in self.secondary_metric_fns
        }
        cv_metric = np.zeros(self.n_replications) if self.compute_clairvoyant else None
        cv_excluded = np.zeros(self.n_replications) if self.compute_clairvoyant else None

        for r in range(self.n_replications):
            if self.compute_clairvoyant:
                cv_metric[r], cv_excluded[r] = self._clairvoyant(models_b, b, r)
            for name, policy in self.policies.items():
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, b, r]))
                result = self.simulate(policy, models_b, rng)
                metrics[name][r] = self.metric_fn(result)
                for m, fn in self.secondary_metric_fns.items():
                    secondary[m][name][r] = fn(result)

        return _DrawOutcome(b, metrics, secondary, cv_metric, cv_excluded)

    def _clairvoyant(self, models_b: FittedModels, b: int, r: int) -> tuple[float, float]:
        # Same (seed, b, r) seed as every policy's run, so the same scenario.
        scenario_rng = np.random.default_rng(np.random.SeedSequence([self.seed, b, r]))
        scenario = generate_scenario(
            self.zones,
            self.config.day_type,
            models_b.arrival,
            models_b.od,
            self.abandonment_model,
            self.config.horizon_seconds,
            scenario_rng,
            self.config.od_bin_minutes,
            models_b.boarding,
        )
        vehicles = [
            Vehicle(vehicle_id=f"veh-{i}", zone=self.zones[i % len(self.zones)])
            for i in range(self.fleet_size)
        ]
        cv_result = solve_clairvoyant_schedule(
            scenario.requests,
            vehicles,
            models_b.travel_time,
            self.config.horizon_seconds,
            self.clairvoyant_bin_minutes,
        )
        cv_wait = cv_result.wait_times
        return (
            float(cv_wait.mean()) if cv_wait.size else float("inf"),
            cv_result.fraction_excluded_by_discretization,
        )


_worker_spec: _DrawSpec | None = None


def _set_worker_spec(spec: _DrawSpec) -> None:
    global _worker_spec
    _worker_spec = spec


def _run_draw_in_worker(b: int, resample_indices: np.ndarray) -> _DrawOutcome:
    return _worker_spec.run_draw(b, resample_indices)


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
    bin_minutes: int = 15,
    od_time_bin_minutes: int = 60,
    compute_clairvoyant: bool = False,
    clairvoyant_bin_minutes: float = 15.0,
    policy_travel_time_model: TravelTimeModel | None = None,
    intra_zone_params: tuple[float, float] | None = None,
    secondary_metric_fns: dict[str, Callable[[SimulationResult], float]] | None = None,
    n_workers: int = 1,
    on_draw_done: Callable[[int], None] | None = None,
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

    `bin_minutes`/`od_time_bin_minutes` are passed straight through to every
    `fit_all_models` call (nominal and every bootstrap draw) — the defaults
    match `fit_all_models`'s own, so this is a no-op unless a caller
    overrides them. `coarsening_ladder.py` (C6) is the reason this is a
    parameter here rather than hardcoded: re-running this exact experiment
    at coarser time resolution is exactly its "how much of the ranking
    survives" question, and duplicating this whole loop elsewhere to vary
    two numbers would violate DRY for no benefit.

    `compute_clairvoyant`, if set, additionally solves B5's offline bound
    (`solve_clairvoyant_schedule`) once per (b, r) — not once per policy,
    since it doesn't depend on which policy is being compared — on the
    *same* realized scenario every policy sees at that (b, r): a second,
    independently-created `np.random.Generator` seeded identically to each
    policy's own (via the same `(seed, b, r)` `SeedSequence`) reproduces it
    exactly, since scenario generation is the first and only thing that
    seed's state determines before any policy-specific randomness begins.
    See `RankingFlipResult.fraction_of_gap_closed` for what this enables —
    off by default since it adds B x R min-cost-flow solves on top of this
    function's existing B x R x |policies| simulation cost, for a result
    most callers don't need.

    `policy_travel_time_model`, if given, is what every policy's own
    `dispatch()`/`reposition()` calls see instead of the true, per-draw
    `models_b.travel_time` — passed straight through to `run_simulation`.
    Defaults to `None` (each policy sees the true model, current
    behavior). The clairvoyant solve above always uses the true model
    regardless of this — it represents an oracle over the *actual* world,
    not a policy's belief about it. See `forecast_degradation.py` for why
    a caller would set this: every policy here already implicitly
    forecasts through whatever `travel_time_model` it receives, and B2-B4
    additionally bake a value function / arrival-rate reference into their
    own construction from models fit the *same way* the true scenario is
    — an oracle-forecast confound this parameter exists to break.

    `n_workers > 1` runs bootstrap draws in parallel worker processes.
    Results are identical to a serial run: every draw's resample indices
    are still generated in order from one stream in this process, and each
    (b, r) cell's randomness depends only on its own seed.
    `on_draw_done(b)` is called as each draw finishes (in completion order).
    `secondary_metric_fns` (e.g. fraction served) are recorded per cell in
    `RankingFlipResult.secondary_metrics` but play no part in ranking.
    """
    policy_names = list(policies.keys())

    # SeedSequence entropy must be plain non-negative ints (no strings) — these
    # sentinels just need to not collide with real (b, r) index values.
    _NOMINAL_SENTINEL = 0x4E4F4D31  # "NOM1"
    _RESAMPLE_SENTINEL = 0x52455331  # "RES1"

    draw_spec = _DrawSpec(
        trips_df=trips_df,
        zones=zones,
        policies=policies,
        fleet_size=fleet_size,
        abandonment_model=abandonment_model,
        config=config,
        n_replications=n_replications,
        seed=seed,
        metric_fn=metric_fn,
        secondary_metric_fns=secondary_metric_fns or {},
        bin_minutes=bin_minutes,
        od_time_bin_minutes=od_time_bin_minutes,
        compute_clairvoyant=compute_clairvoyant,
        clairvoyant_bin_minutes=clairvoyant_bin_minutes,
        policy_travel_time_model=policy_travel_time_model,
        intra_zone_params=intra_zone_params,
    )

    nominal_models = fit_all_models(
        trips_df,
        bin_minutes=bin_minutes,
        od_time_bin_minutes=od_time_bin_minutes,
        intra_zone_params=intra_zone_params,
    )
    nominal_metric: dict[str, float] = {}
    for name, policy in policies.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed, _NOMINAL_SENTINEL]))
        result = draw_spec.simulate(policy, nominal_models, rng)
        nominal_metric[name] = metric_fn(result)
    nominal_ranking = sorted(policy_names, key=lambda name: nominal_metric[name])

    metric_by_policy = {name: np.zeros((n_bootstrap, n_replications)) for name in policy_names}
    secondary = {
        m: {name: np.zeros((n_bootstrap, n_replications)) for name in policy_names}
        for m in draw_spec.secondary_metric_fns
    }
    clairvoyant_metric = np.zeros((n_bootstrap, n_replications)) if compute_clairvoyant else None
    clairvoyant_excluded = np.zeros((n_bootstrap, n_replications)) if compute_clairvoyant else None
    resample_rng = np.random.default_rng(np.random.SeedSequence([seed, _RESAMPLE_SENTINEL]))

    def store(outcome: _DrawOutcome) -> None:
        for name in policy_names:
            metric_by_policy[name][outcome.b] = outcome.metrics[name]
            for m in secondary:
                secondary[m][name][outcome.b] = outcome.secondary[m][name]
        if compute_clairvoyant:
            clairvoyant_metric[outcome.b] = outcome.clairvoyant_metric
            clairvoyant_excluded[outcome.b] = outcome.clairvoyant_excluded
        if on_draw_done is not None:
            on_draw_done(outcome.b)

    def next_indices() -> np.ndarray:
        return bootstrap_indices(trips_df.height, resample_rng)

    if n_workers <= 1:
        for b in range(n_bootstrap):
            store(draw_spec.run_draw(b, next_indices()))
    else:
        # `draw_spec` (trip data, policies) is sent once per worker via the
        # initializer; only the resample indices travel per draw. At most
        # 2 x n_workers draws are in flight, bounding memory. spawn, not
        # fork: polars' thread pool deadlocks in forked children.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            n_workers, mp_context=context, initializer=_set_worker_spec, initargs=(draw_spec,)
        ) as pool:
            in_flight: set[Future] = set()
            for b in range(n_bootstrap):
                if len(in_flight) >= 2 * n_workers:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        store(future.result())
                in_flight.add(pool.submit(_run_draw_in_worker, b, next_indices()))
            for future in in_flight:
                store(future.result())

    rankings = np.empty((n_bootstrap, len(policy_names)), dtype=int)
    for b in range(n_bootstrap):
        means = [metric_by_policy[name][b].mean() for name in policy_names]
        rankings[b] = np.argsort(means)

    return RankingFlipResult(
        policy_names=policy_names,
        metric_by_policy=metric_by_policy,
        clairvoyant_metric_by_draw=clairvoyant_metric,
        clairvoyant_fraction_excluded=clairvoyant_excluded,
        rankings=rankings,
        nominal_ranking=nominal_ranking,
        secondary_metrics=secondary or None,
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

    **This is a floor on input uncertainty's true contribution, not an
    estimate of it** — the plan's own three-part C1 decomposition is
    intrinsic noise, input-model estimation error, *and structural
    ambiguity*, but the bootstrap loop this function's inputs come from
    only ever refits within a fixed, correctly-specified model family
    (NHPP / smoothed-OD / lognormal travel time). Structural
    misspecification is exactly zero here by construction, so
    `across_theta_variance` (and therefore `input_uncertainty_ratio`) can
    only be an underestimate of what a real deployment would see. Use this
    function when only a single-abandonment-spec `RankingFlipResult` is
    available; `variance_decomposition_three_term` is the real
    three-term estimate (the abandonment-hazard sweep as the structural
    stratum), not a floor, when the outer sweep has been run.
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


def variance_decomposition_three_term(
    metric_a_by_spec: dict[float, np.ndarray], metric_b_by_spec: dict[float, np.ndarray]
) -> dict[str, float]:
    """C1's real three-term split: intrinsic simulation noise /
    input-model estimation error / structural ambiguity — not a floor.

    `metric_a_by_spec`/`metric_b_by_spec` map each pre-registered
    abandonment-hazard specification (`mean_patience_seconds`,
    `TECHNICAL_REPORT.md`: `{60, 180, 300, 600, 900}`, "a structural
    axis to sweep, not fit") to that spec's own `metric_by_policy` array
    (shape `(n_bootstrap, n_replications)`) from a `RankingFlipResult` run
    with that `AbandonmentModel` — the abandonment sweep as the outer loop
    around the existing bootstrap, one `run_ranking_flip_experiment` call
    per spec. Both dicts must share the same keys (specs).

    Extends `variance_decomposition`'s nested law-of-total-variance design
    by one more level, outermost: spec > bootstrap draw > replication.
    `diff[s, b, r]` is the paired (CRN) metric difference at spec `s`,
    draw `b`, replication `r`.

    - `intrinsic_variance`: mean, over (spec, draw), of the variance
      across replications within that cell — same quantity
      `variance_decomposition` calls `within_theta_variance`, now also
      averaged over specs.
    - `input_estimation_variance`: mean, over specs, of the variance
      across bootstrap draws' own means within that spec — same quantity
      `variance_decomposition` calls `across_theta_variance`, but now
      computed *within* a fixed abandonment spec before averaging, so
      structural ambiguity from *which spec is true* can't leak into it.
    - `structural_variance`: variance, across specs, of each spec's own
      overall mean — the term `variance_decomposition` could never
      produce, because its bootstrap loop only ever refits within one
      fixed, correctly-specified family. This is what makes
      `input_uncertainty_ratio` below an actual estimate of input-model
      estimation error's contribution, not an underestimate that also
      silently contains uncounted structural ambiguity.

    Doesn't cover every possible structural axis — only the abandonment
    hazard, the one this project's own P1 pre-registration already commits
    to treating as unfit/structural. Model-family misspecification
    (arrival/OD/travel-time functional form) is a separate structural
    question, checked separately by `forecast_degradation.py`
    (`TECHNICAL_REPORT.md`) and not folded in here.
    """
    specs = sorted(metric_a_by_spec)
    if sorted(metric_b_by_spec) != specs:
        raise ValueError("metric_a_by_spec and metric_b_by_spec must share the same spec keys")

    diffs = np.stack([metric_a_by_spec[s] - metric_b_by_spec[s] for s in specs])  # (S, B, R)

    intrinsic = float(diffs.var(axis=2, ddof=1).mean())
    draw_means = diffs.mean(axis=2)  # (S, B) -- per-(spec, draw) mean across replications
    input_estimation = float(draw_means.var(axis=1, ddof=1).mean())
    spec_means = draw_means.mean(axis=1)  # (S,) -- per-spec grand mean
    structural = float(spec_means.var(ddof=1)) if len(specs) > 1 else 0.0

    return {
        "intrinsic_variance": intrinsic,
        "input_estimation_variance": input_estimation,
        "structural_variance": structural,
        "total_variance": intrinsic + input_estimation + structural,
        "input_uncertainty_ratio": input_estimation / intrinsic if intrinsic > 0 else float("inf"),
        "structural_uncertainty_ratio": structural / intrinsic if intrinsic > 0 else float("inf"),
    }


def indifference_set(result: RankingFlipResult, alpha: float = 0.05) -> set[str]:
    """The set of policies that cannot be statistically separated from the
    best-performing policy at level alpha — plan's P3 output #3: "report
    this instead of a winner."

    "Best" is whichever policy has the lowest grand-mean metric across all
    (bootstrap draw, replication) cells. For every other policy, this builds
    a (1-alpha) bootstrap-percentile interval for (that policy's per-draw
    mean - best's per-draw mean) across the B bootstrap draws — paired,
    since both were run against the same theta*_b under CRN. If the
    interval contains zero, the policy is statistically indistinguishable
    from the best and belongs in the indifference set.

    No multiple-comparisons correction is applied across the simultaneous
    |policies|-1 comparisons against "best" — a documented simplification.
    With few policies (the ladder has 6) the uncorrected percentile
    intervals are a reasonable first cut; revisit if this needs to support
    many more candidates at once.
    """
    grand_means = {name: float(arr.mean()) for name, arr in result.metric_by_policy.items()}
    best_name = min(grand_means, key=grand_means.get)
    best_per_draw = result.metric_by_policy[best_name].mean(axis=1)

    indifferent = {best_name}
    for name in result.policy_names:
        if name == best_name:
            continue
        per_draw = result.metric_by_policy[name].mean(axis=1)
        diff = per_draw - best_per_draw
        lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        if lo <= 0 <= hi:
            indifferent.add(name)
    return indifferent


def subset_trips_by_days(
    df: pl.DataFrame, n_days: int, request_ts_col: str = "request_ts"
) -> pl.DataFrame:
    """The first `n_days` distinct calendar days of `df`, by `request_ts`.

    The building block `validate_mde_scaling` uses to empirically check how
    estimation variance actually shrinks with data volume, instead of just
    assuming the scaling `minimum_detectable_effect_curve` extrapolates
    from.
    """
    dates = df[request_ts_col].dt.date().unique().sort()
    keep_dates = set(dates.head(n_days).to_list())
    return df.filter(df[request_ts_col].dt.date().is_in(keep_dates))


def minimum_detectable_effect_curve(
    trips_df: pl.DataFrame,
    variance_decomposition_result: dict[str, float],
    candidate_days: list[int],
    confidence: float = 0.95,
    request_ts_col: str = "request_ts",
) -> dict[int, float]:
    """MDE(n) for each candidate number of real days used to fit input
    models — plan's P3 output #4: "the smallest true effect whose sign is
    resolved at [confidence]." The practically useful artefact: it tells a
    future researcher how much data they'd need before a claimed effect
    size is even resolvable.

    This is one call, not B x R x |policies| x |candidate_days| more
    simulation: it takes ONE already-computed `variance_decomposition`
    result (fit at `trips_df`'s actual number of days) and *extrapolates*,
    assuming across-theta variance (input-model estimation error) shrinks
    proportionally to 1/n_days — the standard asymptotic scaling for
    M-estimator / bootstrap variance — while within-theta variance
    (intrinsic simulator noise for a fixed, already-fitted theta) is held
    constant, since it reflects replication-to-replication randomness, not
    how much data theta was fitted from.

    That 1/n scaling is an assumption, not something measured here — see
    `validate_mde_scaling` for the empirical check against real data at
    several values of `n_days`.
    """
    z = float(norm.ppf(confidence))  # one-sided: P(estimate has the correct sign) = confidence
    n_days_fitted = trips_df[request_ts_col].dt.date().n_unique()
    within = variance_decomposition_result["within_theta_variance"]
    across = variance_decomposition_result["across_theta_variance"]

    mde: dict[int, float] = {}
    for n in candidate_days:
        scaled_across = across * (n_days_fitted / n)
        standard_error = np.sqrt(within + scaled_across)
        mde[n] = z * standard_error
    return mde


def validate_mde_scaling(
    trips_df: pl.DataFrame,
    zones: list[str],
    policy_a: DispatchPolicy,
    policy_b: DispatchPolicy,
    fleet_size: int,
    abandonment_model: AbandonmentModel,
    config: StudyConfig,
    candidate_n_days: list[int],
    n_bootstrap: int,
    n_replications: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    """Empirically checks `minimum_detectable_effect_curve`'s 1/n_days
    assumption for across-theta variance, instead of just asserting it.

    For each `n` in `candidate_n_days` (ascending), subsets `trips_df` to
    its first `n` calendar days (`subset_trips_by_days`) and re-runs the
    full bootstrap-CRN experiment for exactly two policies, measuring
    `variance_decomposition`'s `across_theta_variance` at that `n`. The
    *largest* `n` tested is the anchor — the same direction
    `minimum_detectable_effect_curve` itself extrapolates in, from more
    data down to less — so every `n`'s prediction is
    `variance(anchor_n) * (anchor_n / n)`, and `ratio_measured_to_predicted`
    is exactly 1.0 at the anchor by construction, not evidence the
    assumption holds; the real check is whether the smaller-`n` ratios stay
    near 1.0 too.

    This is `len(candidate_n_days)` full bootstrap experiments, not one —
    genuinely more expensive than the extrapolation it's checking, which is
    exactly why `minimum_detectable_effect_curve` doesn't do this itself by
    default.
    """
    sorted_days = sorted(candidate_n_days)
    anchor_n = sorted_days[-1]

    measured: dict[int, float] = {}
    for n in sorted_days:
        subset = subset_trips_by_days(trips_df, n)
        result = run_ranking_flip_experiment(
            subset,
            zones,
            {"a": policy_a, "b": policy_b},
            fleet_size,
            abandonment_model,
            config,
            n_bootstrap,
            n_replications,
            seed,
        )
        decomp = variance_decomposition(result.metric_by_policy["a"], result.metric_by_policy["b"])
        measured[n] = decomp["across_theta_variance"]

    anchor_variance = measured[anchor_n]
    out: dict[int, dict[str, float]] = {}
    for n in sorted_days:
        predicted = anchor_variance * (anchor_n / n)
        out[n] = {
            "measured_across_theta_variance": measured[n],
            "predicted_by_1_over_n": predicted,
            "ratio_measured_to_predicted": measured[n] / predicted if predicted > 0 else float("nan"),
        }
    return out
