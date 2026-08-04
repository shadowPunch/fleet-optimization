"""The single entrypoint that reproduces the confirmatory study end to end.

Per the project plan's reproducibility-discipline requirement: *"store
(policy, θ_bootstrap_index, replication_seed) → metrics in a single
parquet results table; never recompute a headline number from a notebook
cell. One run_study.py that regenerates every figure from raw data."*
Flagged as not yet built in `docs/pre_registration.md`; this is that.

Pipeline, all against `docs/pre_registration.md`'s committed settings
(primary metric = mean wait, α = 0.05, n_bootstrap ≥ 40):

1. Load trip data (synthetic — no real Delhi NCR/Bengaluru trip-level data
   has been available in this environment; see README's data-status
   section. `--data-source` exists as a switch for when that changes, not
   because a real path is implemented yet).
2. Fit nominal input models and calibrate fleet size against the observed
   wait-time distribution (P1).
3. Tune B1-B4 with an identical evaluation budget (P2 parity condition),
   the same procedure as `analysis/policy_tuning_run.py`.
4. Run the full bootstrap-CRN ranking-flip experiment with B5's
   clairvoyant bound wired in (P3 + the B5 gap-closed normalization).
5. Write every (policy, bootstrap_draw, replication) cell — not just
   summary statistics — to one parquet file. Every headline number this
   script prints (nominal ranking, P(ranked first), indifference set,
   variance decomposition, gap-closed) is computed from that same
   in-memory table, not recomputed separately, so the parquet file and
   the printed numbers can never silently drift apart.

Usage: uv run python run_study.py [--n-bootstrap 40] [--n-replications 4]
                                   [--seed 2026] [--output results/study_results.parquet]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from dispatch_eval.calibration.fleet_size import calibrate_fleet_size
from dispatch_eval.calibration.tuning import ParameterSpec, random_search
from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import (
    fit_all_models,
    indifference_set,
    run_ranking_flip_experiment,
    variance_decomposition,
)
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C", "D", "E"]
HORIZON_SECONDS = 6 * 3600.0
MEAN_PATIENCE_SECONDS = 300.0  # one point from the pre-registered sweep {60,180,300,600,900}
N_TUNING_EVALUATIONS = 25
CANDIDATE_FLEET_SIZES = [10, 15, 20, 25, 30, 40, 50, 60]
BASELINE_POLICY = "B0_nearest_idle"


def tune_policies(models, value_function, abandonment_model) -> dict:
    def eval_with(policy_fn, param_specs):
        def evaluate_fn(params):
            config = StudyConfig(
                zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS,
                dispatch_interval_seconds=params["delta"],
            )
            result = run_simulation(
                30, models.arrival, models.od, models.travel_time, abandonment_model,
                policy_fn(params), config, np.random.default_rng(0),
            )
            return result.mean_wait_seconds

        return random_search(evaluate_fn, param_specs, n_evaluations=N_TUNING_EVALUATIONS, seed=17)

    r1 = eval_with(
        lambda p: BatchedHungarianPolicy(matching_radius_seconds=p["radius"]),
        [ParameterSpec("delta", 5.0, 120.0), ParameterSpec("radius", 120.0, 2400.0, log_scale=True)],
    )
    r2 = eval_with(
        lambda p: ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=p["value_weight"], matching_radius_seconds=p["radius"]
        ),
        [
            ParameterSpec("delta", 5.0, 120.0),
            ParameterSpec("radius", 120.0, 2400.0, log_scale=True),
            ParameterSpec("value_weight", 0.0, 3.0),
        ],
    )

    b2_tuned = ValueCorrectedHungarianPolicy(
        value_function=value_function,
        value_weight=r2.best_params["value_weight"],
        matching_radius_seconds=r2.best_params["radius"],
    )
    return {
        "B0_nearest_idle": NearestIdlePolicy(),
        "B1_batched_hungarian": BatchedHungarianPolicy(matching_radius_seconds=r1.best_params["radius"]),
        "B2_value_corrected": b2_tuned,
        "B3_fluid_balancing": FluidZoneBalancingPolicy(
            dispatch_policy=b2_tuned, arrival_model=models.arrival, day_type="all"
        ),
        "B4_sampling_lookahead": SamplingLookaheadPolicy(
            dispatch_policy=b2_tuned, arrival_model=models.arrival, day_type="all", lookahead_seconds=300.0
        ),
    }


def flatten_to_long_table(result, seed: int) -> pl.DataFrame:
    """(policy, bootstrap_draw, replication) -> metrics, long format — the
    plan's own required results table shape. `replication_seed` is the
    exact SeedSequence entropy (`[seed, bootstrap_draw, replication]`)
    that reproduces that cell's realized scenario, not a separate random
    value — recorded as a string so it round-trips through parquet/CSV
    exactly.
    """
    rows = []
    n_bootstrap, n_replications = next(iter(result.metric_by_policy.values())).shape
    for policy_name, metrics in result.metric_by_policy.items():
        for b in range(n_bootstrap):
            for r in range(n_replications):
                rows.append(
                    {
                        "policy": policy_name,
                        "bootstrap_draw": b,
                        "replication": r,
                        "replication_seed": f"[{seed}, {b}, {r}]",
                        "wait_seconds": float(metrics[b, r]),
                        "clairvoyant_wait_seconds": (
                            float(result.clairvoyant_metric_by_draw[b, r])
                            if result.clairvoyant_metric_by_draw is not None
                            else None
                        ),
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=40)  # pre-registered minimum
    parser.add_argument("--n-replications", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/study_results.parquet"))
    parser.add_argument(
        "--data-source", choices=["synthetic"], default="synthetic",
        help="Only 'synthetic' is implemented — no real trip-level data has been available in this environment.",
    )
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading data ({args.data_source})...")
    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )

    print("Fitting nominal input models and calibrating fleet size (P1)...")
    models = fit_all_models(trips_df)
    abandonment_model = AbandonmentModel(mean_patience_seconds=MEAN_PATIENCE_SECONDS)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )

    observed_wait = trips_df.filter(pl.col("status") == "completed")
    observed_wait_seconds = (
        (observed_wait["pickup_ts"] - observed_wait["request_ts"]).dt.total_seconds().to_numpy()
    )

    def simulate_fn(fleet_size: int):
        config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
        result = run_simulation(
            fleet_size, models.arrival, models.od, models.travel_time, abandonment_model,
            NearestIdlePolicy(), config, np.random.default_rng(0),
        )
        return result.wait_times

    fleet_calibration = calibrate_fleet_size(observed_wait_seconds, simulate_fn, CANDIDATE_FLEET_SIZES)
    print(f"  calibrated fleet_size={fleet_calibration.fleet_size} (loss={fleet_calibration.loss:.1f})")

    print(f"Tuning B1-B4 ({N_TUNING_EVALUATIONS} evaluations each, identical budget — P2)...")
    policies = tune_policies(models, value_function, abandonment_model)

    print(
        f"Running the bootstrap-CRN ranking-flip experiment "
        f"(B={args.n_bootstrap}, R={args.n_replications}, seed={args.seed} — P3 + B5)..."
    )
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, fleet_calibration.fleet_size, abandonment_model, config,
        args.n_bootstrap, args.n_replications, args.seed, compute_clairvoyant=True,
    )

    table = flatten_to_long_table(result, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.output)
    print(f"\nWrote {table.height} rows to {args.output}")

    indifferent = indifference_set(result, alpha=0.05)
    probs = result.probability_ranked_first()
    gap_closed = result.fraction_of_gap_closed(BASELINE_POLICY)
    decomp = variance_decomposition(
        result.metric_by_policy[result.nominal_ranking[0]],
        result.metric_by_policy[result.nominal_ranking[-1]],
    )

    print(f"\nNominal ranking (best first): {result.nominal_ranking}")
    print(f"P(ranked first): {probs}")
    print(f"Indifference set (alpha=0.05): {sorted(indifferent)}")
    print(f"Fraction of clairvoyant gap closed (mean per policy): "
          f"{ {name: float(arr.mean()) for name, arr in gap_closed.items()} }")
    print(
        f"Variance decomposition ({result.nominal_ranking[0]} vs {result.nominal_ranking[-1]}): "
        f"input_uncertainty_ratio={decomp['input_uncertainty_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
