"""The NYC dispatch policy study: which algorithm should run, and how sure
can we be?

For one supply regime (the calibrated fleet, optionally scaled down to model
a driver shortage):

1. Build the validated NYC twin (`studies.validation`) at the calibrated
   same-zone pickup time.
2. Tune every policy in the ladder with an identical budget (`tune_ladder`).
3. Run the bootstrap-CRN experiment (`ranking_flip`): resample trips, refit
   every input model, run every policy on identical request streams.
4. Price each policy's gain in vehicles: how many extra cars greedy
   dispatch would need to match it (`decision_currency`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.decision_currency import (
    build_fleet_wait_curve,
    decision_currency,
    mean_fare_per_trip,
    trips_per_vehicle_per_day,
)
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.registry import POLICY_LABELS, build_ladder, tune_ladder
from dispatch_eval.ranking_flip import (
    RankingFlipResult,
    indifference_set,
    run_ranking_flip_experiment,
    variance_decomposition,
)
from dispatch_eval.studies.validation import build_twin
from dispatch_eval.tracking import tracked_run

REFERENCE_POLICY = "greedy"


def fraction_served(result) -> float:
    # Module-level (not a lambda) so spawned bootstrap workers can pickle it.
    return result.fraction_served


def _interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    tail = 100 * (1 - confidence) / 2
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def summarize(result: RankingFlipResult, reference: str = REFERENCE_POLICY) -> dict:
    """Headline numbers from one bootstrap-CRN experiment.

    Intervals are 95% bootstrap-percentile intervals across draws of each
    draw's replication mean; improvements are paired per draw (CRN).
    """
    served = (result.secondary_metrics or {}).get("fraction_served", {})
    p_first = result.probability_ranked_first()
    ref_wait = result.metric_by_policy[reference].mean(axis=1)

    policies = []
    for name in result.policy_names:
        wait = result.metric_by_policy[name].mean(axis=1)
        gain = ref_wait - wait
        row = {
            "policy": name,
            "label": POLICY_LABELS.get(name, name),
            "mean_wait_seconds": float(wait.mean()),
            "mean_wait_ci": _interval(wait),
            "gain_vs_reference_seconds": float(gain.mean()),
            "gain_vs_reference_ci": _interval(gain),
            "gain_vs_reference_pct": float(100 * (gain / ref_wait).mean()),
            "probability_ranked_first": p_first[name],
        }
        if name in served:
            per_draw = served[name].mean(axis=1)
            row["fraction_served"] = float(per_draw.mean())
            row["fraction_served_ci"] = _interval(per_draw)
        policies.append(row)

    best = min(policies, key=lambda p: p["mean_wait_seconds"])["policy"]
    taus = result.kendall_tau_to_nominal()
    return {
        "policies": policies,
        "best_policy": best,
        "nominal_ranking": result.nominal_ranking,
        "indifference_set": sorted(indifference_set(result)),
        "kendall_tau_mean": float(taus.mean()),
        "fraction_draws_matching_nominal": float(np.mean(taus == 1.0)),
        "variance_decomposition_reference_vs_best": variance_decomposition(
            result.metric_by_policy[reference], result.metric_by_policy[best]
        ),
    }


def long_table(result: RankingFlipResult, regime: str) -> pl.DataFrame:
    """(regime, policy, bootstrap_draw, replication) -> metrics, one row per cell."""
    served = (result.secondary_metrics or {}).get("fraction_served", {})
    frames = []
    for name in result.policy_names:
        waits = result.metric_by_policy[name]
        n_draws, n_reps = waits.shape
        draws, reps = np.meshgrid(np.arange(n_draws), np.arange(n_reps), indexing="ij")
        frames.append(
            pl.DataFrame(
                {
                    "regime": regime,
                    "policy": name,
                    "bootstrap_draw": draws.ravel(),
                    "replication": reps.ravel(),
                    "mean_wait_seconds": waits.ravel(),
                    "fraction_served": served[name].ravel() if name in served else np.nan,
                }
            )
        )
    return pl.concat(frames)


def run_policy_study(
    cfg: dict,
    regime: str,
    validation: dict,
    output_dir: Path,
    n_workers: int,
    n_bootstrap: int | None = None,
) -> dict:
    study = cfg["study"]
    regime_cfg = study["regimes"][regime]
    n_bootstrap = n_bootstrap or study["n_bootstrap"]
    twin = build_twin(cfg)
    sim = cfg["simulation"]

    intra = (float(np.log(validation["intra_zone_median_seconds"])), validation["intra_zone_sigma"])
    fleet_size = round(validation["fleet_size"] * regime_cfg["fleet_scale"])
    models = twin.models
    models.travel_time = models.travel_time.with_intra_zone(intra)
    horizon = twin.config.horizon_seconds

    run_config = {
        **cfg,
        "regime": regime,
        "fleet_size": fleet_size,
        "n_bootstrap": n_bootstrap,
        "n_workers": n_workers,
        "intra_zone_params": intra,
        "validation_summary": {k: v for k, v in validation.items() if not isinstance(v, list)},
    }
    with tracked_run(f"nyc-study-{regime}", "policy-study", run_config, tags=["nyc", regime]) as run:
        value_function = compute_value_function(
            twin.data.zones, models.arrival, models.od, models.travel_time, "all", horizon,
            od_bin_minutes=sim["od_bin_minutes"],
        )

        def evaluate(policy) -> float:
            return twin.simulate(fleet_size, intra, study["tuning_seed"], policy).mean_wait_seconds

        print(f"[{regime}] tuning ({study['tuning_evaluations']} evaluations per policy)...", flush=True)
        params, tuning = tune_ladder(
            evaluate, value_function, models.arrival, study["tuning_evaluations"],
            study["tuning_seed"],
        )
        print(f"[{regime}] tuned {params.as_dict()}", flush=True)
        run.log_table(
            "tuning_history",
            [
                {"policy": name, "evaluation": i, "score": score, "params": json.dumps(p)}
                for name, t in tuning.items()
                for i, (p, score) in enumerate(t.history)
            ],
        )
        policies = build_ladder(params, value_function, models.arrival)

        started = time.time()
        draws_done = 0

        def on_draw_done(b: int) -> None:
            nonlocal draws_done
            draws_done += 1
            elapsed = time.time() - started
            print(f"[{regime}] draw {b} done ({draws_done}/{n_bootstrap}, "
                  f"{elapsed / 60:.1f} min)", flush=True)
            run.log({"draws_done": draws_done, "elapsed_minutes": elapsed / 60})

        result = run_ranking_flip_experiment(
            twin.data.for_simulation(twin.data.calibration),
            twin.data.zones,
            policies,
            fleet_size,
            twin.abandonment,
            twin.config,
            n_bootstrap,
            study["n_replications"],
            study["seed"],
            bin_minutes=sim["arrival_bin_minutes"],
            od_time_bin_minutes=sim["od_bin_minutes"],
            intra_zone_params=intra,
            secondary_metric_fns={"fraction_served": fraction_served},
            n_workers=n_workers,
            on_draw_done=on_draw_done,
        )

        print(f"[{regime}] fleet-equivalence curve for {REFERENCE_POLICY}...", flush=True)
        curve_sizes = sorted({round(fleet_size * s) for s in study["fleet_curve_scales"]})
        curve = build_fleet_wait_curve(
            curve_sizes, models, twin.abandonment, NearestIdlePolicy(), twin.config,
            study["fleet_curve_replications"], study["seed"],
        )
        reference_run = twin.simulate(fleet_size, intra, study["seed"])
        currency = decision_currency(
            result,
            REFERENCE_POLICY,
            curve,
            fleet_size,
            horizon_hours=horizon / 3600.0,
            dollars_per_trip=mean_fare_per_trip(twin.data.calibration),
            reference_trips_per_vehicle_per_day=trips_per_vehicle_per_day(
                len(reference_run.completed_requests), fleet_size, horizon / 3600.0
            ),
        )

        summary = {
            "regime": regime,
            "fleet_size": fleet_size,
            "n_bootstrap": n_bootstrap,
            "n_replications": study["n_replications"],
            "tuned_params": params.as_dict(),
            "runtime_minutes": (time.time() - started) / 60,
            **summarize(result),
            "fleet_curve": {"fleet_sizes": curve.fleet_sizes, "mean_wait_seconds": curve.mean_wait_seconds},
            "vehicles_worth": {name: c.summary() for name, c in currency.items()},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        long_table(result, regime).write_parquet(output_dir / f"nyc_study_{regime}.parquet")
        (output_dir / f"nyc_study_{regime}.json").write_text(json.dumps(summary, indent=2, default=float))

        run.log_table("policy_summary", [
            {k: (json.dumps(v) if isinstance(v, tuple | list) else v) for k, v in p.items()}
            for p in summary["policies"]
        ])
        run.summary.update({
            "best_policy": summary["best_policy"],
            "kendall_tau_mean": summary["kendall_tau_mean"],
            "indifference_set": ",".join(summary["indifference_set"]),
            "runtime_minutes": summary["runtime_minutes"],
        })
    return summary
