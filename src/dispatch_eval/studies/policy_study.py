"""The NYC dispatch policy study: which algorithm should run, and how sure
can we be?

For one supply regime (a fleet size within, or below, the range the wait
data is consistent with):

Policies are tuned to minimize mean wait subject to serving no fewer riders
than greedy dispatch; the experiment then reports both wait and service.

1. Build the validated NYC twin (`studies.validation`) at the regime's
   fleet size and same-zone pickup time (`configs/nyc.yaml`).
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
    FleetWaitCurve,
    build_fleet_wait_curve,
    monotone_fleet_equivalent,
)
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.registry import POLICY_LABELS, build_ladder, tune_ladder
from dispatch_eval.ranking_flip import (
    RankingFlipResult,
    indifference_set,
    merge_results,
    rank_draws,
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


def paired_vehicles_worth(
    result: RankingFlipResult, curve: FleetWaitCurve, horizon_hours: float,
    reference: str = REFERENCE_POLICY,
) -> dict[str, dict]:
    """Extra vehicles greedy dispatch would need to match each policy.

    Per bootstrap draw, each policy's wait is converted to the fleet size at
    which greedy reaches it on the (monotone) nominal curve, and greedy's
    own draw is subtracted, so the offset between the nominal curve and each
    draw's refitted world cancels. A policy whose wait beats greedy's best
    wait anywhere on the curve gets the curve's largest fleet — a lower
    bound — and the share of such draws is reported.
    """
    def equivalents(name: str) -> tuple[np.ndarray, np.ndarray]:
        per_draw = result.metric_by_policy[name].mean(axis=1)
        pairs = [monotone_fleet_equivalent(curve, w) for w in per_draw]
        return np.array([f for f, _ in pairs]), np.array([side for _, side in pairs])

    ref_fleet, _ = equivalents(reference)
    out = {}
    for name in result.policy_names:
        fleet, side = equivalents(name)
        worth = fleet - ref_fleet
        mean = float(worth.mean())
        out[name] = {
            "vehicles_worth_mean": mean,
            "vehicles_worth_ci": _interval(worth),
            "fraction_draws_lower_bound": float(np.mean(side == 1)),
            "delta_driver_hours_mean": mean * horizon_hours,
        }
    return out


def long_table(result: RankingFlipResult, regime: str) -> pl.DataFrame:
    """(regime, policy, bootstrap_draw, replication) -> metrics, one row per cell."""
    served = (result.secondary_metrics or {}).get("fraction_served", {})
    frames = []
    for name in result.policy_names:
        waits = result.metric_by_policy[name]
        n_draws, n_reps = waits.shape
        ids = result.draw_ids if result.draw_ids is not None else np.arange(n_draws)
        draws, reps = np.meshgrid(ids, np.arange(n_reps), indexing="ij")
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


def result_from_long_table(
    table: pl.DataFrame, policy_names: list[str], nominal_ranking: list[str]
) -> RankingFlipResult:
    """Inverse of `long_table`: rebuild the per-cell arrays of one regime."""
    table = table.sort("bootstrap_draw", "replication")
    draw_ids = table["bootstrap_draw"].unique(maintain_order=True).to_numpy()
    n_reps = table["replication"].n_unique()

    def cells(name: str, column: str) -> np.ndarray:
        rows = table.filter(pl.col("policy") == name)[column].to_numpy()
        return rows.reshape(len(draw_ids), n_reps)

    metric = {n: cells(n, "mean_wait_seconds") for n in policy_names}
    return RankingFlipResult(
        policy_names=policy_names,
        metric_by_policy=metric,
        rankings=rank_draws(metric, policy_names),
        nominal_ranking=nominal_ranking,
        secondary_metrics={"fraction_served": {n: cells(n, "fraction_served") for n in policy_names}},
        draw_ids=draw_ids,
    )


def _shard_stem(regime: str, draws: range) -> str:
    return f"nyc_study_{regime}.shard{draws.start:04d}-{draws.stop:04d}"


def finalize_study(meta: dict, result: RankingFlipResult, output_dir: Path) -> dict:
    """Summarize a complete (or merged) experiment and write the regime's
    `nyc_study_<regime>.{json,parquet}`."""
    regime = meta["regime"]
    curve = FleetWaitCurve(**meta["fleet_curve"])
    summary = {
        **meta,
        "n_draws_completed": len(result.draw_ids) if result.draw_ids is not None else None,
        **summarize(result),
        "vehicles_worth": paired_vehicles_worth(result, curve, meta["horizon_hours"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    long_table(result, regime).write_parquet(output_dir / f"nyc_study_{regime}.parquet")
    (output_dir / f"nyc_study_{regime}.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def merge_policy_study(regime: str, output_dir: Path) -> dict:
    """Combine every shard of `regime` in `output_dir` into the final study
    files. Shards must share identical metadata (same config and tuning)."""
    metas = sorted(output_dir.glob(f"nyc_study_{regime}.shard*.json"))
    if not metas:
        raise FileNotFoundError(f"no shards for regime {regime!r} in {output_dir}")
    shard_meta = [json.loads(m.read_text()) for m in metas]
    comparable = [{k: v for k, v in m.items() if k not in ("draws", "runtime_minutes")}
                  for m in shard_meta]
    if any(c != comparable[0] for c in comparable):
        raise ValueError("shards were produced with different settings; refusing to merge")
    meta = comparable[0]
    shards = [
        result_from_long_table(
            pl.read_parquet(path.with_suffix(".parquet")), meta["policy_names"],
            meta["nominal_ranking"],
        )
        for path in metas
    ]
    merged = merge_results(shards)
    meta["runtime_minutes"] = sum(m["runtime_minutes"] for m in shard_meta)
    return finalize_study(meta, merged, output_dir)


def refresh_policy_study(regime: str, output_dir: Path, horizon_hours: float) -> dict:
    """Recompute a finished regime's summary from its saved per-cell table,
    without re-running any simulation (e.g. after a change to how results
    are summarized). Fills metadata that older result files lack."""
    summary = json.loads((output_dir / f"nyc_study_{regime}.json").read_text())
    table = pl.read_parquet(output_dir / f"nyc_study_{regime}.parquet")
    derived = {"n_draws_completed", "policies", "best_policy", "indifference_set",
               "kendall_tau_mean", "fraction_draws_matching_nominal",
               "variance_decomposition_reference_vs_best", "vehicles_worth"}
    meta = {k: v for k, v in summary.items() if k not in derived}
    meta.setdefault("policy_names", table["policy"].unique(maintain_order=True).to_list())
    meta.setdefault("horizon_hours", horizon_hours)
    result = result_from_long_table(table, meta["policy_names"], meta["nominal_ranking"])
    return finalize_study(meta, result, output_dir)


def run_policy_study(
    cfg: dict,
    regime: str,
    validation: dict,
    output_dir: Path,
    n_workers: int,
    n_bootstrap: int | None = None,
    draws: range | None = None,
) -> dict:
    """Run one regime; with `draws` set, run only that shard of the bootstrap
    draws and write shard files for `merge_policy_study` instead."""
    study = cfg["study"]
    regime_cfg = study["regimes"][regime]
    n_bootstrap = n_bootstrap or study["n_bootstrap"]
    draws = draws or range(n_bootstrap)
    twin = build_twin(cfg)
    sim = cfg["simulation"]

    intra = (float(np.log(regime_cfg["intra_zone_median_seconds"])), validation["intra_zone_sigma"])
    fleet_size = regime_cfg["fleet_size"]
    models = twin.models
    models.travel_time = models.travel_time.with_intra_zone(intra)
    horizon = twin.config.horizon_seconds

    run_config = {
        **cfg,
        "regime": regime,
        "fleet_size": fleet_size,
        "n_bootstrap": n_bootstrap,
        "n_workers": n_workers,
        "draws": [draws.start, draws.stop],
        "intra_zone_params": intra,
        "validation_summary": {k: v for k, v in validation.items() if not isinstance(v, list)},
    }
    name = f"nyc-study-{regime}" + ("" if len(draws) == n_bootstrap else f"-d{draws.start}-{draws.stop}")
    with tracked_run(name, "policy-study", run_config, tags=["nyc", regime]) as run:
        value_function = compute_value_function(
            twin.data.zones, models.arrival, models.od, models.travel_time, "all", horizon,
            od_bin_minutes=sim["od_bin_minutes"],
        )

        # Mean wait over completed rides alone rewards turning riders away
        # (a tight pickup radius strands hard requests), so tuning is
        # constrained: a candidate must serve at least as many riders as
        # greedy dispatch, within `service_tolerance`.
        reference = twin.simulate(fleet_size, intra, study["tuning_seed"])
        min_served = reference.fraction_served - study["service_tolerance"]

        def evaluate(policy) -> float:
            result = twin.simulate(fleet_size, intra, study["tuning_seed"], policy)
            return result.mean_wait_seconds if result.fraction_served >= min_served else float("inf")

        print(f"[{regime}] tuning ({study['tuning_evaluations']} evaluations per policy)...", flush=True)
        params, tuning = tune_ladder(
            evaluate, value_function, models.arrival, study["tuning_evaluations"],
            study["tuning_seed"],
        )
        print(f"[{regime}] tuned {params.as_dict()}", flush=True)
        infeasible = [name for name, t in tuning.items() if not np.isfinite(t.best_score)]
        if infeasible:
            print(f"[{regime}] WARNING: no tuning candidate met the service constraint for "
                  f"{infeasible}; their parameters are arbitrary", flush=True)
        run.log_table(
            "tuning_history",
            [
                {"policy": name, "evaluation": i, "score": score, "params": json.dumps(p)}
                for name, t in tuning.items()
                for i, (p, score) in enumerate(t.history)
            ],
        )
        policies = build_ladder(params, value_function, models.arrival)

        print(f"[{regime}] fleet-equivalence curve for {REFERENCE_POLICY}...", flush=True)
        curve_sizes = sorted({round(fleet_size * s) for s in study["fleet_curve_scales"]})
        curve = build_fleet_wait_curve(
            curve_sizes, models, twin.abandonment, NearestIdlePolicy(), twin.config,
            study["fleet_curve_replications"], study["seed"],
        )

        meta = {
            "regime": regime,
            "fleet_size": fleet_size,
            "n_bootstrap": n_bootstrap,
            "n_replications": study["n_replications"],
            "policy_names": list(policies),
            "tuned_params": params.as_dict(),
            "tuning_best_scores": {name: t.best_score for name, t in tuning.items()},
            "tuning_infeasible": infeasible,
            "reference_fraction_served": reference.fraction_served,
            "fleet_curve": {"fleet_sizes": curve.fleet_sizes,
                            "mean_wait_seconds": curve.mean_wait_seconds},
            "horizon_hours": horizon / 3600.0,
        }

        stem = _shard_stem(regime, draws)

        def save_shard(partial: RankingFlipResult, runtime_minutes: float) -> None:
            # Also the checkpoint: a killed run still leaves its completed draws
            # in a mergeable shard file.
            output_dir.mkdir(parents=True, exist_ok=True)
            long_table(partial, regime).write_parquet(output_dir / f"{stem}.parquet")
            shard_meta = {**meta, "nominal_ranking": partial.nominal_ranking,
                          "runtime_minutes": runtime_minutes, "draws": [draws.start, draws.stop]}
            (output_dir / f"{stem}.json").write_text(json.dumps(shard_meta, indent=2, default=float))

        started = time.time()
        draws_done = 0

        def on_draw_done(b: int) -> None:
            nonlocal draws_done
            draws_done += 1
            elapsed = time.time() - started
            print(f"[{regime}] draw {b} done ({draws_done}/{len(draws)}, "
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
            draws=draws,
            checkpoint=lambda partial: save_shard(partial, (time.time() - started) / 60),
        )
        meta["nominal_ranking"] = result.nominal_ranking


        meta["runtime_minutes"] = (time.time() - started) / 60

        if len(draws) < n_bootstrap:
            save_shard(result, meta["runtime_minutes"])
            print(f"[{regime}] wrote shard {stem}; combine with `dispatch-eval merge`")
            return meta

        summary = finalize_study(meta, result, output_dir)
        for suffix in (".json", ".parquet"):  # the full run supersedes its checkpoint
            (output_dir / f"{stem}{suffix}").unlink(missing_ok=True)
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
