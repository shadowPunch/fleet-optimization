"""Out-of-sample validation of the simulator against real NYC wait times.

1. Fit arrival, OD, travel-time and boarding models on the calibration days.
2. Calibrate the latent supply parameters (fleet size, same-zone pickup
   time) against the calibration days' wait times (`calibration.pickup`).
3. Simulate the calibrated twin and compare with the held-out days against
   the pre-registered thresholds: wait-time KS distance and hour-of-day
   demand-shape cosine similarity. Thresholds are never tuned to results.

The reference policy is nearest-idle dispatch, the standard stand-in for a
production matcher whose internals are not public.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.fleet_size import ks_distance
from dispatch_eval.calibration.pickup import calibrate_supply, same_zone_log_sigma
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import FittedModels, fit_all_models
from dispatch_eval.simulator.engine import SimulationResult
from dispatch_eval.simulator.runner import StudyConfig, run_simulation
from dispatch_eval.studies.nyc_data import (
    NYCData,
    NYCScope,
    approach_seconds,
    load_nyc_data,
    wait_seconds,
)
from dispatch_eval.tracking import tracked_run

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass
class NYCTwin:
    """Everything needed to simulate the calibrated NYC digital twin."""

    data: NYCData
    models: FittedModels
    config: StudyConfig
    abandonment: AbandonmentModel

    def simulate(
        self, fleet_size: int, intra_zone_params: tuple[float, float], seed: int, policy=None
    ) -> SimulationResult:
        return run_simulation(
            fleet_size,
            self.models.arrival,
            self.models.od,
            self.models.travel_time.with_intra_zone(intra_zone_params),
            self.abandonment,
            policy or NearestIdlePolicy(),
            self.config,
            np.random.default_rng(seed),
            boarding_model=self.models.boarding,
        )


def build_twin(cfg: dict) -> NYCTwin:
    scope = NYCScope.from_config(cfg["data"])
    data = load_nyc_data(scope, Path(cfg["data"]["cache_dir"]))
    sim = cfg["simulation"]
    models = fit_all_models(
        data.for_simulation(data.calibration),
        bin_minutes=sim["arrival_bin_minutes"],
        od_time_bin_minutes=sim["od_bin_minutes"],
    )
    config = StudyConfig(
        zones=data.zones,
        day_type="all",
        horizon_seconds=scope.horizon_seconds,
        dispatch_interval_seconds=sim["dispatch_interval_seconds"],
        od_bin_minutes=sim["od_bin_minutes"],
        reposition_interval_seconds=sim["reposition_interval_seconds"],
    )
    return NYCTwin(data, models, config, AbandonmentModel(sim["mean_patience_seconds"]))


def hourly_shape(hours_from_window_start: np.ndarray, n_hours: int) -> np.ndarray:
    counts = np.bincount(hours_from_window_start.astype(int), minlength=n_hours)[:n_hours]
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def run_validation(cfg: dict) -> dict:
    val = cfg["validation"]
    twin = build_twin(cfg)
    scope = twin.data.scope
    n_hours = scope.window_end_hour - scope.window_start_hour
    sigma = same_zone_log_sigma(twin.data.calibration)

    run_config = {**cfg, "intra_zone_sigma": sigma, "n_zones": len(twin.data.zones)}
    with tracked_run("nyc-validation", "validation", run_config, tags=["nyc", "validation"]) as run:
        supply = calibrate_supply(
            wait_seconds(twin.data.calibration),
            lambda fleet, params: twin.simulate(fleet, params, val["seed"]).wait_times,
            val["fleet_sizes"],
            val["intra_zone_median_seconds"],
            sigma,
            on_point=lambda row: print(f"  calibration point {row}", flush=True),
        )
        run.log_table("calibration_grid", supply.grid)

        sim_wait, sim_approach, sim_hours = [], [], []
        for r in range(val["n_replications"]):
            result = twin.simulate(
                supply.fleet_size, supply.intra_zone_params, val["seed"] + 100 + r
            )
            done = result.completed_requests
            sim_wait.append(result.wait_times)
            sim_approach.append(np.array([q.on_scene_time - q.request_time for q in done]))
            sim_hours.append(np.array([q.request_time // 3600.0 for q in done]))
        sim_wait = np.concatenate(sim_wait)
        sim_approach = np.concatenate(sim_approach)

        real_wait = wait_seconds(twin.data.held_out)
        real_hours = (
            twin.data.held_out["request_ts"].dt.hour().to_numpy() - scope.window_start_hour
        )
        ks = ks_distance(real_wait, sim_wait)
        cosine = cosine_similarity(
            hourly_shape(real_hours, n_hours), hourly_shape(np.concatenate(sim_hours), n_hours)
        )

        quantile_rows = [
            {
                "quantile": q,
                "held_out_wait": float(np.quantile(real_wait, q)),
                "simulated_wait": float(np.quantile(sim_wait, q)),
                "held_out_approach": float(np.quantile(approach_seconds(twin.data.held_out), q)),
                "simulated_approach": float(np.quantile(sim_approach, q)),
            }
            for q in QUANTILES
        ]
        report = {
            "fleet_size": supply.fleet_size,
            "intra_zone_median_seconds": supply.intra_zone_median_seconds,
            "intra_zone_sigma": sigma,
            "calibration_ks": supply.loss,
            "calibration_hit_boundary": supply.hit_boundary,
            "wait_ks_distance": ks,
            "wait_ks_pass": ks <= val["ks_threshold"],
            "approach_ks_distance": ks_distance(approach_seconds(twin.data.held_out), sim_approach),
            "hour_of_day_cosine": cosine,
            "hour_of_day_cosine_pass": cosine >= val["cosine_threshold"],
            "held_out_trips": int(real_wait.size),
            "simulated_completed": int(sim_wait.size),
            "quantiles": quantile_rows,
            "calibration_grid": supply.grid,
        }
        run.log_table("wait_quantiles", quantile_rows)
        run.summary.update({k: v for k, v in report.items() if not isinstance(v, list)})
    return report
