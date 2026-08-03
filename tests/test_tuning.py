from __future__ import annotations

import numpy as np
import pytest

from dispatch_eval.calibration.tuning import ParameterSpec, random_search
from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

ZONES = ["A", "B", "C"]


def test_random_search_recovers_a_known_optimum_on_a_deterministic_bowl():
    # score(x, y) is minimized at (3, -2); no simulator involved here, this
    # tests the search procedure itself.
    def evaluate_fn(params: dict[str, float]) -> float:
        return (params["x"] - 3.0) ** 2 + (params["y"] + 2.0) ** 2

    result = random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("x", low=-10.0, high=10.0),
            ParameterSpec("y", low=-10.0, high=10.0),
        ],
        n_evaluations=500,
        seed=0,
    )

    assert result.best_params["x"] == pytest.approx(3.0, abs=1.0)
    assert result.best_params["y"] == pytest.approx(-2.0, abs=1.0)
    assert len(result.history) == 500


def test_random_search_is_reproducible_given_the_same_seed():
    def evaluate_fn(params: dict[str, float]) -> float:
        return params["x"] ** 2

    specs = [ParameterSpec("x", low=-5.0, high=5.0)]
    first = random_search(evaluate_fn, specs, n_evaluations=20, seed=42)
    second = random_search(evaluate_fn, specs, n_evaluations=20, seed=42)
    assert first.best_params == second.best_params
    assert first.history == second.history


def test_maximize_flag_picks_the_highest_scoring_candidate():
    def evaluate_fn(params: dict[str, float]) -> float:
        return -((params["x"] - 4.0) ** 2)  # maximized at x=4

    result = random_search(
        evaluate_fn,
        param_specs=[ParameterSpec("x", low=-10.0, high=10.0)],
        n_evaluations=300,
        seed=1,
        minimize=False,
    )
    assert abs(result.best_params["x"] - 4.0) < 1.0


def _uniform_models(zones, rate_per_minute: float, mean_patience: float):
    rates = {(z, "all", b): rate_per_minute for z in zones for b in range(96)}
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=15)
    dest_probs = {(z, h): {zz: 1.0 / len(zones) for zz in zones} for z in zones for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(zones) for z in zones})
    params = {
        (o, d, h): (float(np.log(300.0)), 0.3) for o in zones for d in zones for h in range(24)
    }
    tt_model = TravelTimeModel(params=params, fallback_params=(float(np.log(300.0)), 0.3))
    abandonment_model = AbandonmentModel(mean_patience_seconds=mean_patience)
    return arrival_model, od_model, tt_model, abandonment_model


def test_tuning_batched_hungarian_delta_and_r_improves_on_a_bad_fixed_choice():
    arrival_model, od_model, tt_model, abandonment_model = _uniform_models(
        ZONES, rate_per_minute=0.5, mean_patience=240.0
    )

    def evaluate_fn(params: dict[str, float]) -> float:
        config = StudyConfig(
            zones=ZONES,
            day_type="all",
            horizon_seconds=2 * 3600.0,
            dispatch_interval_seconds=params["delta"],
        )
        result = run_simulation(
            fleet_size=15,
            arrival_model=arrival_model,
            od_model=od_model,
            travel_time_model=tt_model,
            abandonment_model=abandonment_model,
            policy=BatchedHungarianPolicy(matching_radius_seconds=params["r"]),
            config=config,
            rng=np.random.default_rng(0),  # same seed for every candidate: parity
        )
        return result.mean_wait_seconds

    result = random_search(
        evaluate_fn,
        param_specs=[
            ParameterSpec("delta", low=5.0, high=300.0),
            ParameterSpec("r", low=60.0, high=1200.0),
        ],
        n_evaluations=15,
        seed=0,
    )

    # a deliberately bad fixed choice: huge batch window, tiny matching radius
    bad_score = evaluate_fn({"delta": 300.0, "r": 60.0})

    assert result.best_score <= bad_score
