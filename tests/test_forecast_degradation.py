from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from dispatch_eval.forecast_degradation import (
    DegradationLevel,
    build_degraded_models,
    build_policy_ladder,
    flatten_travel_time_model,
    homogenize_arrival_model,
    marginalize_od_model,
    run_degradation_sweep,
    spatially_uniform_arrival_model,
)
from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNED_PARAMS = {
    "B1_batched_hungarian": {"best_params": {"delta": 15.0, "radius": 600.0}},
    "B2_value_corrected": {"best_params": {"delta": 15.0, "radius": 600.0, "value_weight": 1.0}},
    "B3_fluid_balancing": {"best_params": {"delta": 15.0, "radius": 600.0, "value_weight": 1.0}},
    "B4_sampling_lookahead": {
        "best_params": {"delta": 15.0, "radius": 600.0, "value_weight": 1.0, "lookahead": 120.0}
    },
}

ZONES = ["A", "B", "C"]


# --- homogenize_arrival_model --------------------------------------------------


def test_homogenize_arrival_model_preserves_average_rate_per_zone_day_type():
    model = NHPPArrivalModel(
        rates={("A", "all", 0): 1.0, ("A", "all", 1): 3.0, ("B", "all", 0): 10.0},
        bin_minutes=15,
    )
    out = homogenize_arrival_model(model)

    assert out.rates[("A", "all", 0)] == pytest.approx(2.0)
    assert out.rates[("A", "all", 1)] == pytest.approx(2.0)
    assert out.rates[("B", "all", 0)] == pytest.approx(10.0)  # unaffected, only one bin
    assert out.bin_minutes == 15


def test_homogenize_arrival_model_keeps_zones_independent():
    model = NHPPArrivalModel(
        rates={("A", "all", 0): 4.0, ("A", "all", 1): 0.0, ("B", "all", 0): 1.0, ("B", "all", 1): 1.0},
        bin_minutes=15,
    )
    out = homogenize_arrival_model(model)
    assert out.rates[("A", "all", 0)] == pytest.approx(2.0)
    assert out.rates[("B", "all", 0)] == pytest.approx(1.0)  # B's own average, not mixed with A


# --- spatially_uniform_arrival_model --------------------------------------------


def test_spatially_uniform_arrival_model_shares_one_rate_across_zones_and_time():
    model = NHPPArrivalModel(
        rates={("A", "all", 0): 4.0, ("A", "all", 1): 0.0, ("B", "all", 0): 1.0, ("B", "all", 1): 1.0},
        bin_minutes=15,
    )
    out = spatially_uniform_arrival_model(model)
    overall_average = (4.0 + 0.0 + 1.0 + 1.0) / 4  # == 1.5
    assert all(rate == pytest.approx(overall_average) for rate in out.rates.values())
    assert out.rates.keys() == model.rates.keys()


def test_spatially_uniform_is_more_severe_than_homogenize():
    # homogenize keeps B's average (1.0) distinct from A's (2.0);
    # spatially_uniform collapses both to the single citywide average.
    model = NHPPArrivalModel(
        rates={("A", "all", 0): 4.0, ("A", "all", 1): 0.0, ("B", "all", 0): 1.0, ("B", "all", 1): 1.0},
        bin_minutes=15,
    )
    homogenized = homogenize_arrival_model(model)
    uniform = spatially_uniform_arrival_model(model)
    assert homogenized.rates[("A", "all", 0)] != homogenized.rates[("B", "all", 0)]
    assert uniform.rates[("A", "all", 0)] == uniform.rates[("B", "all", 0)]


# --- marginalize_od_model --------------------------------------------------------


def test_marginalize_od_model_averages_across_cells():
    model = ODModel(
        dest_probs={
            ("A", 0): {"A": 1.0, "B": 0.0},
            ("A", 1): {"A": 0.0, "B": 1.0},
        },
        fallback_probs={"A": 0.5, "B": 0.5},
    )
    out = marginalize_od_model(model)

    # every original key now maps to the SAME averaged distribution
    assert out.dest_probs[("A", 0)] == out.dest_probs[("A", 1)]
    assert out.dest_probs[("A", 0)]["A"] == pytest.approx(0.5)
    assert out.dest_probs[("A", 0)]["B"] == pytest.approx(0.5)


def test_marginalize_od_model_result_sums_to_one():
    model = ODModel(
        dest_probs={
            ("A", 0): {"A": 0.2, "B": 0.3, "C": 0.5},
            ("B", 0): {"A": 0.9, "B": 0.1, "C": 0.0},
        },
        fallback_probs={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
    )
    out = marginalize_od_model(model)
    for probs in out.dest_probs.values():
        assert sum(probs.values()) == pytest.approx(1.0)
    assert sum(out.fallback_probs.values()) == pytest.approx(1.0)


# --- flatten_travel_time_model ---------------------------------------------------


def test_flatten_travel_time_model_zeroes_sigma_keeps_mu():
    model = TravelTimeModel(
        params={("A", "B", 8): (5.0, 0.4), ("A", "C", 9): (6.0, 0.2)},
        fallback_params=(4.0, 0.5),
    )
    out = flatten_travel_time_model(model)

    assert out.params[("A", "B", 8)] == (5.0, 0.0)
    assert out.params[("A", "C", 9)] == (6.0, 0.0)
    assert out.fallback_params == (4.0, 0.0)


def test_flatten_travel_time_model_makes_expected_equal_exp_mu():
    model = TravelTimeModel(params={}, fallback_params=(float(np.log(300.0)), 0.5))
    out = flatten_travel_time_model(model)
    # without the sigma^2/2 correction term, expected() == exp(mu) exactly
    assert out.expected("X", "Y", 10) == pytest.approx(300.0)


# --- build_degraded_models --------------------------------------------------------


@pytest.fixture(scope="module")
def trips_df():
    return generate_synthetic_trips(
        n_days=6, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=30, seed=3
    )


def test_no_flags_and_no_n_days_matches_plain_fit(trips_df):
    degraded = build_degraded_models(trips_df)
    plain = fit_all_models(trips_df)

    assert degraded.arrival.rates == plain.arrival.rates
    assert degraded.od.dest_probs == plain.od.dest_probs
    assert degraded.travel_time.params == plain.travel_time.params


def test_n_days_fits_on_a_subset_not_the_full_data(trips_df):
    degraded = build_degraded_models(trips_df, n_days=1)
    plain = fit_all_models(trips_df)
    # fitting on 1 of 6 days should produce a visibly different arrival fit
    assert degraded.arrival.rates != plain.arrival.rates


def test_structural_flags_compose_independently(trips_df):
    degraded = build_degraded_models(
        trips_df, homogenize_arrival=True, flatten_travel_time=True, marginalize_od=False
    )
    plain = fit_all_models(trips_df)

    # arrival: homogenized -> flat within each (zone, day_type)
    by_zone_day: dict[tuple[str, str], set[float]] = {}
    for (zone, day_type, _bin), rate in degraded.arrival.rates.items():
        by_zone_day.setdefault((zone, day_type), set()).add(round(rate, 9))
    assert all(len(rates) == 1 for rates in by_zone_day.values())

    # travel time: flattened -> every sigma is 0
    assert all(sigma == 0.0 for (_mu, sigma) in degraded.travel_time.params.values())
    assert degraded.travel_time.fallback_params[1] == 0.0

    # od: NOT degraded (flag was False) -> matches the plain fit exactly
    assert degraded.od.dest_probs == plain.od.dest_probs


# --- build_policy_ladder / run_degradation_sweep (integration) -----------------


def test_build_policy_ladder_uses_degraded_models_not_true_ones(trips_df):
    true_models = fit_all_models(trips_df)
    degraded = build_degraded_models(trips_df, homogenize_arrival=True, flatten_travel_time=True)

    policies = build_policy_ladder(degraded, TUNED_PARAMS, ZONES, day_type="all")

    assert set(policies.keys()) == {
        "B0_nearest_idle", "B1_batched_hungarian", "B2_value_corrected",
        "B3_fluid_balancing", "B4_sampling_lookahead",
    }
    b3 = policies["B3_fluid_balancing"]
    assert isinstance(b3, FluidZoneBalancingPolicy)
    assert b3.arrival_model is degraded.arrival
    assert b3.arrival_model is not true_models.arrival
    b4 = policies["B4_sampling_lookahead"]
    assert isinstance(b4, SamplingLookaheadPolicy)
    assert b4.arrival_model is degraded.arrival


def test_run_degradation_sweep_end_to_end(trips_df):
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    levels = [
        DegradationLevel("true_forecast"),
        DegradationLevel(
            "fully_degraded", n_days=1, homogenize_arrival=True, marginalize_od=True,
            flatten_travel_time=True,
        ),
    ]
    results = run_degradation_sweep(
        trips_df, ZONES, TUNED_PARAMS, fleet_size=15, abandonment_model=abandonment_model,
        config=config, n_bootstrap=3, n_replications=2, seed=5, levels=levels,
    )

    assert set(results.keys()) == {"true_forecast", "fully_degraded"}
    for result in results.values():
        assert set(result.policy_names) == set(TUNED_PARAMS.keys()) | {"B0_nearest_idle"}
        for name in result.policy_names:
            assert result.metric_by_policy[name].shape == (3, 2)
