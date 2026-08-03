from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from dispatch_eval.calibration.arrivals import dispersion_by_zone, fit_arrival_model
from dispatch_eval.calibration.fare import fit_fare_model
from dispatch_eval.calibration.fleet_size import calibrate_fleet_size, ks_distance
from dispatch_eval.calibration.od import fit_od_model
from dispatch_eval.calibration.travel_time import fit_travel_time_model
from dispatch_eval.sources.synthetic import default_ground_truth, generate_synthetic_trips

ZONES = ["A", "B", "C"]


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_trips(
        n_days=15, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=11
    )


@pytest.fixture(scope="module")
def ground_truth():
    return default_ground_truth(ZONES, seed=11)


def test_fit_arrival_model_recovers_ground_truth_rates(synthetic_df, ground_truth):
    gt_arrival_model, *_ = ground_truth
    fitted = fit_arrival_model(synthetic_df, day_type_col=None, bin_minutes=15)

    for zone in ZONES:
        for b in (0, 36, 60, 72):
            truth = gt_arrival_model.rates[(zone, "all", b)]
            got = fitted.rates.get((zone, "all", b), 0.0)
            assert got == pytest.approx(truth, rel=0.35, abs=0.05)


def test_dispersion_by_zone_returns_one_row_per_zone(synthetic_df):
    result = dispersion_by_zone(synthetic_df, bin_minutes=15)
    assert set(result["origin_zone"].to_list()) == set(ZONES)
    assert (result["mean_count"] > 0).all()


def test_fit_od_model_recovers_ground_truth_distribution(synthetic_df, ground_truth):
    _, gt_od_model, *_ = ground_truth
    fitted = fit_od_model(synthetic_df, time_bin_minutes=60)

    for origin in ZONES:
        truth = gt_od_model.dest_probs[(origin, 9)]
        got = fitted.dest_probs.get((origin, 9))
        assert got is not None
        for dest in ZONES:
            assert got[dest] == pytest.approx(truth[dest], abs=0.1)


def test_fit_travel_time_model_recovers_ground_truth_params(synthetic_df, ground_truth):
    *_, gt_tt_model, _ = ground_truth
    fitted = fit_travel_time_model(synthetic_df)

    for o in ZONES:
        for d in ZONES:
            if o == d:
                continue
            truth_mu, truth_sigma = gt_tt_model.params[(o, d, 9)]
            got_mu, got_sigma = fitted.params.get((o, d, 9), fitted.fallback_params)
            assert got_mu == pytest.approx(truth_mu, abs=0.15)
            assert got_sigma == pytest.approx(truth_sigma, abs=0.15)


def test_fit_fare_model_recovers_linear_coefficients(synthetic_df):
    # generate_synthetic_trips's ground truth (see sources/synthetic.py):
    # FARE_BASE=10, FARE_PER_KM=18, FARE_PER_MINUTE=0.5. Recovery is noisier
    # than a naive regression tolerance would suggest because distance is
    # now *derived* from duration (through Bengaluru's real average speed,
    # not drawn independently) — the two regressors are correlated
    # (r~0.9), which is realistic (real trip distance and duration are
    # correlated too) but makes fare_per_km/fare_per_minute harder to
    # separate cleanly than the old fully-independent-distance version.
    fitted = fit_fare_model(synthetic_df)
    assert fitted.intercept == pytest.approx(10.0, abs=3.0)
    assert fitted.distance_coef == pytest.approx(18.0, abs=4.0)
    assert fitted.duration_coef == pytest.approx(0.5, abs=0.2)


def test_calibrate_fleet_size_recovers_a_known_monotone_relationship():
    # Fake simulate_fn: wait time strictly decreases with fleet size (no
    # plateau, so every candidate has a distinct expected median/p90) — this
    # test is about the search procedure recovering a unique optimum, not
    # about the real simulator.
    rng = np.random.default_rng(0)
    true_best = 25

    def simulate_fn(fleet_size: int) -> np.ndarray:
        base = 1000.0 / fleet_size
        return base + rng.normal(0, 5, size=500)

    observed = simulate_fn(true_best)
    result = calibrate_fleet_size(
        observed, simulate_fn, candidate_fleet_sizes=[5, 10, 15, 20, 25, 30, 35]
    )
    assert result.fleet_size == true_best


def test_ks_distance_is_zero_for_identical_samples():
    sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ks_distance(sample, sample) == pytest.approx(0.0)


def test_ks_distance_is_positive_for_different_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(5, 1, size=500)
    assert ks_distance(a, b) > 0.5
