from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import (
    bootstrap_resample_trips,
    fit_all_models,
    run_ranking_flip_experiment,
    variance_decomposition,
)
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C"]


@pytest.fixture(scope="module")
def trips_df():
    return generate_synthetic_trips(
        n_days=3, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=40, seed=1
    )


def test_bootstrap_resample_preserves_size_and_draws_from_original(trips_df):
    rng = np.random.default_rng(0)
    resampled = bootstrap_resample_trips(trips_df, rng)
    assert resampled.height == trips_df.height
    original_ids = set(trips_df["trip_id"].to_list())
    assert set(resampled["trip_id"].to_list()) <= original_ids
    # with replacement: essentially never an exact permutation of the original
    assert resampled["trip_id"].to_list() != trips_df["trip_id"].to_list()


def test_bootstrap_resample_is_deterministic_given_rng_state(trips_df):
    a = bootstrap_resample_trips(trips_df, np.random.default_rng(5))
    b = bootstrap_resample_trips(trips_df, np.random.default_rng(5))
    assert a["trip_id"].to_list() == b["trip_id"].to_list()


def test_fit_all_models_returns_all_four_models(trips_df):
    fitted = fit_all_models(trips_df)
    assert len(fitted.arrival.rates) > 0
    assert len(fitted.od.dest_probs) > 0
    assert len(fitted.travel_time.params) > 0
    assert fitted.fare.distance_coef > 0


def test_identical_policies_get_exactly_identical_metrics_every_draw(trips_df):
    # The core CRN claim: two policies that behave identically must see the
    # identical realized scenario at every (b, r), so their metrics should
    # match exactly, not just on average.
    policies = {"copy_a": NearestIdlePolicy(), "copy_b": NearestIdlePolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = run_ranking_flip_experiment(
        trips_df=trips_df,
        zones=ZONES,
        policies=policies,
        fleet_size=20,
        abandonment_model=abandonment_model,
        config=config,
        n_bootstrap=4,
        n_replications=2,
        seed=7,
    )

    assert np.array_equal(result.metric_by_policy["copy_a"], result.metric_by_policy["copy_b"])


def test_ranking_flip_experiment_shapes_and_probabilities(trips_df):
    policies = {"B0": NearestIdlePolicy(), "B1": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = run_ranking_flip_experiment(
        trips_df=trips_df,
        zones=ZONES,
        policies=policies,
        fleet_size=20,
        abandonment_model=abandonment_model,
        config=config,
        n_bootstrap=5,
        n_replications=3,
        seed=42,
    )

    assert set(result.policy_names) == {"B0", "B1"}
    for name in result.policy_names:
        assert result.metric_by_policy[name].shape == (5, 3)
    assert result.rankings.shape == (5, 2)
    assert set(result.nominal_ranking) == {"B0", "B1"}

    probs = result.probability_ranked_first()
    assert set(probs.keys()) == {"B0", "B1"}
    assert probs["B0"] + probs["B1"] == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())

    taus = result.kendall_tau_to_nominal()
    assert taus.shape == (5,)
    assert np.all((taus >= -1.0) & (taus <= 1.0))


def test_variance_decomposition_is_zero_for_identical_series():
    metric = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = variance_decomposition(metric, metric)
    assert result["within_theta_variance"] == pytest.approx(0.0)
    assert result["across_theta_variance"] == pytest.approx(0.0)
    assert result["total_variance"] == pytest.approx(0.0)


def test_variance_decomposition_recovers_known_within_and_across_structure():
    # Construct a difference series with a KNOWN within-draw variance (noise
    # added per replication) and a KNOWN across-draw variance (a fixed shift
    # per bootstrap draw), and check the decomposition recovers each piece.
    rng = np.random.default_rng(0)
    n_bootstrap, n_replications = 200, 20
    draw_shift = rng.normal(0, 3.0, size=n_bootstrap)  # across-theta signal, sd=3
    noise = rng.normal(0, 1.0, size=(n_bootstrap, n_replications))  # within-theta noise, sd=1
    diff = draw_shift[:, None] + noise

    metric_a = diff
    metric_b = np.zeros_like(diff)
    result = variance_decomposition(metric_a, metric_b)

    assert result["within_theta_variance"] == pytest.approx(1.0, rel=0.25)
    assert result["across_theta_variance"] == pytest.approx(9.0, rel=0.35)
    assert (
        result["input_uncertainty_ratio"] > 1.0
    )  # across-theta (9) should dominate within-theta (1)
