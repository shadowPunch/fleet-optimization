from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.ranking_flip import (
    RankingFlipResult,
    bootstrap_resample_trips,
    fit_all_models,
    indifference_set,
    minimum_detectable_effect_curve,
    run_ranking_flip_experiment,
    subset_trips_by_days,
    validate_mde_scaling,
    variance_decomposition,
)
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C"]


def _fake_result(policy_names: list[str], metric_arrays: list[np.ndarray]) -> RankingFlipResult:
    n_bootstrap = metric_arrays[0].shape[0]
    return RankingFlipResult(
        policy_names=policy_names,
        metric_by_policy=dict(zip(policy_names, metric_arrays, strict=True)),
        rankings=np.zeros(
            (n_bootstrap, len(policy_names)), dtype=int
        ),  # unused by indifference_set
        nominal_ranking=policy_names,
    )


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


def test_compute_clairvoyant_false_by_default_leaves_field_none(trips_df):
    policies = {"B0": NearestIdlePolicy(), "B1": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, fleet_size=20, abandonment_model=abandonment_model,
        config=config, n_bootstrap=3, n_replications=2, seed=1,
    )
    assert result.clairvoyant_metric_by_draw is None


def test_compute_clairvoyant_true_fills_a_plausible_bound(trips_df):
    policies = {"B0": NearestIdlePolicy(), "B1": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, fleet_size=20, abandonment_model=abandonment_model,
        config=config, n_bootstrap=3, n_replications=2, seed=1, compute_clairvoyant=True,
    )

    assert result.clairvoyant_metric_by_draw is not None
    assert result.clairvoyant_metric_by_draw.shape == (3, 2)
    assert np.all(np.isfinite(result.clairvoyant_metric_by_draw))
    assert np.all(result.clairvoyant_metric_by_draw >= 0.0)
    # a true (or near-true, given B5's documented pessimistic-pinning
    # approximation) upper bound should on average not exceed a real
    # online policy's own mean wait
    assert result.clairvoyant_metric_by_draw.mean() <= result.metric_by_policy["B0"].mean()


def test_fraction_of_gap_closed_requires_clairvoyant_computed(trips_df):
    policies = {"B0": NearestIdlePolicy(), "B1": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = run_ranking_flip_experiment(
        trips_df, ZONES, policies, fleet_size=20, abandonment_model=abandonment_model,
        config=config, n_bootstrap=2, n_replications=2, seed=1,
    )
    with pytest.raises(ValueError, match="compute_clairvoyant"):
        result.fraction_of_gap_closed("B0")


def test_fraction_of_gap_closed_arithmetic_on_a_fake_result():
    baseline = np.array([[100.0, 100.0], [100.0, 100.0]])
    matches_clairvoyant = np.array([[50.0, 50.0], [50.0, 50.0]])
    same_as_baseline = baseline.copy()
    clairvoyant = np.array([[50.0, 50.0], [50.0, 50.0]])

    result = RankingFlipResult(
        policy_names=["baseline", "matches_clairvoyant", "same_as_baseline"],
        metric_by_policy={
            "baseline": baseline,
            "matches_clairvoyant": matches_clairvoyant,
            "same_as_baseline": same_as_baseline,
        },
        rankings=np.zeros((2, 3), dtype=int),
        nominal_ranking=["matches_clairvoyant", "same_as_baseline", "baseline"],
        clairvoyant_metric_by_draw=clairvoyant,
    )

    gap_closed = result.fraction_of_gap_closed("baseline")
    assert np.allclose(gap_closed["baseline"], 0.0)
    assert np.allclose(gap_closed["matches_clairvoyant"], 1.0)
    assert np.allclose(gap_closed["same_as_baseline"], 0.0)


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


def test_indifference_set_excludes_a_clearly_worse_policy():
    rng = np.random.default_rng(0)
    n_bootstrap, n_replications = 100, 5
    best = rng.normal(10.0, 1.0, size=(n_bootstrap, n_replications))
    worse = rng.normal(50.0, 1.0, size=(n_bootstrap, n_replications))
    result = _fake_result(["best", "worse"], [best, worse])

    assert indifference_set(result, alpha=0.05) == {"best"}


def test_indifference_set_includes_statistically_indistinguishable_policies():
    rng = np.random.default_rng(0)
    n_bootstrap, n_replications = 100, 5
    a = rng.normal(10.0, 1.0, size=(n_bootstrap, n_replications))
    b = a + rng.normal(0.0, 0.01, size=(n_bootstrap, n_replications))  # noise-level difference
    result = _fake_result(["a", "b"], [a, b])

    assert indifference_set(result, alpha=0.05) == {"a", "b"}


def test_indifference_set_always_contains_the_best_policy():
    rng = np.random.default_rng(1)
    metrics = [rng.normal(mean, 1.0, size=(50, 4)) for mean in (10.0, 20.0, 30.0)]
    result = _fake_result(["a", "b", "c"], metrics)

    assert "a" in indifference_set(result, alpha=0.05)


def _trips_df_spanning_days(n_days: int) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(days=d, hours=1) for d in range(n_days)]
    return pl.DataFrame({"request_ts": timestamps})


def test_subset_trips_by_days_keeps_only_the_first_n_calendar_days():
    df = _trips_df_spanning_days(10)
    subset = subset_trips_by_days(df, n_days=3, request_ts_col="request_ts")
    assert subset["request_ts"].dt.date().n_unique() == 3
    assert subset["request_ts"].dt.date().max() == df["request_ts"].dt.date()[2]


def test_mde_curve_decreases_as_candidate_days_increase():
    trips_df = _trips_df_spanning_days(10)
    variance_result = {"within_theta_variance": 4.0, "across_theta_variance": 16.0}
    curve = minimum_detectable_effect_curve(
        trips_df, variance_result, candidate_days=[5, 10, 20, 40], confidence=0.95
    )
    values = [curve[n] for n in (5, 10, 20, 40)]
    assert values == sorted(values, reverse=True)  # strictly more data -> smaller MDE
    assert all(v > 0 for v in values)


def test_mde_curve_at_fitted_days_matches_the_unscaled_standard_error():
    trips_df = _trips_df_spanning_days(10)
    variance_result = {"within_theta_variance": 4.0, "across_theta_variance": 16.0}
    curve = minimum_detectable_effect_curve(
        trips_df, variance_result, candidate_days=[10], confidence=0.95
    )
    from scipy.stats import norm

    expected = norm.ppf(0.95) * np.sqrt(4.0 + 16.0)  # n == n_days_fitted: no rescaling
    assert curve[10] == pytest.approx(expected)


def test_validate_mde_scaling_anchor_ratio_is_exactly_one_by_construction():
    trips_df = generate_synthetic_trips(
        n_days=6, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=25, seed=4
    )
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    result = validate_mde_scaling(
        trips_df,
        ZONES,
        policy_a=NearestIdlePolicy(),
        policy_b=BatchedHungarianPolicy(),
        fleet_size=15,
        abandonment_model=abandonment_model,
        config=config,
        candidate_n_days=[2, 4, 6],
        n_bootstrap=3,
        n_replications=2,
        seed=21,
    )

    assert set(result.keys()) == {2, 4, 6}
    anchor = result[6]  # largest n tested
    assert anchor["ratio_measured_to_predicted"] == pytest.approx(1.0)
    assert anchor["measured_across_theta_variance"] == pytest.approx(anchor["predicted_by_1_over_n"])

    # predicted_by_1_over_n must strictly decrease as n grows (anchor_n/n shrinks)
    predicted_values = [result[n]["predicted_by_1_over_n"] for n in (2, 4, 6)]
    assert predicted_values == sorted(predicted_values, reverse=True)
    for n in (2, 4, 6):
        assert result[n]["measured_across_theta_variance"] >= 0.0
