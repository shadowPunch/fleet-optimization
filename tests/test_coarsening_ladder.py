from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from dispatch_eval.coarsening_ladder import (
    CoarseningRung,
    apply_coarsening,
    coarsen_destroy_od_correlation,
    coarsen_fare_distance_to_od_mean,
    ranking_shift_summary,
    run_coarsening_ladder,
)
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

ZONES = ["A", "B", "C"]


def _tiny_trip_df() -> pl.DataFrame:
    base_ts = datetime(2026, 1, 1, 8, 0, 0)
    return pl.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3", "t4"],
            "origin_zone": ["A", "A", "B", "B"],
            "dest_zone": ["B", "B", "A", "A"],
            "request_ts": [base_ts] * 4,
            "pickup_ts": [base_ts] * 4,
            "dropoff_ts": [base_ts] * 4,
            "trip_distance_km": [10.0, 20.0, 5.0, None],
            "fare": [100.0, 200.0, 50.0, None],
            "driver_pay": [100.0, 200.0, 50.0, None],
            "status": ["completed", "completed", "completed", "cancelled_customer"],
            "vehicle_type": ["auto"] * 4,
            "source": ["synthetic"] * 4,
        }
    )


# --- coarsen_fare_distance_to_od_mean -----------------------------------------


def test_coarsen_fare_distance_removes_within_pair_variance():
    df = _tiny_trip_df()
    out = coarsen_fare_distance_to_od_mean(df)

    ab_rows = out.filter((pl.col("origin_zone") == "A") & (pl.col("dest_zone") == "B"))
    assert ab_rows["fare"].n_unique() == 1
    assert ab_rows["fare"][0] == pytest.approx(150.0)  # mean of 100, 200
    assert ab_rows["trip_distance_km"][0] == pytest.approx(15.0)  # mean of 10, 20


def test_coarsen_fare_distance_preserves_nulls_for_incomplete_trips():
    df = _tiny_trip_df()
    out = coarsen_fare_distance_to_od_mean(df)
    cancelled = out.filter(pl.col("status") == "cancelled_customer")
    assert cancelled["fare"][0] is None
    assert cancelled["trip_distance_km"][0] is None


# --- coarsen_destroy_od_correlation -------------------------------------------


def test_destroy_od_correlation_preserves_marginal_dest_distribution():
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {"origin_zone": ["A"] * 80 + ["B"] * 20, "dest_zone": ["X"] * 30 + ["Y"] * 70}
    )
    out = coarsen_destroy_od_correlation(df, rng)
    assert sorted(out["dest_zone"].to_list()) == sorted(df["dest_zone"].to_list())


def test_destroy_od_correlation_breaks_perfect_origin_dest_pairing():
    rng = np.random.default_rng(0)
    # perfectly deterministic mapping: every X origin -> Y dest, every P -> Q
    df = pl.DataFrame(
        {"origin_zone": ["X"] * 50 + ["P"] * 50, "dest_zone": ["Y"] * 50 + ["Q"] * 50}
    )
    out = coarsen_destroy_od_correlation(df, rng)

    x_dests = out.filter(pl.col("origin_zone") == "X")["dest_zone"]
    assert not (x_dests == "Y").all()  # perfect pairing should not survive a real shuffle


# --- apply_coarsening ----------------------------------------------------------


def test_apply_coarsening_is_identity_when_no_flags_set():
    df = _tiny_trip_df()
    rung = CoarseningRung(
        "none", bin_minutes=15, od_time_bin_minutes=60,
        destroy_od_correlation=False, coarsen_fare_distance=False,
    )
    out = apply_coarsening(df, rung, np.random.default_rng(0))
    assert out["fare"].to_list() == df["fare"].to_list()
    assert out["dest_zone"].to_list() == df["dest_zone"].to_list()


def test_apply_coarsening_with_both_flags_shuffles_then_recomputes_means():
    df = _tiny_trip_df()
    rung = CoarseningRung(
        "both", bin_minutes=1440, od_time_bin_minutes=1440,
        destroy_od_correlation=True, coarsen_fare_distance=True,
    )
    out = apply_coarsening(df, rung, np.random.default_rng(1))
    # dest_zone marginal preserved even after both transforms
    assert sorted(out["dest_zone"].to_list()) == sorted(df["dest_zone"].to_list())


# --- run_coarsening_ladder / ranking_shift_summary (integration) ---------------


@pytest.fixture(scope="module")
def trips_df():
    return generate_synthetic_trips(
        n_days=3, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=25, seed=2
    )


def test_run_coarsening_ladder_end_to_end(trips_df):
    policies = {"B0": NearestIdlePolicy(), "B1": BatchedHungarianPolicy()}
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=6 * 3600.0)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    rungs = [
        CoarseningRung("fine", bin_minutes=1, od_time_bin_minutes=15,
                        destroy_od_correlation=False, coarsen_fare_distance=False),
        CoarseningRung("coarse", bin_minutes=1440, od_time_bin_minutes=1440,
                        destroy_od_correlation=True, coarsen_fare_distance=True),
    ]

    results = run_coarsening_ladder(
        trips_df, ZONES, policies, fleet_size=15, abandonment_model=abandonment_model,
        config=config, n_bootstrap=3, n_replications=2, seed=9, rungs=rungs,
    )

    assert set(results.keys()) == {"fine", "coarse"}
    for rung_result in results.values():
        assert set(rung_result.policy_names) == {"B0", "B1"}
        for name in rung_result.policy_names:
            assert rung_result.metric_by_policy[name].shape == (3, 2)

    summary = ranking_shift_summary(results, baseline_rung="fine")
    assert set(summary.keys()) == {"fine", "coarse"}
    assert summary["fine"]["kendall_tau_to_baseline"] == pytest.approx(1.0)
    for rung_summary in summary.values():
        probs = rung_summary["probability_ranked_first"]
        assert sum(probs.values()) == pytest.approx(1.0)
