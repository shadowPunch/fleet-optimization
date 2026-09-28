from __future__ import annotations

import numpy as np
import pytest

from dispatch_eval.calibration.value_function import ValueFunction
from dispatch_eval.models import NHPPArrivalModel
from dispatch_eval.policies.registry import POLICY_LABELS, LadderParams, build_ladder, tune_ladder
from dispatch_eval.ranking_flip import RankingFlipResult
from dispatch_eval.studies.policy_study import long_table, summarize

VALUE_FUNCTION = ValueFunction(values={}, bin_minutes=15, n_bins=24)
ARRIVALS = NHPPArrivalModel(rates={})


def test_ladder_has_every_labelled_policy():
    ladder = build_ladder(LadderParams(600.0, 600.0, 1.0, 300.0), VALUE_FUNCTION, ARRIVALS)
    assert list(ladder) == list(POLICY_LABELS)


def test_tune_ladder_gives_every_tunable_policy_the_same_budget():
    calls = []

    def evaluate(policy):
        calls.append(type(policy).__name__)
        return float(len(calls))

    params, tuning = tune_ladder(evaluate, VALUE_FUNCTION, ARRIVALS, n_evaluations=4, seed=0)
    assert {name: len(t.history) for name, t in tuning.items()} == {
        "batched": 4, "value_aware": 4, "lookahead_rebalance": 4,
    }
    assert len(calls) == 12
    assert 120.0 <= params.batched_radius_seconds <= 2400.0


def _result() -> RankingFlipResult:
    rng = np.random.default_rng(0)
    greedy = 200 + rng.normal(0, 5, (50, 3))
    better = greedy - 30 + rng.normal(0, 1, (50, 3))  # clearly better, paired
    served = {"greedy": np.full((50, 3), 0.9), "batched": np.full((50, 3), 0.92)}
    return RankingFlipResult(
        policy_names=["greedy", "batched"],
        metric_by_policy={"greedy": greedy, "batched": better},
        rankings=np.tile([1, 0], (50, 1)),
        nominal_ranking=["batched", "greedy"],
        secondary_metrics={"fraction_served": served},
    )


def test_summarize_reports_paired_gains_and_ranking_stability():
    summary = summarize(_result())
    by_name = {p["policy"]: p for p in summary["policies"]}

    assert summary["best_policy"] == "batched"
    assert summary["indifference_set"] == ["batched"]
    assert summary["fraction_draws_matching_nominal"] == 1.0
    assert by_name["batched"]["probability_ranked_first"] == 1.0
    assert by_name["batched"]["gain_vs_reference_seconds"] == pytest.approx(30.0, abs=1.0)
    lo, hi = by_name["batched"]["gain_vs_reference_ci"]
    assert 0 < lo < hi
    assert by_name["greedy"]["gain_vs_reference_seconds"] == 0.0
    assert by_name["batched"]["fraction_served"] == pytest.approx(0.92)


def test_long_table_has_one_row_per_cell():
    table = long_table(_result(), "calibrated")
    assert table.height == 2 * 50 * 3
    assert set(table.columns) >= {"regime", "policy", "bootstrap_draw", "replication",
                                  "mean_wait_seconds", "fraction_served"}
