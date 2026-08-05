from __future__ import annotations

import numpy as np
import pytest

from dispatch_eval.models import NHPPArrivalModel

ZONES = ["A", "B", "C"]


def _high_rate_model(bin_minutes: int = 15, rate_per_minute: float = 5.0) -> NHPPArrivalModel:
    rates = {(z, "all", b): rate_per_minute for z in ZONES for b in range(96)}
    return NHPPArrivalModel(rates=rates, bin_minutes=bin_minutes)


# --- generate_arrival_minutes: no request should ever land outside [start, end) ------


@pytest.mark.parametrize("start_minute,end_minute", [
    (0.0, 60.0),
    (7.0, 52.0),       # start/end not aligned to any 15-min bin boundary
    (713.3, 800.1),    # arbitrary fractional boundaries, later in the day
    (0.0, 1440.0),     # a full day
])
def test_generate_arrival_minutes_never_produces_an_out_of_window_arrival(start_minute, end_minute):
    model = _high_rate_model()
    rng = np.random.default_rng(0)
    arrivals = model.generate_arrival_minutes("A", "all", start_minute, end_minute, rng)

    assert len(arrivals) > 0  # high rate over a real window: this check needs actual arrivals
    assert all(start_minute <= t < end_minute for t in arrivals)


def test_generate_arrival_minutes_is_sorted():
    model = _high_rate_model()
    rng = np.random.default_rng(1)
    arrivals = model.generate_arrival_minutes("A", "all", 0.0, 200.0, rng)
    assert arrivals == sorted(arrivals)


def test_generate_arrival_minutes_empty_window_returns_nothing():
    model = _high_rate_model()
    rng = np.random.default_rng(2)
    assert model.generate_arrival_minutes("A", "all", 100.0, 100.0, rng) == []


def test_generate_arrival_minutes_zero_rate_zone_returns_nothing():
    model = NHPPArrivalModel(rates={}, bin_minutes=15)  # no observed cells anywhere -> rate 0.0
    rng = np.random.default_rng(3)
    assert model.generate_arrival_minutes("A", "all", 0.0, 1440.0, rng) == []


def test_generate_arrival_minutes_respects_a_narrow_sub_bin_window():
    # window entirely inside one bin, not aligned to its start -- window_len
    # used for the Poisson rate must reflect the true (narrow) window, not
    # the full bin width, or this would over-generate arrivals.
    model = _high_rate_model(bin_minutes=15, rate_per_minute=100.0)  # very high, to make the check tight
    rng = np.random.default_rng(4)
    arrivals = model.generate_arrival_minutes("A", "all", 3.0, 5.0, rng)  # 2-minute window inside bin 0
    assert all(3.0 <= t < 5.0 for t in arrivals)
