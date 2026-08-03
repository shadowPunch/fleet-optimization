from __future__ import annotations

import numpy as np

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.models import NHPPArrivalModel, ODModel, TravelTimeModel

ZONES = ["busy", "quiet"]


def _stationary_models(rates: dict[str, float], travel_seconds: float = 300.0):
    n_bins_per_day = 96
    arrival_rates = {(z, "all", b): rates[z] for z in ZONES for b in range(n_bins_per_day)}
    arrival_model = NHPPArrivalModel(rates=arrival_rates, bin_minutes=15)

    dest_probs = {(z, h): {zz: 1.0 / len(ZONES) for zz in ZONES} for z in ZONES for h in range(24)}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs={z: 1.0 / len(ZONES) for z in ZONES})

    params = {
        (o, d, h): (float(np.log(travel_seconds)), 0.0)
        for o in ZONES
        for d in ZONES
        for h in range(24)
    }
    tt_model = TravelTimeModel(params=params, fallback_params=(float(np.log(travel_seconds)), 0.0))
    return arrival_model, od_model, tt_model


def test_terminal_boundary_is_zero():
    arrival_model, od_model, tt_model = _stationary_models({"busy": 0.5, "quiet": 0.5})
    vf = compute_value_function(
        ZONES, arrival_model, od_model, tt_model, day_type="all", horizon_seconds=3600.0
    )
    for zone in ZONES:
        assert vf.cost_to_go(zone, vf.n_bins) == 0.0
        assert vf.cost_to_go(zone, vf.n_bins + 5) == 0.0  # clipped, doesn't KeyError


def test_busier_zone_has_lower_cost_to_go():
    # "busy" gets far more requests than "quiet"; a vehicle idling in "busy"
    # should have a much shorter expected future pickup burden.
    arrival_model, od_model, tt_model = _stationary_models({"busy": 2.0, "quiet": 0.02})
    vf = compute_value_function(
        ZONES, arrival_model, od_model, tt_model, day_type="all", horizon_seconds=4 * 3600.0
    )
    assert vf.cost_to_go("busy", 0) < vf.cost_to_go("quiet", 0)


def test_cost_to_go_is_non_increasing_as_horizon_end_approaches_under_stationary_rates():
    arrival_model, od_model, tt_model = _stationary_models({"busy": 0.5, "quiet": 0.5})
    vf = compute_value_function(
        ZONES, arrival_model, od_model, tt_model, day_type="all", horizon_seconds=4 * 3600.0
    )
    checkpoints = [0, vf.n_bins // 4, vf.n_bins // 2, vf.n_bins - 1, vf.n_bins]
    values_over_time = [vf.cost_to_go("busy", t) for t in checkpoints]
    assert values_over_time == sorted(values_over_time, reverse=True)
    assert all(v >= 0 for v in values_over_time)
