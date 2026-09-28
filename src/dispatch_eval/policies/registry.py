"""The dispatch policy ladder: readable names, construction and tuning.

Each rung adds one mechanism to the one below it:

    greedy               B0  nearest idle vehicle, one request at a time
    batched              B1  batched optimal matching (Hungarian) with a pickup radius
    value_aware          B2  batched matching + value of the drop-off zone
    fluid_rebalance      B3  value_aware + move idle cars toward demand share
    lookahead_rebalance  B4  value_aware + move idle cars toward sampled demand

(B5, the offline clairvoyant bound, is not a deployable policy; see
`clairvoyant.py`.) The B-codes are the ids used by the original synthetic
study's outputs; `LEGACY_IDS` maps them to the names here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from dispatch_eval.calibration.tuning import ParameterSpec, TuningResult, random_search
from dispatch_eval.calibration.value_function import ValueFunction
from dispatch_eval.models import NHPPArrivalModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.policies.batched_hungarian import BatchedHungarianPolicy
from dispatch_eval.policies.fluid_zone_balancing import FluidZoneBalancingPolicy
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.policies.sampling_lookahead import SamplingLookaheadPolicy
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy

POLICY_LABELS = {
    "greedy": "Greedy nearest-vehicle",
    "batched": "Batched matching",
    "value_aware": "Value-aware matching",
    "fluid_rebalance": "Value-aware + fluid rebalancing",
    "lookahead_rebalance": "Value-aware + lookahead rebalancing",
}

LEGACY_IDS = {
    "B0_nearest_idle": "greedy",
    "B1_batched_hungarian": "batched",
    "B2_value_corrected": "value_aware",
    "B3_fluid_balancing": "fluid_rebalance",
    "B4_sampling_lookahead": "lookahead_rebalance",
}


@dataclass
class LadderParams:
    batched_radius_seconds: float
    value_radius_seconds: float
    value_weight: float
    lookahead_seconds: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def build_ladder(
    params: LadderParams,
    value_function: ValueFunction,
    arrival_model: NHPPArrivalModel,
    day_type: str = "all",
) -> dict[str, DispatchPolicy]:
    value_aware = ValueCorrectedHungarianPolicy(
        value_function, params.value_weight, params.value_radius_seconds
    )
    return {
        "greedy": NearestIdlePolicy(),
        "batched": BatchedHungarianPolicy(params.batched_radius_seconds),
        "value_aware": value_aware,
        "fluid_rebalance": FluidZoneBalancingPolicy(value_aware, arrival_model, day_type),
        "lookahead_rebalance": SamplingLookaheadPolicy(
            value_aware, arrival_model, day_type, params.lookahead_seconds
        ),
    }


def tune_ladder(
    evaluate: Callable[[DispatchPolicy], float],
    value_function: ValueFunction,
    arrival_model: NHPPArrivalModel,
    n_evaluations: int,
    seed: int,
    radius_range: tuple[float, float] = (120.0, 2400.0),
    lookahead_range: tuple[float, float] = (60.0, 900.0),
    day_type: str = "all",
) -> tuple[LadderParams, dict[str, TuningResult]]:
    """Tune every tunable rung with the same random-search budget and seed
    (the parity condition: a ranking must not reflect a better search).

    `evaluate(policy)` returns the score to minimize. Rungs are tuned in
    order because rebalancing wraps the tuned value-aware dispatcher.
    """
    radius = ParameterSpec("radius", *radius_range, log_scale=True)

    def search(make_policy, specs):
        return random_search(lambda p: evaluate(make_policy(p)), specs, n_evaluations, seed)

    batched = search(lambda p: BatchedHungarianPolicy(p["radius"]), [radius])
    value = search(
        lambda p: ValueCorrectedHungarianPolicy(value_function, p["value_weight"], p["radius"]),
        [radius, ParameterSpec("value_weight", 0.0, 3.0)],
    )
    value_aware = ValueCorrectedHungarianPolicy(
        value_function, value.best_params["value_weight"], value.best_params["radius"]
    )
    lookahead = search(
        lambda p: SamplingLookaheadPolicy(value_aware, arrival_model, day_type, p["lookahead"]),
        [ParameterSpec("lookahead", *lookahead_range)],
    )
    params = LadderParams(
        batched_radius_seconds=batched.best_params["radius"],
        value_radius_seconds=value.best_params["radius"],
        value_weight=value.best_params["value_weight"],
        lookahead_seconds=lookahead.best_params["lookahead"],
    )
    return params, {"batched": batched, "value_aware": value, "lookahead_rebalance": lookahead}
