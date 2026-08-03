"""Generic fixed-budget parameter search for policy tuning.

The project plan's parity condition for P2: every policy in the baseline
ladder gets an identical tuning budget (same number of evaluations, same
seeds), so that "policy X beats policy Y" can't secretly mean "policy X got
a better search." This module is deliberately a plain random-search harness,
not a specific BO library — the parity requirement is about the evaluation
budget and protocol being identical across policies, not about which
acquisition function is used. Swapping in a smarter sampler (TPE, GP-based,
e.g. via optuna or scikit-optimize) later is a drop-in change behind the same
`evaluate_fn(params) -> score` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterSpec:
    name: str
    low: float
    high: float
    log_scale: bool = False


@dataclass
class TuningResult:
    best_params: dict[str, float]
    best_score: float
    history: list[tuple[dict[str, float], float]]


def random_search(
    evaluate_fn: Callable[[dict[str, float]], float],
    param_specs: list[ParameterSpec],
    n_evaluations: int,
    seed: int,
    minimize: bool = True,
) -> TuningResult:
    """Fixed-budget random search over `param_specs`.

    `evaluate_fn` is called exactly `n_evaluations` times; how it turns
    `params` into a score (e.g. running the simulator and reading off mean
    wait time) is entirely the caller's concern — this function only owns
    the sampling and the budget.
    """
    rng = np.random.default_rng(seed)
    history: list[tuple[dict[str, float], float]] = []

    for _ in range(n_evaluations):
        params: dict[str, float] = {}
        for spec in param_specs:
            if spec.log_scale:
                params[spec.name] = float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
            else:
                params[spec.name] = float(rng.uniform(spec.low, spec.high))
        history.append((params, evaluate_fn(params)))

    sign = 1.0 if minimize else -1.0
    best_params, best_score = min(history, key=lambda h: sign * h[1])
    return TuningResult(best_params=best_params, best_score=best_score, history=history)
