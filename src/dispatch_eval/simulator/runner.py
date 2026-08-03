"""Compose fitted input models + a policy into a single simulate() call.

This is the seam `calibration.fleet_size` needs: something callable
repeatedly with a candidate fleet size that returns simulated outcomes,
without the caller needing to know anything about the engine's internals.

Generates the request `Scenario` fresh on every call, from whatever `rng` is
passed in — so calling this twice with two different policies but a freshly
seeded `rng` of the same seed reproduces the identical scenario for both
(see `dispatch_eval.scenario`), which is exactly what a fair comparison
needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.scenario import generate_scenario
from dispatch_eval.simulator.engine import SimulationEngine, SimulationResult
from dispatch_eval.simulator.entities import Vehicle


@dataclass
class StudyConfig:
    zones: list[str]
    day_type: str
    horizon_seconds: float
    dispatch_interval_seconds: float = 5.0
    od_bin_minutes: float = 60.0


def run_simulation(
    fleet_size: int,
    arrival_model: NHPPArrivalModel,
    od_model: ODModel,
    travel_time_model: TravelTimeModel,
    abandonment_model: AbandonmentModel,
    policy: DispatchPolicy,
    config: StudyConfig,
    rng: np.random.Generator,
) -> SimulationResult:
    scenario = generate_scenario(
        config.zones,
        config.day_type,
        arrival_model,
        od_model,
        abandonment_model,
        config.horizon_seconds,
        rng,
        config.od_bin_minutes,
    )
    vehicles = [
        Vehicle(vehicle_id=f"veh-{i}", zone=config.zones[i % len(config.zones)])
        for i in range(fleet_size)
    ]
    engine = SimulationEngine(
        vehicles=vehicles,
        scenario=scenario,
        travel_time_model=travel_time_model,
        policy=policy,
        zones=config.zones,
        horizon_seconds=config.horizon_seconds,
        rng=rng,
        dispatch_interval_seconds=config.dispatch_interval_seconds,
    )
    return engine.run()
