"""Compose fitted input models + a policy into a single simulate() call.

This is the seam `calibration.fleet_size` needs: something callable
repeatedly with a candidate fleet size that returns simulated outcomes,
without the caller needing to know anything about the engine's internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.base import DispatchPolicy
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
    vehicles = [
        Vehicle(vehicle_id=f"veh-{i}", zone=config.zones[i % len(config.zones)])
        for i in range(fleet_size)
    ]
    engine = SimulationEngine(
        vehicles=vehicles,
        arrival_model=arrival_model,
        od_model=od_model,
        travel_time_model=travel_time_model,
        abandonment_model=abandonment_model,
        policy=policy,
        zones=config.zones,
        day_type=config.day_type,
        horizon_seconds=config.horizon_seconds,
        rng=rng,
        dispatch_interval_seconds=config.dispatch_interval_seconds,
        od_bin_minutes=config.od_bin_minutes,
    )
    return engine.run()
