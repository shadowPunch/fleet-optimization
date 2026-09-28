"""Pre-generate the exogenous request scenario for a simulation run.

Common random numbers requires the realized set of requests (who arrives
when, wanting what, with how much patience) to be identical across every
policy being compared for the same (bootstrap draw, replication) —
otherwise "policy X did better" might just mean "policy X's own dispatch
decisions happened to consume the shared rng stream differently, shifting
which random draws later requests got." That was a real bug in this
codebase: `SimulationEngine` used to draw arrivals, destinations, and
patience reactively during the event loop, interleaved with whatever
travel-time samples a policy's dispatch decisions also drew from the same
stream — so two policies run with the "same seed" did not actually see the
same requests.

The fix: generate the whole scenario once, upfront, in an order that depends
only on the input models and zone list — never on anything a policy does —
then hand the identical `Scenario` to every policy's `SimulationEngine` run.

Travel-time *realizations* are deliberately not part of this scenario: which
(vehicle, request) pairs actually occur depends on the policy, so there's no
way to pre-fix "the" travel time for a trip that might not even happen under
some policy. Those are still sampled reactively from the engine's own rng
during the event loop, continuing from wherever scenario generation left
off — which matches how CRN is normally applied to this class of problem:
fix the arrival process, let realized trips vary by policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dispatch_eval.models import AbandonmentModel, BoardingModel, NHPPArrivalModel, ODModel
from dispatch_eval.simulator.entities import Request


@dataclass
class Scenario:
    """A fixed, already-realized set of requests for one simulation horizon.

    Treat this as read-only and reusable: the same `Scenario` is meant to be
    handed to multiple `SimulationEngine` runs (one per policy) without
    being mutated by any of them.
    """

    requests: list[Request]


def generate_scenario(
    zones: list[str],
    day_type: str,
    arrival_model: NHPPArrivalModel,
    od_model: ODModel,
    abandonment_model: AbandonmentModel,
    horizon_seconds: float,
    rng: np.random.Generator,
    od_bin_minutes: float = 60.0,
    boarding_model: BoardingModel | None = None,
) -> Scenario:
    """Generate every request that will arrive over the horizon, upfront.

    Zones are processed in a fixed (sorted) order, and each zone's full
    arrival stream — including every arrival's destination and patience —
    is drawn before moving to the next zone, so the sequence of draws from
    `rng` depends only on `zones` and the fitted models, never on simulation
    or policy behaviour. Boarding time is a rider attribute like patience,
    so it is fixed here too; with no `boarding_model` it is zero and no
    extra draws are made (older scenarios reproduce exactly).
    """
    horizon_minutes = horizon_seconds / 60.0
    requests: list[Request] = []
    counter = 0

    for zone in sorted(zones):
        arrival_minutes = arrival_model.generate_arrival_minutes(
            zone, day_type, 0.0, horizon_minutes, rng
        )
        for minute in arrival_minutes:
            request_time = minute * 60.0
            od_time_bin = int(minute // od_bin_minutes)
            dest_zone = od_model.sample_destination(zone, od_time_bin, rng)
            patience = abandonment_model.sample_patience(rng)
            boarding = boarding_model.sample(rng) if boarding_model is not None else 0.0
            requests.append(
                Request(
                    request_id=f"req-{counter}",
                    origin_zone=zone,
                    dest_zone=dest_zone,
                    request_time=request_time,
                    abandon_at=request_time + patience,
                    boarding_seconds=boarding,
                )
            )
            counter += 1

    requests.sort(key=lambda r: r.request_time)
    return Scenario(requests=requests)
