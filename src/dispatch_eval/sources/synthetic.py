"""Synthetic trip-record generator.

Produces data matching `TRIP_RECORD_SCHEMA` from chosen ground-truth
parameters by actually running the P1 engine, so calibration routines have
something with a known answer to recover. This is the development/test
data source until real Delhi NCR / Bengaluru data is downloaded and adapted
(see `sources.delhi_ncr`, `sources.bengaluru`) — never treat its output as a
claim about a real city.

Abandoned requests are folded into "cancelled_customer" in the emitted
records, matching the same observability constraint real sources have (see
docs/observability_table.md): the point of this project is that the
abandonment hazard isn't identified from data like this, so the synthetic
test-bed shouldn't leak the answer either.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.schema import TRIP_RECORD_SCHEMA
from dispatch_eval.simulator.runner import StudyConfig, run_simulation


def default_ground_truth(
    zones: list[str],
    base_rate_per_minute: float = 0.5,
    bin_minutes: int = 15,
    mean_patience_seconds: float = 300.0,
    seed: int = 0,
) -> tuple[NHPPArrivalModel, ODModel, TravelTimeModel, AbandonmentModel]:
    """An arbitrary but plausible ground truth — not calibrated to any real city."""
    rng = np.random.default_rng(seed)
    n_bins = int(24 * 60 / bin_minutes)

    rates: dict[tuple[str, str, int], float] = {}
    for zone in zones:
        zone_scale = rng.uniform(0.5, 1.5)
        for b in range(n_bins):
            hour = (b * bin_minutes) // 60
            morning = np.exp(-((hour - 9) ** 2) / 8)
            evening = np.exp(-((hour - 18) ** 2) / 8)
            shape = 1.0 + 0.8 * morning + 1.2 * evening
            rates[(zone, "all", b)] = base_rate_per_minute * zone_scale * shape
    arrival_model = NHPPArrivalModel(rates=rates, bin_minutes=bin_minutes)

    dest_probs: dict[tuple[str, int], dict[str, float]] = {}
    for zone in zones:
        for hour_bin in range(24):
            weights = rng.dirichlet(np.ones(len(zones)) * 2.0)
            dest_probs[(zone, hour_bin)] = dict(zip(zones, weights.tolist()))
    fallback_probs = {z: 1.0 / len(zones) for z in zones}
    od_model = ODModel(dest_probs=dest_probs, fallback_probs=fallback_probs)

    params: dict[tuple[str, str, int], tuple[float, float]] = {}
    for o in zones:
        for d in zones:
            for hour in range(24):
                base_minutes = 5.0 if o == d else rng.uniform(8.0, 25.0)
                mu = float(np.log(base_minutes * 60.0))
                params[(o, d, hour)] = (mu, 0.35)
    travel_time_model = TravelTimeModel(params=params, fallback_params=(float(np.log(600.0)), 0.4))

    abandonment_model = AbandonmentModel(mean_patience_seconds=mean_patience_seconds)

    return arrival_model, od_model, travel_time_model, abandonment_model


def generate_synthetic_trips(
    n_days: int,
    zones: list[str],
    start_date: datetime,
    seed: int = 0,
    fleet_size: int = 40,
    fare_per_km: float = 15.0,
    fare_per_minute: float = 2.0,
    fare_base: float = 20.0,
    driver_pay_fraction: float = 0.75,
    distance_mean_km: float = 6.0,
    distance_sd_km: float = 3.0,
) -> pl.DataFrame:
    """Simulate `n_days` of trip-level data by running the P1 engine with B0."""
    rng = np.random.default_rng(seed)
    arrival_model, od_model, travel_time_model, abandonment_model = default_ground_truth(
        zones, seed=seed
    )
    policy = NearestIdlePolicy()

    rows: list[dict[str, object]] = []
    trip_counter = 0
    for day in range(n_days):
        day_start = start_date + timedelta(days=day)
        config = StudyConfig(zones=zones, day_type="all", horizon_seconds=24 * 3600.0)
        result = run_simulation(
            fleet_size=fleet_size,
            arrival_model=arrival_model,
            od_model=od_model,
            travel_time_model=travel_time_model,
            abandonment_model=abandonment_model,
            policy=policy,
            config=config,
            rng=rng,
        )

        for request in result.completed_requests:
            distance_km = float(max(0.5, rng.normal(distance_mean_km, distance_sd_km)))
            duration_minutes = (request.dropoff_time - request.pickup_time) / 60.0
            fare = fare_base + fare_per_km * distance_km + fare_per_minute * duration_minutes
            rows.append(
                {
                    "trip_id": f"day{day}-{trip_counter}",
                    "origin_zone": request.origin_zone,
                    "dest_zone": request.dest_zone,
                    "request_ts": day_start + timedelta(seconds=request.request_time),
                    "pickup_ts": day_start + timedelta(seconds=request.pickup_time),
                    "dropoff_ts": day_start + timedelta(seconds=request.dropoff_time),
                    "trip_distance_km": distance_km,
                    "fare": fare,
                    "driver_pay": fare * driver_pay_fraction,
                    "status": "completed",
                    "vehicle_type": "auto",
                    "source": "synthetic",
                }
            )
            trip_counter += 1

        for request in result.abandoned_requests:
            rows.append(
                {
                    "trip_id": f"day{day}-{trip_counter}",
                    "origin_zone": request.origin_zone,
                    "dest_zone": request.dest_zone,
                    "request_ts": day_start + timedelta(seconds=request.request_time),
                    "pickup_ts": None,
                    "dropoff_ts": None,
                    "trip_distance_km": None,
                    "fare": None,
                    "driver_pay": None,
                    "status": "cancelled_customer",
                    "vehicle_type": None,
                    "source": "synthetic",
                }
            )
            trip_counter += 1

    return pl.DataFrame(rows, schema=TRIP_RECORD_SCHEMA)
