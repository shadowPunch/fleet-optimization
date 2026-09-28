"""Synthetic trip-record generator.

Produces data matching `TRIP_RECORD_SCHEMA` from chosen ground-truth
parameters by actually running the P1 engine, so calibration routines have
something with a known answer to recover. This is the development/test
data source until real Delhi NCR / Bengaluru data is downloaded and adapted
(see `sources.delhi_ncr`, `sources.bengaluru`) — never treat its output as a
claim about a real city.

Abandoned requests are folded into "cancelled_customer" in the emitted
records, matching the same observability constraint real sources have (see
TECHNICAL_REPORT.md): the point of this project is that the
abandonment hazard isn't identified from data like this, so the synthetic
test-bed shouldn't leak the answer either.

Several defaults below are anchored to real, publicly documented Bengaluru
facts rather than picked arbitrarily — see `analysis/nyc_reference_comparison.py`
for the NYC shape-comparison exercise that prompted grounding them, and the
inline citations at each constant. The rule applied throughout: a
*distributional form* (e.g. right-skewed trip distances) generalizes across
cities and is fair to borrow from anywhere; a *magnitude or structural ratio*
(e.g. how heavily fare depends on distance vs. duration) does not, and is
only changed here where a real Bengaluru-specific number was found to anchor
it. Where no such anchor exists (e.g. how concentrated demand is across
zones), the old arbitrary default is left alone rather than quietly copying
another city's number — see the README.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dispatch_eval.models import AbandonmentModel, NHPPArrivalModel, ODModel, TravelTimeModel
from dispatch_eval.policies.nearest_idle import NearestIdlePolicy
from dispatch_eval.schema import TRIP_RECORD_SCHEMA
from dispatch_eval.simulator.runner import StudyConfig, run_simulation

# BBMP raised bar/club/licensed-hotel closing time to 1am in 2024 (standalone
# pubs historically stricter, ~11:30pm, with a 1am extension approved in
# 2025) — see the Business Standard / Deccan Herald coverage cited in the
# README. Bengaluru's nightlife-driven demand is real, just later-peaking and
# narrower than a US city's; it is not the same shape as NYC's, only present
# for the same underlying reason (a city with real nightlife has one).
NIGHTLIFE_PEAK_HOUR = 1.0
NIGHTLIFE_WIDTH_SQ = 4.0
NIGHTLIFE_WEIGHT = 0.6

# BBMP-regulated auto-rickshaw meter fare, effective Aug 2025: ₹36 flat for
# the first 2km, ₹18/km after that (Deccan Herald / CarDekho coverage, see
# README). Fit as a straight line through (2km, ₹36) and (10km, ₹180) gives
# slope=18, intercept≈0 — i.e. the *marginal* rate alone tracks the real
# meter well past the first 2km; FARE_BASE below is a small flag-fall
# standing in for the flat minimum this linear model can't represent exactly.
# The meter has essentially no continuous per-minute component (only a
# ₹10/15-min *waiting* charge while stopped, not a moving-time charge) —
# unlike NYC's TLC formula, which has a real per-minute term throughout the
# trip. FARE_PER_MINUTE is kept small and non-zero as a rough stand-in for
# that waiting charge averaged over a whole trip, not a NYC-style time fare.
FARE_BASE = 10.0
FARE_PER_KM = 18.0
FARE_PER_MINUTE = 0.5
# Meter rule: 1.5x between 10pm and 5am.
NIGHT_SURCHARGE_MULTIPLIER = 1.5
NIGHT_SURCHARGE_HOURS = frozenset([22, 23, 0, 1, 2, 3, 4])

# Namma Yatri markets itself as zero-commission — "100% goes to the driver"
# — unlike Uber/Ola's cut or NYC's regulated ~72-75% driver share (see the
# NYC comparison in analysis/nyc_reference_comparison.py, which found NYC's
# real median close to the *previous* default of 0.75 here — a coincidence
# worth not leaning on, since Namma Yatri's actual commission model is
# structurally different, not just numerically different).
DRIVER_PAY_FRACTION = 1.0

# TomTom Traffic Index 2024: Bengaluru's average speed ~17.6-21.9 km/h
# depending on measurement window (among the world's slowest); midpoint used
# here. Trip distance is derived from realized trip duration through this
# speed (with multiplicative lognormal noise) rather than drawn independently
# of duration — the previous version drew distance and duration from two
# unrelated distributions, which meant a synthetic "10-minute trip" and a
# "60-minute trip" were equally likely to be 2km or 20km. Deriving distance
# from duration also naturally produces a right-skewed distance distribution
# (matching real trip-distance data) without needing a separate shape fix.
IMPLIED_AVG_SPEED_KMH = 19.0


def _circular_gaussian(hour: float, center: float, width_sq: float) -> float:
    """Gaussian bump on a 24-hour clock: distance wraps at the midnight boundary.

    A bump centered near midnight needs this — a naive `hour - center` would
    treat 11pm and 1am as 22 hours apart instead of 2.
    """
    diff = abs(hour - center)
    diff = min(diff, 24 - diff)
    return float(np.exp(-(diff**2) / width_sq))


def default_ground_truth(
    zones: list[str],
    base_rate_per_minute: float = 0.5,
    bin_minutes: int = 15,
    mean_patience_seconds: float = 300.0,
    seed: int = 0,
) -> tuple[NHPPArrivalModel, ODModel, TravelTimeModel, AbandonmentModel]:
    """A plausible ground truth for a Bengaluru-like city — see the module
    docstring for which parts are grounded in real facts and which remain
    arbitrary placeholders pending real data."""
    rng = np.random.default_rng(seed)
    n_bins = int(24 * 60 / bin_minutes)

    rates: dict[tuple[str, str, int], float] = {}
    for zone in zones:
        zone_scale = rng.uniform(0.5, 1.5)
        for b in range(n_bins):
            hour = (b * bin_minutes) // 60
            morning = _circular_gaussian(hour, 9, 8)
            evening = _circular_gaussian(hour, 18, 8)
            nightlife = _circular_gaussian(hour, NIGHTLIFE_PEAK_HOUR, NIGHTLIFE_WIDTH_SQ)
            shape = 1.0 + 0.8 * morning + 1.2 * evening + NIGHTLIFE_WEIGHT * nightlife
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
    fare_per_km: float = FARE_PER_KM,
    fare_per_minute: float = FARE_PER_MINUTE,
    fare_base: float = FARE_BASE,
    night_surcharge_multiplier: float = NIGHT_SURCHARGE_MULTIPLIER,
    driver_pay_fraction: float = DRIVER_PAY_FRACTION,
    implied_avg_speed_kmh: float = IMPLIED_AVG_SPEED_KMH,
    speed_noise_sigma: float = 0.3,
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
            duration_minutes = (request.dropoff_time - request.pickup_time) / 60.0
            speed_noise = rng.lognormal(0.0, speed_noise_sigma)
            distance_km = max(0.3, (duration_minutes / 60.0) * implied_avg_speed_kmh * speed_noise)
            fare = fare_base + fare_per_km * distance_km + fare_per_minute * duration_minutes
            pickup_hour = int((request.pickup_time // 3600) % 24)
            if pickup_hour in NIGHT_SURCHARGE_HOURS:
                fare *= night_surcharge_multiplier
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
