"""Runtime input models.

These are the objects the simulator actually calls during a run: an arrival
process, an OD (destination-choice) model, a travel-time model, and an
abandonment (patience) model. Each is deliberately a plain, picklable data
object plus a couple of methods — no framework.

`dispatch_eval.calibration` fits these from trip data. `dispatch_eval.sources.synthetic`
builds them directly from chosen ground-truth parameters, so the calibration
routines have something with a known answer to be tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SECONDS_PER_MINUTE = 60


@dataclass
class NHPPArrivalModel:
    """Piecewise-constant-rate non-homogeneous Poisson process per zone.

    `rates[(zone, day_type, bin_index)]` is the arrival rate in requests per
    minute for that (zone, day_type, time bin). `bin_index = minute_of_day //
    bin_minutes`.
    """

    rates: dict[tuple[str, str, int], float]
    bin_minutes: int = 15

    def rate_per_minute(self, zone: str, day_type: str, minute_of_day: float) -> float:
        bin_index = int(minute_of_day // self.bin_minutes)
        return self.rates.get((zone, day_type, bin_index), 0.0)

    def generate_arrival_minutes(
        self,
        zone: str,
        day_type: str,
        start_minute: float,
        end_minute: float,
        rng: np.random.Generator,
    ) -> list[float]:
        """Sample arrival times (minutes since midnight) in [start_minute, end_minute)."""
        arrivals: list[float] = []
        bin_start = (int(start_minute) // self.bin_minutes) * self.bin_minutes
        while bin_start < end_minute:
            bin_end = min(bin_start + self.bin_minutes, end_minute)
            window_start = max(bin_start, start_minute)
            window_len = bin_end - window_start
            if window_len > 0:
                rate = self.rate_per_minute(zone, day_type, window_start)
                n = rng.poisson(rate * window_len)
                if n > 0:
                    arrivals.extend(rng.uniform(window_start, bin_end, size=n).tolist())
            bin_start += self.bin_minutes
        arrivals.sort()
        return arrivals


@dataclass
class ODModel:
    """Row-normalised, smoothed OD distribution: destination | (origin, time bin)."""

    dest_probs: dict[tuple[str, int], dict[str, float]]
    fallback_probs: dict[str, float] = field(default_factory=dict)

    def sample_destination(self, origin_zone: str, time_bin: int, rng: np.random.Generator) -> str:
        probs = self.dest_probs.get((origin_zone, time_bin), self.fallback_probs)
        if not probs:
            return origin_zone
        zones = list(probs.keys())
        p = np.array(list(probs.values()))
        return zones[rng.choice(len(zones), p=p / p.sum())]


DEFAULT_INTRA_ZONE_PARAMS = (float(np.log(60.0)), 0.3)


@dataclass
class TravelTimeModel:
    """Lognormal travel time per (origin, dest, hour), log-seconds parameters.

    Same-zone moves (a pickup by a vehicle already in the rider's zone) use
    `intra_zone_params` instead of the fitted trip cells: a passenger trip
    that starts and ends in one zone is a poor proxy for a driver's approach
    within it. The value is a calibrated latent parameter, not a fitted one
    (see `calibration.pickup`).
    """

    params: dict[tuple[str, str, int], tuple[float, float]]
    fallback_params: tuple[float, float]
    intra_zone_params: tuple[float, float] = DEFAULT_INTRA_ZONE_PARAMS
    _expected_cache: dict[tuple[str, str, int], float] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _mu_sigma(self, origin: str, dest: str, hour: int) -> tuple[float, float]:
        if origin == dest:
            return self.intra_zone_params
        return self.params.get((origin, dest, hour), self.fallback_params)

    def expected(self, origin: str, dest: str, hour: int) -> float:
        # Called once per candidate pair on every dispatch tick; memoized
        # since the parameters never change after construction.
        key = (origin, dest, hour)
        value = self._expected_cache.get(key)
        if value is None:
            mu, sigma = self._mu_sigma(origin, dest, hour)
            value = self._expected_cache[key] = float(np.exp(mu + sigma**2 / 2))
        return value

    def sample(self, origin: str, dest: str, hour: int, rng: np.random.Generator) -> float:
        mu, sigma = self._mu_sigma(origin, dest, hour)
        return float(rng.lognormal(mu, sigma))


@dataclass
class FareModel:
    """Fare as a linear function of distance and duration, plus a driver-pay split.

    The driver-pay fraction is a modeling assumption, not a fitted quantity —
    see TECHNICAL_REPORT.md ("Driver pay" row) for why it can't be
    fit from either data source. Defaults to 1.0, matching Namma Yatri's own
    stated zero-commission policy (the project's real target platform) —
    not NYC TLC's ~0.72-0.75, which was checked against real data and found
    close only by coincidence (see analysis/nyc_reference_comparison.py):
    NYC's driver-pay structure is a regulated commission cut, structurally
    different from Namma Yatri's claimed 100%-to-driver model, not just a
    different number.
    """

    intercept: float
    distance_coef: float
    duration_coef: float
    driver_pay_fraction: float = 1.0

    def expected_fare(self, distance_km: float, duration_minutes: float) -> float:
        return max(
            0.0,
            self.intercept
            + self.distance_coef * distance_km
            + self.duration_coef * duration_minutes,
        )

    def expected_driver_pay(self, distance_km: float, duration_minutes: float) -> float:
        return self.expected_fare(distance_km, duration_minutes) * self.driver_pay_fraction


@dataclass
class BoardingModel:
    """Time from the driver arriving on scene to the rider boarding.

    Empirical inverse CDF over `quantiles` (seconds, at evenly spaced
    probabilities from 0 to 1): no parametric family is imposed, and
    sampling interpolates between quantiles so draws stay continuous.
    """

    quantiles: np.ndarray

    def sample(self, rng: np.random.Generator) -> float:
        u = rng.random() * (len(self.quantiles) - 1)
        lo = int(u)
        hi = min(lo + 1, len(self.quantiles) - 1)
        return float(self.quantiles[lo] + (u - lo) * (self.quantiles[hi] - self.quantiles[lo]))


@dataclass
class AbandonmentModel:
    """Exponential patience clock: one point on the structural-ambiguity axis.

    The project plan treats the abandonment hazard as latent and mandates
    running the whole study under 3-5 specifications spanning plausible
    behaviour (see P1). `mean_patience_seconds` is that one free parameter;
    build several `AbandonmentModel` instances to sweep it.
    """

    mean_patience_seconds: float

    def sample_patience(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.mean_patience_seconds))
