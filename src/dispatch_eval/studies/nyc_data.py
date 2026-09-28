"""NYC TLC high-volume FHV data: fetch, cache, split, and align for simulation.

The study window is Manhattan-only Uber (HV0003) trips on January 2024
weekdays, 12:00-18:00, split by day into a calibration set and a held-out
set that is never touched while fitting. Scope and split are fixed in
`configs/nyc.yaml` and were chosen before any validation was run (see
TECHNICAL_REPORT.md §5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from dispatch_eval.sources.nyc_tlc import adapt_nyc_tlc_trips

TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{month}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TIMESTAMP_COLUMNS = ("request_ts", "on_scene_ts", "pickup_ts", "dropoff_ts")


@dataclass(frozen=True)
class NYCScope:
    month: str  # "2024-01"
    operator: str  # hvfhs_license_num, e.g. "HV0003" (Uber)
    borough: str
    window_start_hour: int
    window_end_hour: int
    calibration_days: tuple[int, ...]
    held_out_days: tuple[int, ...]

    @property
    def horizon_seconds(self) -> float:
        return (self.window_end_hour - self.window_start_hour) * 3600.0

    @classmethod
    def from_config(cls, cfg: dict) -> NYCScope:
        return cls(
            month=cfg["month"],
            operator=cfg["operator"],
            borough=cfg["borough"],
            window_start_hour=cfg["window_start_hour"],
            window_end_hour=cfg["window_end_hour"],
            calibration_days=tuple(cfg["calibration_days"]),
            held_out_days=tuple(cfg["held_out_days"]),
        )


@dataclass
class NYCData:
    scope: NYCScope
    zones: list[str]
    calibration: pl.DataFrame  # canonical trip records, real timestamps
    held_out: pl.DataFrame

    def for_simulation(self, trips: pl.DataFrame) -> pl.DataFrame:
        """Shift timestamps so the window start maps to simulated midnight.

        The simulator's clock always starts at 0 and every fitted model bins
        by absolute hour-of-day; fit on unshifted 12:00-18:00 timestamps and
        every rate lands in bins 12-17, which the simulated clock never
        visits. Durations (wait, trip time) are unaffected by the shift.
        """
        offset = pl.duration(hours=self.scope.window_start_hour)
        return trips.with_columns([(pl.col(c) - offset).alias(c) for c in TIMESTAMP_COLUMNS])


def wait_seconds(trips: pl.DataFrame) -> np.ndarray:
    """Rider wait (request -> pickup), the study's primary metric."""
    waits = (trips["pickup_ts"] - trips["request_ts"]).dt.total_seconds().to_numpy()
    return waits[waits >= 0]


def approach_seconds(trips: pl.DataFrame) -> np.ndarray:
    """Request -> driver on scene: the dispatch + drive part of the wait."""
    secs = (trips["on_scene_ts"] - trips["request_ts"]).dt.total_seconds().drop_nulls().to_numpy()
    return secs[secs >= 0]


def load_nyc_data(scope: NYCScope, cache_dir: Path) -> NYCData:
    """Load the study window, downloading and caching it on first use."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zone_ids = _borough_zone_ids(scope.borough, cache_dir / "nyc_taxi_zone_lookup.csv")
    raw = _window_trips(scope, zone_ids, cache_dir)
    trips = adapt_nyc_tlc_trips(raw)
    day = pl.col("request_ts").dt.day()
    return NYCData(
        scope=scope,
        zones=[str(z) for z in zone_ids],
        calibration=trips.filter(day.is_in(scope.calibration_days)),
        held_out=trips.filter(day.is_in(scope.held_out_days)),
    )


def _borough_zone_ids(borough: str, cache_path: Path) -> list[int]:
    if not cache_path.exists():
        pl.read_csv(ZONE_LOOKUP_URL).write_csv(cache_path)
    lookup = pl.read_csv(cache_path)
    return lookup.filter(pl.col("Borough") == borough)["LocationID"].to_list()


def _window_trips(scope: NYCScope, zone_ids: list[int], cache_dir: Path) -> pl.DataFrame:
    days = sorted(scope.calibration_days + scope.held_out_days)
    cache_path = cache_dir / (
        f"nyc_{scope.borough.lower()}_{scope.operator.lower()}_{scope.month}"
        f"_{scope.window_start_hour}-{scope.window_end_hour}_d{'-'.join(map(str, days))}.parquet"
    )
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    year, month = (int(p) for p in scope.month.split("-"))
    request = pl.col("request_datetime")
    # The window is defined on request time: that is the event the arrival
    # model fits and the horizon bounds.
    df = (
        pl.scan_parquet(TRIP_DATA_URL.format(month=scope.month))
        .filter(
            (pl.col("hvfhs_license_num") == scope.operator)
            & pl.col("PULocationID").is_in(zone_ids)
            & pl.col("DOLocationID").is_in(zone_ids)
            & (request.dt.year() == year)
            & (request.dt.month() == month)
            & request.dt.day().is_in(days)
            & (request.dt.hour() >= scope.window_start_hour)
            & (request.dt.hour() < scope.window_end_hour)
        )
        .collect()
    )
    df.write_parquet(cache_path)
    return df
