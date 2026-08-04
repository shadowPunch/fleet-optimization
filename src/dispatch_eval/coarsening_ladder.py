"""C6 — Coarsening ladder: how much of the ranking survives realistic data loss?

The plan's question: this project's synthetic Delhi NCR data is far richer
than what Bengaluru's real open data (Namma Yatri's ward aggregates)
actually offers. Re-running P3's exact bootstrap-ranking-flip experiment on
progressively coarsened versions of the *same* underlying data — instead of
just asserting "real data is worse" — turns that gap into a graded,
measured answer: how much of the C2 ranking survives at each rung, down to
something close to what a real Indian city's public data resolution would
force.

Every rung still produces a `TRIP_RECORD_SCHEMA`-shaped DataFrame, so it
runs through `run_ranking_flip_experiment` completely unchanged — the
"coarsening" is two things, applied cumulatively:

1. **Coarser time/OD binning** — `bin_minutes`/`od_time_bin_minutes` passed
   straight to `fit_all_models` (see `ranking_flip.py`'s docstring for why
   that's a parameter there now). No data is altered, only the resolution
   the *models* are fit at.
2. **Information actually destroyed in the data itself** —
   `coarsen_fare_distance_to_od_mean` (replace each trip's fare/distance
   with its (origin, dest)-pair average, modeling "you only know typical
   route economics, not individual receipts") and
   `coarsen_destroy_od_correlation` (shuffle the whole `dest_zone` column,
   modeling "you only observe origin-ward counts, not where requests
   actually went" — real Namma Yatri ward-aggregate data has no
   destination field at all). Both keep the schema valid but genuinely
   remove information a coarser real source wouldn't have.

The rungs below are one reasonable operationalization of the plan's stated
sequence ("15-min time rounding → ward-level OD only → ward-only aggregate
counts with no timestamps, fare rounding at each step"), not a unique
correct one — see `DEFAULT_RUNGS`' own inline comments for the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import kendalltau

from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.base import DispatchPolicy
from dispatch_eval.ranking_flip import RankingFlipResult, run_ranking_flip_experiment
from dispatch_eval.simulator.runner import StudyConfig


def coarsen_fare_distance_to_od_mean(df: pl.DataFrame) -> pl.DataFrame:
    """Replace each completed trip's `trip_distance_km`/`fare` with the mean
    for its `(origin_zone, dest_zone)` pair — removes individual-trip
    variance while preserving the route-level average. Rows with a null
    fare/distance (cancelled/incomplete trips) stay null.
    """
    means = df.filter(pl.col("status") == "completed").group_by(["origin_zone", "dest_zone"]).agg(
        pl.col("trip_distance_km").mean().alias("_mean_distance"),
        pl.col("fare").mean().alias("_mean_fare"),
    )
    out = df.join(means, on=["origin_zone", "dest_zone"], how="left")
    out = out.with_columns(
        pl.when(pl.col("trip_distance_km").is_not_null())
        .then(pl.col("_mean_distance"))
        .otherwise(None)
        .alias("trip_distance_km"),
        pl.when(pl.col("fare").is_not_null()).then(pl.col("_mean_fare")).otherwise(None).alias("fare"),
    )
    return out.drop(["_mean_distance", "_mean_fare"])


def coarsen_destroy_od_correlation(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Shuffle the entire `dest_zone` column (not within any group).

    Preserves the marginal popularity of each zone as a destination but
    destroys the true joint (origin, dest) structure — a full-column
    permutation, not a within-origin one, since permuting within a group
    that's already fixed to one origin value is a no-op on the observed
    (origin, dest) distribution.
    """
    dest = df["dest_zone"].to_numpy().copy()
    permuted = dest[rng.permutation(len(dest))]
    return df.with_columns(pl.Series("dest_zone", permuted))


@dataclass
class CoarseningRung:
    name: str
    bin_minutes: int
    od_time_bin_minutes: int
    destroy_od_correlation: bool
    coarsen_fare_distance: bool


# Rung 0 is deliberately *finer* than this project's own P2/P3 default
# (bin_minutes=15) — representing a platform's internal, full-resolution
# view — so rung 1 ("15-min time rounding," the plan's literal first step)
# is a real degradation from it, not a no-op. Each later rung adds one more
# real information loss on top of the previous rung's.
DEFAULT_RUNGS: list[CoarseningRung] = [
    CoarseningRung("rung0_fine_grained", bin_minutes=1, od_time_bin_minutes=15,
                    destroy_od_correlation=False, coarsen_fare_distance=False),
    CoarseningRung("rung1_15min_rounding", bin_minutes=15, od_time_bin_minutes=60,
                    destroy_od_correlation=False, coarsen_fare_distance=False),
    CoarseningRung("rung2_ward_level_od_fare_rounding", bin_minutes=15, od_time_bin_minutes=60,
                    destroy_od_correlation=False, coarsen_fare_distance=True),
    CoarseningRung("rung3_ward_only_aggregate_no_timestamps", bin_minutes=1440, od_time_bin_minutes=1440,
                    destroy_od_correlation=True, coarsen_fare_distance=True),
]


def apply_coarsening(df: pl.DataFrame, rung: CoarseningRung, rng: np.random.Generator) -> pl.DataFrame:
    """Apply `rung`'s data-destroying transforms (not its binning
    parameters, which go straight to `run_ranking_flip_experiment` instead)
    to `df`. OD destruction runs before fare/distance coarsening so the
    latter's (origin, dest)-pair means are computed against whatever
    destinations this rung actually has — at rung 3, that means the
    "route-level average" is itself computed on shuffled, fake routes,
    which is the honest consequence of not having real OD data anymore.
    """
    out = df
    if rung.destroy_od_correlation:
        out = coarsen_destroy_od_correlation(out, rng)
    if rung.coarsen_fare_distance:
        out = coarsen_fare_distance_to_od_mean(out)
    return out


def run_coarsening_ladder(
    trips_df: pl.DataFrame,
    zones: list[str],
    policies: dict[str, DispatchPolicy],
    fleet_size: int,
    abandonment_model: AbandonmentModel,
    config: StudyConfig,
    n_bootstrap: int,
    n_replications: int,
    seed: int,
    rungs: list[CoarseningRung] = DEFAULT_RUNGS,
) -> dict[str, RankingFlipResult]:
    """Run the full C2 bootstrap-ranking-flip experiment once per rung, on a
    progressively coarsened version of `trips_df`. Every rung's coarsening
    RNG is seeded from `(seed, rung_index)` — the rung's position in
    `rungs`, not its name (Python's built-in `hash()` on a string is
    process-randomized by default and would silently break
    reproducibility across runs).
    """
    results: dict[str, RankingFlipResult] = {}
    for i, rung in enumerate(rungs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        coarsened = apply_coarsening(trips_df, rung, rng)
        results[rung.name] = run_ranking_flip_experiment(
            coarsened,
            zones,
            policies,
            fleet_size,
            abandonment_model,
            config,
            n_bootstrap,
            n_replications,
            seed,
            bin_minutes=rung.bin_minutes,
            od_time_bin_minutes=rung.od_time_bin_minutes,
        )
    return results


def ranking_shift_summary(
    results: dict[str, RankingFlipResult], baseline_rung: str
) -> dict[str, dict]:
    """For every rung (including the baseline itself, as a sanity check —
    its own tau should be 1.0), Kendall's tau between that rung's nominal
    ranking and the baseline rung's, plus each rung's own
    P(ranked first) distribution — the plan's "report the ranking shift at
    each rung."
    """
    baseline = results[baseline_rung]
    baseline_order = [baseline.policy_names.index(name) for name in baseline.nominal_ranking]

    summary: dict[str, dict] = {}
    for rung_name, result in results.items():
        order = [result.policy_names.index(name) for name in result.nominal_ranking]
        tau, _ = kendalltau(baseline_order, order)
        summary[rung_name] = {
            "nominal_ranking": result.nominal_ranking,
            "kendall_tau_to_baseline": float(tau),
            "probability_ranked_first": result.probability_ranked_first(),
        }
    return summary
