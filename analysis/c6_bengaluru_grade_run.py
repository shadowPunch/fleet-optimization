"""C6, one rung past what's been run so far: what survives at Namma
Yatri-grade observability specifically — ward-level zones and trip-only
timestamps, not just coarser time/OD binning. No real Bengaluru data
required or used; this measures what the *resolution* itself costs,
independent of ever obtaining that data (see TECHNICAL_REPORT.md §2 for
why Bengaluru access has stayed C6-applicability-only throughout).

The coarsening ladder's existing rungs (`coarsening_ladder.py`) already
coarsen time binning, destroy OD correlation, and round fare/distance to
route means. Two things they never touch: the zone geography itself
(still this project's five
fine synthetic zones throughout) and the request-vs-pickup distinction
(still two separate, precise timestamps). Real published ward-aggregate
data has neither — it reports trip counts between a handful of wards,
timestamped once, for realized trips only. This adds a fifth rung,
cumulative on top of rung 3, with both of those actually applied, and runs
it on `value_weight_sweep_run.py`'s genuine indifference-set pair
(`weight_0.5` vs. `weight_1.5`) — the same pair `coarsening_ladder_close_pair_run.py`
already used for the first four rungs, giving coarsening its best shot at
actually separating a close pair.

**Why this needs its own script, not just a 5th entry in the existing
one's rung list**: `ValueCorrectedHungarianPolicy` bakes its value
function in at construction time. The existing close-pair run reuses one
value function, fit once at nominal (fine-zone) resolution, unchanged
across every rung — fine as long as no rung relabels the zones. This one
does. Reusing that same value function against ward-relabeled zones would
silently degenerate `cost_to_go` to 0.0 for every candidate (no key in the
value function matches a ward label), collapsing both `weight_0.5` and
`weight_1.5` to identical behavior for a reason that has nothing to do
with coarsening — so this script builds a second, ward-level value
function (refit from the ward-aggregated data, the same thing a real
deployment would have to do if this were genuinely all it could observe)
and passes it in via `run_coarsening_ladder`'s `policies_by_rung` override,
specifically for this one rung.

Usage: uv run python analysis/c6_bengaluru_grade_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dispatch_eval.calibration.value_function import compute_value_function
from dispatch_eval.coarsening_ladder import (
    DEFAULT_RUNGS,
    CoarseningRung,
    apply_coarsening,
    ranking_shift_summary,
    run_coarsening_ladder,
)
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.policies.value_corrected_hungarian import ValueCorrectedHungarianPolicy
from dispatch_eval.ranking_flip import fit_all_models
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "c6_bengaluru_grade_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 20
N_REPLICATIONS = 4
SEED = 23

# Five street-level synthetic zones collapsed to three ward-equivalent
# ones -- meaningfully fewer, not a relabeling exercise.
WARD_MAP = {"A": "W1", "B": "W1", "C": "W2", "D": "W2", "E": "W3"}
WARD_ZONES = sorted(set(WARD_MAP.values()))

BENGALURU_GRADE_RUNG = CoarseningRung(
    "rung4_bengaluru_ward_grade",
    bin_minutes=1440,
    od_time_bin_minutes=1440,
    destroy_od_correlation=True,
    coarsen_fare_distance=True,
    zone_map=WARD_MAP,
    strip_request_pickup_split=True,
)
RUNGS = [*DEFAULT_RUNGS, BENGALURU_GRADE_RUNG]


def main() -> None:
    tuned = json.loads(TUNING_RESULTS_PATH.read_text())
    radius = tuned["B2_value_corrected"]["best_params"]["radius"]

    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    value_function = compute_value_function(
        ZONES, models.arrival, models.od, models.travel_time, day_type="all", horizon_seconds=24 * 3600.0
    )
    fine_policies = {
        "weight_0.5": ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=0.5, matching_radius_seconds=radius
        ),
        "weight_1.5": ValueCorrectedHungarianPolicy(
            value_function=value_function, value_weight=1.5, matching_radius_seconds=radius
        ),
    }

    # Ward-level value function: apply the exact same coarsening
    # `run_coarsening_ladder` will apply for this rung's own position in
    # `RUNGS` (seeded identically, so this is the same coarsened data its
    # bootstrap draws are resampled from, not a separate approximation),
    # then refit -- what a real deployment observing only this resolution
    # would actually have to fit its value function from.
    rung_index = RUNGS.index(BENGALURU_GRADE_RUNG)
    coarsening_rng = np.random.default_rng(np.random.SeedSequence([SEED, rung_index]))
    ward_coarsened = apply_coarsening(trips_df, BENGALURU_GRADE_RUNG, coarsening_rng)
    print("(building ward-level value function from the same coarsened data this rung's bootstrap draws use)")

    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)
    abandonment_model = AbandonmentModel(mean_patience_seconds=300.0)

    print(f"Running the coarsening ladder ({len(RUNGS)} rungs, including the new Bengaluru-grade one) "
          f"on weight_0.5 vs weight_1.5...")
    results = run_coarsening_ladder(
        trips_df, ZONES, fine_policies, FLEET_SIZE, abandonment_model, config,
        N_BOOTSTRAP, N_REPLICATIONS, SEED, rungs=RUNGS,
        policies_by_rung={BENGALURU_GRADE_RUNG.name: _ward_policies(ward_coarsened, radius)},
    )
    summary = ranking_shift_summary(results, baseline_rung="rung0_fine_grained")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    for rung_name, s in summary.items():
        print(f"=== {rung_name} ===")
        print(f"  nominal ranking: {s['nominal_ranking']}")
        print(f"  Kendall tau to rung0: {s['kendall_tau_to_baseline']:.3f}")
        print(f"  P(ranked first): {s['probability_ranked_first']}")
        print()


def _ward_policies(ward_coarsened_df, radius: float) -> dict[str, ValueCorrectedHungarianPolicy]:
    ward_models = fit_all_models(
        ward_coarsened_df, bin_minutes=BENGALURU_GRADE_RUNG.bin_minutes,
        od_time_bin_minutes=BENGALURU_GRADE_RUNG.od_time_bin_minutes,
    )
    ward_value_function = compute_value_function(
        WARD_ZONES, ward_models.arrival, ward_models.od, ward_models.travel_time, day_type="all",
        horizon_seconds=24 * 3600.0,
    )
    return {
        "weight_0.5": ValueCorrectedHungarianPolicy(
            value_function=ward_value_function, value_weight=0.5, matching_radius_seconds=radius
        ),
        "weight_1.5": ValueCorrectedHungarianPolicy(
            value_function=ward_value_function, value_weight=1.5, matching_radius_seconds=radius
        ),
    }


if __name__ == "__main__":
    main()
