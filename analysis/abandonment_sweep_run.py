"""C1's real three-term variance decomposition: the abandonment-hazard
sweep as the structural stratum, run as an outer loop around the existing
P3 bootstrap.

`TECHNICAL_REPORT.md` already commits to treating the abandonment
hazard as structural, not fit: "fleet size and the abandonment hazard are
not jointly identified from matched-trip data alone... `mean_patience_seconds
∈ {60, 180, 300, 600, 900}`... any P3 conclusion that holds across all five
is reported as robust; any that doesn't is itself a finding." What was
missing was folding that sweep into `variance_decomposition`'s own
arithmetic — every `input_uncertainty_ratio` reported so far
(`TECHNICAL_REPORT.md`, `run_study.py`) only ever varied the
bootstrap draw at one *fixed* abandonment spec, so it was a floor on input
uncertainty's contribution, not an estimate (its own docstring said so).
This runs `run_ranking_flip_experiment` once per pre-registered spec and
feeds all five into `variance_decomposition_three_term` — the real
intrinsic / input-estimation / structural split.

Tracked pair: B2 (this project's consistent winner) vs. B0 (the reactive
baseline) — fixed across every spec so the paired difference means the
same thing at each one, rather than re-picking "best vs. worst" per spec
(which could silently swap identities and made the across-spec comparison
meaningless).

Usage: uv run python analysis/abandonment_sweep_run.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dispatch_eval.forecast_degradation import DegradedModels, build_policy_ladder
from dispatch_eval.models import AbandonmentModel
from dispatch_eval.ranking_flip import (
    fit_all_models,
    indifference_set,
    run_ranking_flip_experiment,
    variance_decomposition_three_term,
)
from dispatch_eval.simulator.runner import StudyConfig
from dispatch_eval.sources.synthetic import generate_synthetic_trips

TUNING_RESULTS_PATH = Path(__file__).parent / "cache" / "policy_tuning_results.json"
OUTPUT_PATH = Path(__file__).parent / "cache" / "abandonment_sweep_results.json"

ZONES = ["A", "B", "C", "D", "E"]
FLEET_SIZE = 30
HORIZON_SECONDS = 6 * 3600.0
N_BOOTSTRAP = 25
N_REPLICATIONS = 4
SEED = 23

# Pre-registered sweep (TECHNICAL_REPORT.md): 1, 3, 5, 10, 15 minutes.
ABANDONMENT_SPECS = [60.0, 180.0, 300.0, 600.0, 900.0]
TRACKED_PAIR = ("B2_value_corrected", "B0_nearest_idle")  # (a, b) for variance_decomposition's diff = a - b


def main() -> None:
    tuned_params = json.loads(TUNING_RESULTS_PATH.read_text())
    trips_df = generate_synthetic_trips(
        n_days=10, zones=ZONES, start_date=datetime(2026, 1, 5), fleet_size=60, seed=5
    )
    models = fit_all_models(trips_df)
    # Reused unchanged across every spec, matching this project's
    # established convention (forecast_degradation.build_policy_ladder's
    # own docstring): the abandonment hazard is the axis under test here,
    # not policy tuning -- changing both at once would leave it ambiguous
    # which one moved any result.
    nominal_models = DegradedModels(arrival=models.arrival, od=models.od, travel_time=models.travel_time)
    config = StudyConfig(zones=ZONES, day_type="all", horizon_seconds=HORIZON_SECONDS)

    metric_a_by_spec: dict[float, object] = {}
    metric_b_by_spec: dict[float, object] = {}
    per_spec_summary = {}

    for spec in ABANDONMENT_SPECS:
        print(f"\n=== mean_patience_seconds={spec} ===")
        abandonment_model = AbandonmentModel(mean_patience_seconds=spec)
        policies = build_policy_ladder(nominal_models, tuned_params, ZONES, config.day_type)
        result = run_ranking_flip_experiment(
            trips_df, ZONES, policies, FLEET_SIZE, abandonment_model, config, N_BOOTSTRAP, N_REPLICATIONS, SEED,
        )
        indifferent = indifference_set(result, alpha=0.05)
        probs = result.probability_ranked_first()
        print(f"  nominal ranking: {result.nominal_ranking}")
        print(f"  P(ranked first): {probs}")
        print(f"  indifference set: {sorted(indifferent)}")

        metric_a_by_spec[spec] = result.metric_by_policy[TRACKED_PAIR[0]]
        metric_b_by_spec[spec] = result.metric_by_policy[TRACKED_PAIR[1]]
        per_spec_summary[str(spec)] = {
            "nominal_ranking": result.nominal_ranking,
            "probability_ranked_first": probs,
            "indifference_set_alpha_0.05": sorted(indifferent),
        }

    decomp = variance_decomposition_three_term(metric_a_by_spec, metric_b_by_spec)

    # Robustness check the pre-registration itself calls for: does the
    # winner (B2) actually stay first across every spec, or is that itself
    # the finding?
    b2_always_first = all(
        s["nominal_ranking"][0] == "B2_value_corrected" for s in per_spec_summary.values()
    )

    print(f"\n{'='*70}")
    print(f"THREE-TERM VARIANCE DECOMPOSITION ({TRACKED_PAIR[0]} vs {TRACKED_PAIR[1]})")
    print(f"{'='*70}")
    print(f"  intrinsic_variance:        {decomp['intrinsic_variance']:.2f}")
    print(f"  input_estimation_variance: {decomp['input_estimation_variance']:.2f}")
    print(f"  structural_variance:       {decomp['structural_variance']:.2f}")
    print(f"  input_uncertainty_ratio:     {decomp['input_uncertainty_ratio']:.3f}")
    print(f"  structural_uncertainty_ratio: {decomp['structural_uncertainty_ratio']:.3f}")
    print(f"\n  B2 ranked first at every spec: {b2_always_first}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "abandonment_specs": ABANDONMENT_SPECS,
        "tracked_pair": list(TRACKED_PAIR),
        "per_spec": per_spec_summary,
        "variance_decomposition_three_term": decomp,
        "b2_ranked_first_at_every_spec": b2_always_first,
    }, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
