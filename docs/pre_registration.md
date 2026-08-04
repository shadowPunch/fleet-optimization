# Pre-registration

**Committed: 2026-08-04.** Per the project plan's own reproducibility-discipline
requirement: *"write down validation acceptance thresholds, the α for the
indifference set, and the primary metric before running P1 validation and
P3. Commit the file with a timestamp. This is the process correction that
mattered most in the superconductivity work and it matters more here,
because the whole paper is a claim about statistical discipline."*

## Scope: what this does and doesn't cover

This pre-registers the **final confirmatory study** — P1 validation
against real held-out data, and the P3 ranking-flip experiment run at
publication scale. It does **not** retroactively cover the exploratory
and methodology-development work already in this repository as of this
commit: building and unit-testing the simulator itself; tuning B1-B4's
own parameters (`analysis/policy_tuning_run.py`); searching for whether
this project's own synthetic-data baseline ladder produces a real
indifference set (`docs/indifference_search.md`); running the coarsening
ladder on both a large-effect and a close pair (`docs/coarsening_ladder.md`);
validating the MDE curve's scaling assumption
(`docs/mde_scaling_validation.md`); or wiring and sanity-checking B5's
clairvoyant bound. All of that was necessary methodology development and
debugging — it found and fixed real bugs (the CRN seeding bug), found a
real indifference set, and found a real flaw in B5's bound at one
parameter setting — not a search for a favorable headline result, and
none of it used real Bengaluru/Delhi NCR data (none has been available in
this environment). Pre-registering *after* that development work but
*before* the confirmatory run against real data is the correct order, not
a late pre-registration — the thresholds below were not chosen by looking
at what the confirmatory run produces, because that run hasn't happened.

## P1 validation acceptance thresholds

Per the plan's P1 section, checked against held-out days not used for
fitting:

| Check | Metric | Threshold | Status if failed |
|---|---|---|---|
| Wait-time distribution | KS distance (`calibration.fleet_size.ks_distance`) between simulated and observed wait times | ≤ 0.10 | Simulator does not reproduce the wait-time distribution's *shape* — see "if this fails" below |
| Hour-of-day arrival shape | Cosine similarity between simulated and observed hourly request-count vectors (same metric as `analysis/nyc_reference_comparison.py`'s `hourly_cosine_similarity`) | ≥ 0.90 | Arrival timing shape is not well recovered |
| Ward-level trip counts | Simulated vs. Namma Yatri's published per-ward aggregate counts, relative error | within ±20% | Spatial demand shape is off — investigate before trusting C5 |
| Cancellation rate | Simulated vs. published aggregate cancellation rate | within ±5 percentage points (absolute) | Abandonment-hazard specification needs revisiting |
| Trips/vehicle-hour, empty-mile fraction, revenue/vehicle-hour | vs. any available real anchor | **Secondary** — checked and reported if a real anchor exists at all (per `docs/observability_table.md`, most of these are latent in the public data), not a hard pass/fail gate |

**0.10 for KS distance is a practical-similarity threshold, not a formal
hypothesis-test critical value** — at the trip volumes this project
works with, the formal two-sample KS critical value at α=0.05 would be
far smaller than 0.10 and would reject on almost any real-vs-simulated
comparison, which is not informative for "is this simulator decision-
useful," the actual question. 0.10 is the practical bar; the formal test
statistic should still be reported alongside it, not hidden.

**If validation fails these thresholds**, per the plan's own risk table:
*"This is a bigger result about identifiability from public data. Write
it up as the primary finding; C2 becomes secondary."* That is the
pre-registered fallback, not a silent threshold adjustment after seeing
the numbers.

## Abandonment-hazard sweep (P1)

Since fleet size and the abandonment hazard are not jointly identified
from matched-trip data alone (see `calibration/fleet_size.py`), the
abandonment hazard is a structural axis to sweep, not fit. Pre-registered
sweep, per the plan's "3-5 specifications spanning plausible behaviour":
`mean_patience_seconds ∈ {60, 180, 300, 600, 900}` (1, 3, 5, 10, 15
minutes). Any P3 conclusion that holds across all five is reported as
robust; any that doesn't is itself a finding, not grounds to pick
whichever specification looks best.

## P3: primary metric and indifference-set α

- **Primary metric**: mean wait time per completed request
  (`ranking_flip.run_ranking_flip_experiment`'s default `metric_fn`).
  Secondary/exploratory metrics — p90 wait time, fraction of requests
  served — may be reported alongside it but must not be substituted as
  the ranking criterion after seeing which one makes a given policy look
  better.
- **Indifference-set α**: 0.05, applied via
  `ranking_flip.indifference_set`'s bootstrap-percentile method. This
  matches the value already used as the code default and in every
  exploratory run this session — stated here as a genuine commitment, not
  merely inherited from not having changed a default.

## Bootstrap budget

`docs/mde_scaling_validation.md` found this directly, not as a
theoretical concern: a first pass at `n_bootstrap=15` produced a
non-monotonic, unstable `across_theta_variance` estimate; re-running at
`n_bootstrap=50` produced a clean, monotonic signal. **Pre-registered
minimum for the confirmatory study: `n_bootstrap ≥ 40`**, with the plan's
own suggested B≈200-500 as the target if compute allows — not the B≤5-50
range used for this repository's exploratory/development runs so far,
which were sized for iteration speed, not for a publishable variance
estimate.

## B5 clairvoyant gap-closed reporting

`docs/coarsening_ladder.md` and `clairvoyant.py`'s own docstring document
a real finding: at `mean_patience_seconds=300.0`, every tuned online
policy from B1 up beat B5's "upper" bound by 5-10%, because the pinning
approximation is loose enough at that patience setting to not actually be
an upper bound. **Pre-registered commitment**: any confirmatory-study
gap-closed report must either (a) also compute B5 at a shorter patience
window (60-120s, from the sweep above) and use that version as the
headline bound, or (b) explicitly caption the 300s-patience numbers as
"relative to a conservative reference schedule," never as "fraction of
the theoretical maximum" without that qualifier.

## `run_study.py`

The plan also calls for "one `run_study.py` that regenerates every figure
from raw data." Built: `run_study.py` at the project root runs P1
calibration, P2 tuning, and the P3 bootstrap-CRN experiment (with B5
wired in) end to end, and writes every `(policy, bootstrap_draw,
replication) → metrics` cell to one parquet file — not just summary
statistics — with every headline number it prints computed from that same
in-memory table. Real run at the pre-registered minimum
(`--n-bootstrap 40 --n-replications 4`, ~4.5 minutes): 800 rows written,
consistent with every other finding in this document (B2 cleanly wins,
every online policy still beats B5's bound at 300s patience by ~17-23%).
It does not yet cover C3-C6 (decision currency, compute parity, the
crosswalk, the coarsening ladder) — those remain the separate
`analysis/*.py` scripts they already are, which is fine for exploratory
work; folding them into `run_study.py` as it matures toward "regenerates
every figure" is a reasonable future step, not done here.
