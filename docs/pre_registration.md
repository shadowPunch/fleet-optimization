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

## Amendment, 2026-08-04 — data-source roles restructured

Made **before** any real-data validation was attempted against any
source (see `dispatch-evaluation-project-plan.md`'s own dated amendment,
§2, for the full rationale) — a legitimate time to amend a pre-commitment
because the primary source turned out unable to support it, which is a
different thing from amending after seeing a validation fail. In short:
Namma Yatri's ward aggregates have no `request_datetime`/`pickup_datetime`
split at all, so wait time — the metric the KS-distance threshold below
was written against — is not observable from that source under any
circumstance, not just difficult to obtain. Delhi NCR's Kaggle set
remains of unverified, plausibly-synthetic provenance and is dropped from
every result.

**What changes below**: "Wait-time distribution" and "Hour-of-day arrival
shape" are validated against **real NYC TLC data**
(`sources/nyc_tlc.py`, verified against the live schema), not Bengaluru —
this is the only source in this project with real trip-level timestamps
to validate against at all. "Ward-level trip counts" and "Cancellation
rate" are unchanged in mechanism (Namma Yatri's aggregates are exactly
the right shape for a count-level check) but now belong to Bengaluru's
own separate applicability study (C6, pointed at real Namma Yatri data),
not to the main confirmatory P1 run — a study that answers "what can be
concluded from ward-level aggregates alone," not "does the simulator
reproduce Bengaluru specifically." Neither NYC nor Bengaluru data has
actually been fetched and run against these thresholds yet as of this
commit — that is the concrete next step this amendment sets up, not
something this amendment itself completes.

## P1 validation acceptance thresholds

Per the plan's P1 section, checked against held-out days not used for
fitting. **Target source for each row updated per the amendment above.**

| Check | Metric | Threshold | Validate against | Status if failed |
|---|---|---|---|---|
| Wait-time distribution | KS distance (`calibration.fleet_size.ks_distance`) between simulated and observed wait times | ≤ 0.10 | **NYC TLC** (real trip-level `request_datetime`) | Simulator does not reproduce the wait-time distribution's *shape* — see "if this fails" below |
| Hour-of-day arrival shape | Cosine similarity between simulated and observed hourly request-count vectors (same metric as `analysis/nyc_reference_comparison.py`'s `hourly_cosine_similarity`) | ≥ 0.90 | **NYC TLC** | Arrival timing shape is not well recovered |
| Ward-level trip counts | Simulated vs. published per-ward aggregate counts, relative error | within ±20% | **Namma Yatri** (Bengaluru applicability study, C6 — not the main P1 run) | Spatial demand shape is off |
| Cancellation rate | Simulated vs. published aggregate cancellation rate | within ±5 percentage points (absolute) | **Namma Yatri** (same, C6) | Abandonment-hazard specification needs revisiting |
| Trips/vehicle-hour, empty-mile fraction, revenue/vehicle-hour | vs. any available real anchor | **Secondary** — checked and reported if a real anchor exists at all (per `docs/observability_table.md`, most of these are latent in the public data), not a hard pass/fail gate | NYC where available | — |

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

**Checked against the plan's own target scale, not just asserted**:
`run_study.py` was run once at the pre-registered minimum
(`--n-bootstrap 40 --n-replications 4`, 800 rows) and once at the plan's
target (`--n-bootstrap 200 --n-replications 5`, 5000 rows). The ranking
itself was already fully converged at 40 draws: identical nominal
ranking and indifference set (`{B2_value_corrected}`) at both scales,
confirmed again after the B5 fix below (same result a third time, at
B=200, `results/study_results_at_scale_v2.parquet`). **This validates
the `n_bootstrap ≥ 40` minimum for *ranking stability* specifically.** It
does *not* extend to fine-grained variance estimates: the
`input_uncertainty_ratio` itself moved between scales (0.314 at B=40 →
0.239 at B=200, same qualitative conclusion both times) — consistent
with `docs/mde_scaling_validation.md`'s own finding that variance-type
statistics need more draws to stabilize than a ranking does. Report a
ranking off 40 draws with reasonable confidence; report a variance
decomposition off nothing less than the full 200.

Gap-closed fractions from the *first* B=40/B=200 comparison (1.05-1.23,
every policy beating the bound) are superseded by the B5 fix directly
below — they were an artifact of the two bugs described there, not a
finding about these policies. The corrected B=200 run
(`study_results_at_scale_v2.parquet`) reports B1/B3/B4 closing 90-93% of
the gap and B2 closing 95%, all properly below 1.0.

## B5 clairvoyant gap-closed reporting

**Fixed, not just worked around.** `docs/coarsening_ladder.md` and
`clairvoyant.py`'s own docstring originally documented every tuned online
policy beating B5's "upper" bound by 5-10% — diagnosed down to two actual
bugs (pessimistic exit-pinning that over-constrained capacity, and a
`bin_minutes` too coarse for `mean_patience_seconds=300.0`, silently
excluding 67.5% of requests as "unservable" before the solver ever saw
them) and fixed at the source, not patched around with a caveat. Both
fixes together, re-measured at B=200: gap-closed fractions of 0.90-0.95
across B1-B4, all properly below 1.0, with 9.4% of requests still
excluded by discretization (`ClairvoyantResult.fraction_excluded_by_discretization`
— report this number alongside any gap-closed figure; it isn't yet
zero). **Pre-registered commitment, updated**: any confirmatory-study
gap-closed report must state `clairvoyant_bin_minutes` used and the
resulting `fraction_excluded_by_discretization` explicitly — a number
not near zero means the bound isn't trustworthy yet at that setting,
regardless of how plausible the resulting gap-closed figure looks.

## `run_study.py`

The plan also calls for "one `run_study.py` that regenerates every figure
from raw data." Built: `run_study.py` at the project root runs P1
calibration, P2 tuning, and the P3 bootstrap-CRN experiment (with B5
wired in) end to end, and writes every `(policy, bootstrap_draw,
replication) → metrics` cell to one parquet file — not just summary
statistics — with every headline number it prints computed from that same
in-memory table. Run at both the pre-registered minimum
(`--n-bootstrap 40`) and the plan's target scale (`--n-bootstrap 200`,
after the B5 fix): consistent with every other finding in this document
(B2 cleanly wins; B1-B4 close 90-95% of the clairvoyant gap, properly
below 1.0). It does not yet cover C3-C6 (decision currency, compute
parity, the crosswalk, the coarsening ladder) — those remain the separate
`analysis/*.py` scripts they already are, which is fine for exploratory
work; folding them into `run_study.py` as it matures toward "regenerates
every figure" is a reasonable future step, not done here.
