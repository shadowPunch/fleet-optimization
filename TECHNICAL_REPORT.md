# dispatch-eval: Technical Report

Input-uncertainty-aware evaluation of ride-hailing dispatch policies — full
methodology, architecture, every measured result, and how to reproduce all
of it. This is the detailed reference; [README.md](README.md) is the
short version.

## Table of contents

1. [Thesis and motivation](#1-thesis-and-motivation)
2. [Data sources and status](#2-data-sources-and-status)
3. [System architecture](#3-system-architecture)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
   - [5.1 No indifference between mechanisms — until there is](#51-no-indifference-between-mechanisms--until-there-is)
   - [5.2 The oracle-forecast confound, ruled out](#52-the-oracle-forecast-confound-ruled-out)
   - [5.3 C1 — the real three-term variance decomposition](#53-c1--the-real-three-term-variance-decomposition)
   - [5.4 C6 — the coarsening ladder, and where it breaks](#54-c6--the-coarsening-ladder-and-where-it-breaks)
   - [5.5 P1 — validation against real data](#55-p1--validation-against-real-data)
   - [5.6 C3 — decision currency and B5's clairvoyant bound](#56-c3--decision-currency-and-b5s-clairvoyant-bound)
   - [5.7 C4 — compute parity](#57-c4--compute-parity)
   - [5.8 C5 — Census/BBMP ward crosswalk](#58-c5--censusbbmp-ward-crosswalk)
6. [Literature novelty check (P0.1)](#6-literature-novelty-check-p01)
7. [Limitations](#7-limitations)
8. [Reproducibility](#8-reproducibility)
   - [Test coverage](#test-coverage)
   - [Analysis scripts](#analysis-scripts)
   - [Running the confirmatory study](#running-the-confirmatory-study)
   - [Fetching real data](#fetching-real-data)
9. [Silent-exclusion audit](#9-silent-exclusion-audit)

---

## 1. Thesis and motivation

Published comparisons of ride-hailing dispatch policies typically report a
single point estimate of one policy's improvement over another — 1-10%
gains on simulators calibrated from partial data — without propagating the
uncertainty in the *input models* that calibration produced. This project
builds a small, honest digital twin (an event-driven dispatch simulator)
plus a bootstrap-over-input-models evaluation harness, and asks: for a
standard ladder of five dispatch mechanisms, do policy rankings actually
survive that propagation? Where they don't, the harness is built to report
the *indifference set* — the policies that can't be statistically
separated — rather than declare a winner.

The method: resample the underlying trip data with replacement, refit every
input model (arrival process, origin-destination distribution, travel-time
distribution) on each resample, re-run every policy under common random
numbers (CRN) so the comparison is paired, and report the distribution of
outcomes across resamples — not a single number.

## 2. Data sources and status

**Amended 2026-08-04**, before any real-data validation was attempted (not
after one failed — see §5.5 for why that distinction matters and is
verifiable from timestamps). The original plan treated Bengaluru's Namma
Yatri aggregates as the primary validation source; that turned out
untenable, not just imperfect: those aggregates have **no
`request_datetime`/`pickup_datetime` split at all**, so wait time — this
project's central metric — is not observable from that source under any
circumstance. Roles restructured accordingly:

- **NYC TLC data — the primary methodological study.** Real, schema-verified
  against a live scan of the actual parquet files (`src/dispatch_eval/sources/nyc_tlc.py`),
  with real trip-level timestamps — the only source here that can actually
  falsify a wait-time validation threshold. `hvfhs_license_num` distinguishes
  operators (HV0003 = Uber); no authentication needed, fetched from
  `https://d37ci6vzurychx.cloudfront.net/trip-data/`.
- **Bengaluru — Namma Yatri Open Data — an applicability study, not a
  validation source.** Real, live, ward-level aggregates only. Can't
  support wait-time validation, but is exactly the right target for
  asking "what can honestly be concluded from ward-level Indian open
  mobility data alone" — answered in §5.4 without needing the data itself,
  and a genuine, documented attempt to obtain the actual data is in §7.
- **Delhi NCR — a Kaggle ride-booking dataset — dropped from every
  result.** Trip-level shape, but unverified, plausibly-synthetic
  provenance; presenting it as real would undercut a project about
  statistical honesty. The schema adapter (`sources/delhi_ncr.py`) stays
  as a working, tested capability, unused for any claimed result.

Everything not explicitly run against real NYC data (§5.5) runs against
`dispatch_eval.sources.synthetic`, a generator that produces schema-valid
trip records from a chosen, arbitrary ground truth — useful for building
and testing the harness and for real methodology development (it found and
fixed real bugs, §9), but not itself a claim about any real city.

### Validating the synthetic generator's *shape*, not its numbers

Before any P1 validation existed, the synthetic generator's plausibility
was checked against real NYC data in a different sense: not "does the
simulator's dispatch mechanism reproduce real wait times" (§5.5), but "is
the *shape* of the generator's assumptions realistic at all" — trip
distance skew, fare structure, driver-pay fraction, and demand's
hour-of-day shape (`analysis/nyc_reference_comparison.py`).

The first pass at this got something backwards: comparing to NYC and then
recommending the generator's parameters move *toward* NYC's numbers
defeats the point of using NYC only as a real-data-availability
convenience — the project's actual target is India. The rule that
survived a second pass: a *distributional form* (e.g. "trip distance
should be right-skewed") generalizes across cities and is fair to borrow
from anywhere; a *magnitude or structural ratio* (e.g. how heavily fare
depends on distance vs. duration) does not, and should only change when
grounded in a real India-specific fact.

What that produced, once real Bengaluru anchors replaced NYC's:

- **Trip distance**: derived from each trip's simulated duration through
  Bengaluru's real average traffic speed (TomTom Traffic Index, ~18-22
  km/h, among the world's slowest), with lognormal noise. Skewness moved
  from 0.18 to 1.6.
- **Fare structure**: anchored to BBMP's real regulated auto-rickshaw
  meter (₹36 first 2km, then ₹18/km, +50% surcharge 10pm-5am, effective
  Aug 2025) instead of NYC's TLC formula. Distance:duration weighting went
  from 7.5x to **36.7x** — the *opposite* direction from "move toward
  NYC's 3.1x," because Indian auto meters have essentially no continuous
  per-minute charge.
- **Driver pay fraction**: 0.75 → **1.0**, matching Namma Yatri's stated
  zero-commission policy. The old 0.75 happened to sit close to NYC's real
  median (0.72) — a coincidence the first pass mistook for reassurance,
  masking a structurally different commission model.
- **Nightlife demand bump**: peaks ~1am (BBMP raised bar/club closing
  hours to 1am in 2024), not NYC's midnight shape. Net effect: hourly-shape
  similarity to NYC *dropped* (96.4% → 93.6%) — expected, since resembling
  NYC was never the goal.
- **Left alone, on purpose**: OD/zone concentration — no real Bengaluru
  hub data exists to anchor a number, so the arbitrary near-uniform
  default stays arbitrary rather than borrowing NYC's hub geography.
- A supporting finding, not a generator bug: only 13% of NYC's own
  zone×zone×hour cells have ≥5 trips (covering 48% of trips) — even
  454,813 trips in one day isn't enough to densely populate a
  fine-grained OD-time cube. Evidence for this project's actual thesis
  (input identifiability is hard even with abundant real data), not
  something to fix.

## 3. System architecture

```
src/dispatch_eval/
  schema.py                canonical TRIP_RECORD / WARD_AGGREGATE shapes + validation
  models.py                 runtime input models: NHPPArrivalModel, ODModel,
                             TravelTimeModel, FareModel, AbandonmentModel
  scenario.py               pre-generates the exogenous request scenario
                             (arrivals, destinations, patience) once, upfront
                             — the fix for a real CRN bug (below)
  simulator/
    entities.py              Vehicle, Request, and their state enums
    events.py                Event + EventType for the heapq event queue
    engine.py                SimulationEngine: the event loop, driven by a
                             pre-built Scenario rather than generating
                             requests reactively
    runner.py                wires fitted models + a policy + fleet size
                             into one run_simulation() call
  policies/
    base.py                  DispatchPolicy protocol (same shape for B0-B5),
                             optional RepositioningPolicy protocol (B3+)
    nearest_idle.py            B0 — nearest idle vehicle, greedy, no batching
    batched_hungarian.py        B1 — batched optimal assignment (scipy Hungarian),
                             tunable batch window and matching radius
    value_corrected_hungarian.py B2 — B1's assignment, cost matrix adds a
                             value-function correction so a policy avoids
                             stranding a vehicle in a low-demand zone
    fluid_zone_balancing.py     B3 — wraps any DispatchPolicy unchanged, adds
                             reposition(): idle vehicles move toward a
                             fluid target allocation (idle-vehicle share per
                             zone proportional to that zone's demand share)
    sampling_lookahead.py        B4 — same wrapping pattern as B3, but
                             reposition() samples one realization of
                             near-future requests from the arrival model and
                             Hungarian-matches idle vehicles to them directly
    zone_index.py              exact zone-level reductions every policy uses
                             (see "Scaling to NYC" below)
    registry.py                readable policy names, ladder construction,
                             equal-budget tuning of every tunable rung
  calibration/
    arrivals.py               fit NHPP rates per (zone, day_type, bin);
                             index-of-dispersion diagnostic;
                             arrival_sparsity_report (§9)
    od.py                      fit smoothed (Dirichlet) OD distribution per
                             (origin, time bin)
    travel_time.py             fit lognormal travel time per (origin, dest, hour)
    fare.py                    fit linear fare ~ distance + duration
    boarding.py                empirical on-scene → rider-aboard delay (NYC only)
    pickup.py                  joint grid calibration of the two latent supply
                             parameters: fleet size × same-zone pickup time
    fleet_size.py               calibrate the one latent quantity that most
                             matters: fleet size, matched to observed
                             wait-time median/p90 (absolute_moment_loss) or
                             the P90/median ratio alone (ratio_loss, §5.5);
                             plus hit_boundary convergence diagnostic (§9)
                             and a KS-distance helper for validation
    tuning.py                   fixed-budget random-search harness so every
                             policy in the ladder gets an identical tuning
                             budget (the P2 parity condition)
    value_function.py            backward-induction DP for B2: C(zone, time_bin)
                             = expected future idle time from being
                             positioned there, over the fitted
                             arrival/OD/travel-time models
  sources/
    synthetic.py               ground-truth generator + simulator-backed
                             trip generation, for development and tests
    nyc_tlc.py                  adapter for NYC TLC HVFHS data — schema
                             verified against a live scan, this project's
                             one real-data source
    delhi_ncr.py                adapter for the Kaggle Delhi NCR CSV
                             (schema unverified, unused for any result)
    bengaluru.py                 adapter for Namma Yatri's ward aggregates
                             (schema unverified, unused for any result)
  clairvoyant.py               B5 — offline upper bound, not a DispatchPolicy:
                             time-expanded min-cost flow over (zone, time
                             bin) nodes, since one vehicle can serve many
                             requests in sequence and that's a flow, not a
                             one-shot assignment
  ranking_flip.py               P3 — the ranking-flip experiment: bootstrap
                             resampling, refitting all input models per
                             draw, the B x R x |policies| CRN loop, and all
                             of P3's outputs — P(ranked first), Kendall's
                             tau to nominal, C1's variance decomposition
                             (two-term and three-term, §5.3), the
                             indifference set, and the
                             minimum-detectable-effect curve
  decision_currency.py           C3 — express a policy's per-draw wait-time
                             gain (reusing ranking_flip's own bootstrap
                             output, no extra simulation) as "worth N
                             vehicles" relative to a reference policy's
                             fleet-size curve, plus Δdriver-hours (exact)
                             and Δ$/day (a documented flat-fare approximation)
  compute_parity.py              C4 — wall-clock dispatch()/reposition()
                             latency as a function of open requests and
                             idle vehicles, and which problem sizes fit a
                             given real-time budget
  coarsening_ladder.py            C6 — re-runs the exact bootstrap-CRN
                             experiment on progressively coarsened trip
                             data: coarser time/OD binning, fare/distance
                             collapsed to route means, destinations
                             shuffled to destroy OD structure, zone
                             aggregation to ward-equivalents, and stripping
                             the request/pickup timestamp split — reports
                             the ranking shift at each rung
  studies/                      the NYC study, driven by configs/nyc.yaml
    nyc_data.py                fetch/cache/split the TLC window, clock alignment
    validation.py              calibrate on 10 days, validate on 4 held-out days
    policy_study.py            tune + bootstrap-compare the ladder per supply
                             regime, price gains in vehicles
    eda.py                      exploratory tables of the real data
    report.py                   every figure, rebuilt from results/ only
  cli.py                        `dispatch-eval validate|study|eda|report`
  tracking.py                   W&B runs (DISPATCH_WANDB=0 disables)
  geo/
    crosswalk.py                C5 prep (geopandas/shapely/pyproj are dev
                             dependencies, not simulator runtime deps):
                             areal_interpolate (sums extensive/count
                             variables) and areal_interpolate_weighted_average
                             (area-weighted averages for intensive/rate
                             variables), plus a validator against an
                             authoritative reference
  forecast_degradation.py        decouples "models the true simulated world
                             runs on" from "models a policy believes" — four
                             functions each collapse a fitted model to a
                             deliberately wrong structural family (or fit
                             on fewer days), and run_degradation_sweep
                             re-runs the bootstrap experiment once per
                             degradation level while the true world stays
                             correctly specified (§5.2)
```

At the project root: `run_study.py` — the single entrypoint that runs P1
calibration, P2 tuning, and the P3 bootstrap-CRN experiment (B5 wired in)
end to end from one command, writing every
`(policy, bootstrap_draw, replication) → metrics` cell to one parquet file
rather than just a summary. `run_pipeline()` is factored out for reuse by
other scripts (e.g. `analysis/decision_currency_run.py`) that need the
live in-memory result, not just the parquet.

### Scaling to NYC

The synthetic study ran 5 zones and ~40 vehicles; NYC is 69 zones, ~4,000-
10,000 vehicles and ~38,000 requests per simulated 6-hour window. Three
exact reductions (verified against brute force in `tests/test_zone_index.py`)
make that tractable without changing any policy's decisions:

- **Vehicles in one zone are interchangeable** — every cost depends on a
  vehicle only through its zone. Greedy dispatch searches zones, not
  vehicles (O(R·Z) instead of O(R·V); identical picks, ties included).
- **Batched matching keeps at most R candidate vehicles per zone** — an
  assignment of R requests can never use more — and solves a rectangular
  instead of a padded V×V assignment problem.
- **Rebalancing is a zone-level transportation LP** (HiGHS) instead of a
  vehicle-by-slot assignment; the transportation matrix is totally
  unimodular, so the LP optimum is integral.

Plus memoized expected travel times, an ordered waiting set in the engine,
a configurable rebalancing interval (60s for NYC), and process-parallel
bootstrap draws that reproduce serial results bit-for-bit. Net effect on a
one-hour NYC run of greedy dispatch: 7.4s → 0.9s; a full 6-hour run of any
policy takes 4-12s.

### Design choices worth knowing about

- **One canonical schema, many adapters.** The simulator, calibration
  code, and tests only ever see `TRIP_RECORD_SCHEMA` or
  `WARD_AGGREGATE_SCHEMA` (`schema.py`). Real-source adapters and the
  synthetic generator are the only things that know about a specific data
  source's raw shape.
- **Decision vs. realization are separate calls.** `TravelTimeModel.expected()`
  is what a dispatch policy uses to decide who's "nearest";
  `.sample()` is what actually happens when a trip runs. Conflating them
  would make the variance in the fitted travel-time model pointless.
- **Fleet size and the abandonment hazard are never both fit from data.**
  They aren't jointly identified from matched-trip data alone. Fleet size
  is calibrated against the observed wait-time distribution; the
  abandonment hazard is a structural axis swept across specifications
  (§5.3), not estimated.
- **The synthetic generator hides exactly what real data would hide.**
  Abandoned requests come out of `generate_synthetic_trips` folded into
  `"cancelled_customer"`, matching the same censoring every real source in
  this project has.
- **Assignment-cost terms constant across a row or column get dropped, not
  computed.** B2's value correction only ever adds `C` at the *candidate*
  destination, never at the vehicle's own current position — that term is
  the same for every entry in a vehicle's column, so it can never change
  which column a row prefers (a standard property of the assignment
  problem). Same reasoning is why B1's radius cutoff is a hard exclusion
  rather than a soft penalty term.
- **A real CRN bug, fixed.** `SimulationEngine` used to draw arrivals,
  destinations, and patience reactively during the event loop, interleaved
  with whatever travel-time samples a policy's dispatch decisions also
  drew from the same stream — so two policies run with "the same seed" did
  not actually see the same requests, silently breaking every paired
  comparison. Fixed by pre-generating the whole request scenario upfront
  (`scenario.py`), in an order depending only on the input models and zone
  list, never on policy behavior. Verified directly, not just asserted:
  two identically-behaved policies produce *exactly* identical metrics at
  every single (bootstrap draw, replication) cell in the test suite.

## 4. Methodology

### The baseline ladder

Each policy has a readable name (used by the NYC study, the CLI and every
figure) and a B-code (used by the original synthetic-study outputs);
`policies/registry.py` maps between them.

| Policy | Name | Mechanism |
|---|---|---|
| B0 — reactive nearest-idle | `greedy` | Greedy: each request assigned to the nearest available idle vehicle, no batching. |
| B1 — batched Hungarian | `batched` | Globally-optimal bipartite assignment (`scipy.optimize.linear_sum_assignment`) over all waiting requests and idle vehicles within a matching radius, at each dispatch tick. |
| B2 — value-corrected | `value_aware` | B1's assignment cost augmented with a backward-induction value function `C(zone, time_bin)` — expected discounted future idle cost — at the destination; `value_weight=0` recovers B1 exactly. |
| B3 — fluid zone balancing | `fluid_rebalance` | B2's dispatch, plus a distinct repositioning mechanism: idle vehicles moved toward each zone's proportional share of expected arrival rate, via the same bipartite-assignment machinery. |
| B4 — sampling lookahead | `lookahead_rebalance` | B2's dispatch, plus repositioning informed by a one-sample Monte Carlo lookahead over near-future demand. |
| B5 — clairvoyant bound | `oracle_bound` | Not a deployable policy: an offline, perfect-information upper bound (time-expanded min-cost flow) used to normalize the other four's performance as "fraction of the achievable gap closed," not an absolute number. Uses a pessimistic-pinning approximation (a request's exit point is pinned to its patience deadline, not its true solution-dependent pickup time) — see the module docstring; confirmed to matter via a two-request-chain regression test. |

Every tunable hyperparameter (matching radius, value weight, lookahead
window, dispatch interval) was given an identical, fixed random-search
budget (`analysis/policy_tuning_run.py`, 25 evaluations each, same
seed-per-candidate protocol) — the parity condition ensuring "policy X
beats policy Y" cannot secretly mean "policy X got a better search."

### Uncertainty propagation (P3)

For B ≈ 40-500 bootstrap draws: resample the trip data with replacement,
refit every input model on the resample, run every policy in the ladder
under R common-random-numbers replications against that draw's fitted
models, and record the full `(policy, bootstrap draw, replication) →
metric` table — not summary statistics alone. From that table:

- The **nominal ranking** — fit on the un-resampled data.
- **P(policy ranked first)** across bootstrap draws.
- **Kendall's tau** between each draw's ranking and the nominal one.
- The **indifference set** — every policy statistically indistinguishable
  from the best at α=0.05 via a paired bootstrap-percentile interval on
  the per-draw metric difference (no multiple-comparisons correction
  across the simultaneous checks — a documented first-cut simplification,
  reasonable with the ladder's 5-6 policies).
- **C1's variance decomposition** — see §5.3.
- The **minimum-detectable-effect (MDE) curve** — extrapolates the
  smallest true effect whose sign is resolved at a given confidence, from
  one measured variance decomposition, assuming input-model estimation
  error scales as 1/n_days (checked empirically, not just assumed — see
  §7).

Pre-registered before any real-data validation was attempted
(2026-08-04): primary metric (mean wait time per completed request),
indifference-set α (0.05), bootstrap-budget minimum (`n_bootstrap ≥ 40`,
checked directly — ranking and indifference set were identical at B=40
and B=200, though the variance decomposition itself moved between scales,
0.314 → 0.239, so report that one off the full B=200 only), and the P1
acceptance thresholds used in §5.5.

## 5. Results

### 5.1 No indifference between mechanisms — until there is

Four full bootstrap-CRN experiments, across fleet sizes {15, 30, 60}
(`analysis/fleet_size_sensitivity_run.py`, `analysis/tuned_ranking_flip_run.py`)
and the closest-possible pair (B3 vs. B4, both wrapping identical dispatch
logic, differing only in repositioning heuristic —
`analysis/b3_vs_b4_run.py`, B=50): B2 (or B4, in the B3-vs-B4 case) wins
**100% of bootstrap draws every time**, indifference set a singleton at
α=0.05 in all four. The B3-vs-B4 comparison's own variance decomposition
shows intrinsic simulation noise *exceeding* input-model-estimation
variance (`within_theta_variance=102.2`, `across_theta_variance=27.6`) —
the ranking still never flips a single draw, meaning the mechanism gap is
large enough to swamp both noise sources at once, not narrowly surviving
one.

**Indifference does appear** — inside one mechanism's own tuning dial, not
between mechanisms. B2 swept across `value_weight ∈ {0.0, 0.5, 0.75, 1.0,
1.5}` (`analysis/value_weight_sweep_run.py`, B=30, matching radius held
fixed at B2's own tuned value): `weight_0.0` (no value correction) is
clearly worse and excluded, but `{0.5, 0.75, 1.0, 1.5}` — a 3x range
including the independently-tuned value — are **all** in the
indifference set, `P(ranked first)` split roughly evenly (`0.5: 0.23,
0.75: 0.20, 1.0: 0.30, 1.5: 0.27`). This is the pair used in §5.4's
coarsening ladder.

For context, real published applied dispatch papers report margins in
this project's own indifference band: the closest applied comparison
found in the P0.1 novelty check (§6) reports point-estimate mean waits of
82.3s vs. 85.3s vs. 85.8s across three of its own retrieval/calibration
variants — margins comparable to the 3x `value_weight` range this project
finds statistically indistinguishable.

### 5.2 The oracle-forecast confound, ruled out

Every result in §5.1 let B2-B4 forecast the world through models fit the
*same way* as the model generating the true simulated world — an oracle
advantage no real deployment has. `forecast_degradation.py`
(`analysis/forecast_degradation_sweep.py`) breaks this: the true world
stays fixed at a correctly-specified, bootstrap-varying fit, while each
policy's own *belief* (value function, reposition reference, per-tick
travel-time estimate, via a new `policy_travel_time_model` split in the
engine) is built from a deliberately degraded model — nine levels: the
true forecast as baseline; three data-limited levels (fit on only 1, 3, or
5 of 10 days); each of four structural degradations in isolation
(homogeneous arrival, spatially-uniform arrival — strictly more severe,
destroys *which zone is busier* too — marginal OD, flat travel time); and
a "fully degraded" level combining data-limiting with every structural
degradation at once.

**B2 wins 100% of bootstrap draws at every level, including the most
severe.** This is the opposite of the working hypothesis (that oracle
forecasting inflated the apparent gap), reported as found, not reframed.

A follow-up mechanism diagnostic
(`analysis/b2_mechanism_diagnostic.py`) resolves *why*, via two checks at
six of the nine levels:

1. **`var_across_zones(C[:, t])`** — variance across zones of the value
   function's cost-to-go, per time bin. Exactly **0.0** at
   `spatially_uniform_arrival_only` and `fully_degraded` — confirming the
   degradation genuinely destroys all spatial signal, not approximately.
2. **`B1_radius_from_B2`** — B1 with B2's tuned radius (154s vs. B1's own
   205s) and `value_weight=0` (mechanically identical to B1's own cost
   function otherwise), added as a third bootstrap-CRN arm at every
   level (safe to add — CRN is keyed off `(seed, draw, replication)` per
   policy, not policy count, so this can't change the other arms'
   results). Ranks **dead last at every level tested**, `P(ranked
   first) = 0.0` throughout, never once in the indifference set.

**One-sentence answer**: B2's surviving advantage is not its matching
radius (that alone makes things worse) and not stale tuning (retuning B1
and B2 fresh per degradation level doesn't close the gap either — B2
retuned still beats B1 by 25-45s at every level) — it's the value
function's temporal, finite-horizon decay structure, which keeps
rewarding "get this vehicle back into service sooner" even after every
trace of spatial forecast content has been deliberately destroyed. This
is a real, mechanistic finding about *this* value function's construction
(a mean-field backward induction with a hard terminal boundary at the
horizon), not a claim that no dispatch policy could ever derive a genuine
advantage from spatial forecasting.

### 5.3 C1 — the real three-term variance decomposition

`variance_decomposition` (the two-term version, used throughout §5.1-5.2)
splits `Var(y_a - y_b)` into within-theta (intrinsic simulation noise) and
across-theta (input-model estimation error) via the law of total variance
over the nested (bootstrap draw, replication) design. This was always a
**floor** on input uncertainty's true contribution: the bootstrap loop
only ever refits within one fixed, correctly-specified model family, so
structural misspecification is exactly zero by construction.

`variance_decomposition_three_term` (`analysis/abandonment_sweep_run.py`)
closes that gap by folding the pre-registered abandonment-hazard sweep
(`mean_patience_seconds ∈ {60, 180, 300, 600, 900}` — five specifications
already treated as structural, not fit, since fleet size and the
abandonment hazard are not jointly identified from matched-trip data
alone) in as the outer loop around the existing bootstrap. Tracked pair:
B2 vs. B0, fixed across every spec so the paired difference means the
same thing at each one.

| Term | Variance | Ratio to intrinsic |
|---|---|---|
| Intrinsic (within spec, within draw) | 2,464.30 | 1.0 |
| Input-model estimation (across draws, within spec) | 519.58 | 0.211 |
| **Structural (across abandonment specs)** | **62,412.37** | **25.327** |

**Structural uncertainty dominates**, by roughly two orders of magnitude
over input-model estimation error. This reframes C1's headline claim from
generic ("input uncertainty matters more than a point estimate suggests")
to specific and actionable: structural ambiguity about a latent
behavioral parameter (abandonment patience) matters far more than
sampling-driven input-model estimation error, which says *where* to spend
measurement effort. B2 remains the winner at every one of the five
specifications (the pre-registration's own robustness criterion is
satisfied), but B1's *position* is not robust: runner-up at patience ≤
300s, drops behind both B3 and B4 at patience ≥ 600s — a real,
structural-axis-driven ranking shift within the ladder.

Scope note: this covers exactly one structural axis — the abandonment
hazard. Model-family misspecification (§5.2) is a separate structural
question, checked separately, and not folded into this number; a fully
general structural term covering every possible axis at once isn't
attempted.

### 5.4 C6 — the coarsening ladder, and where it breaks

Re-running the exact bootstrap-CRN comparison on progressively coarsened
versions of the same trip data (`coarsening_ladder.py`): coarser
time/OD binning; fare/distance collapsed to (origin, dest)-pair means;
destinations shuffled to destroy true OD structure while preserving
marginal popularity; zone aggregation via a many-to-one map; and stripping
the request/pickup timestamp split down to one trip-start timestamp on
completed trips only (matching what real published ward-aggregate data
actually offers).

**On a large-effect pair (B0 vs. B1,** `analysis/coarsening_ladder_sweep.py`**)**:
survives every rung trivially, `P(ranked first) = 1.0` throughout —
expected, not a strong test, since this pair sits well outside the
indifference zone even at full resolution.

**On the genuinely close pair from §5.1** (`weight_0.5` vs. `weight_1.5`,
`analysis/coarsening_ladder_close_pair_run.py` for rungs 0-3,
`analysis/c6_bengaluru_grade_run.py` for the 5th rung — which needed its
own ward-level value function, refit from the same coarsened data the
rung's bootstrap draws use, since a policy carrying a value function baked
in at construction time would otherwise silently degenerate once the
zones get relabeled underneath it):

| Rung | Nominal ranking | P(weight_1.5 ranked first) |
|---|---|---|
| 0 — fine-grained | weight_0.5 first | 0.65 |
| 1 — 15-min rounding | weight_0.5 first | 0.70 |
| 2 — ward-level OD | weight_0.5 first | 0.70 |
| 3 — ward-only aggregate, no timestamps | weight_0.5 first | 0.60 |
| **4 — Namma-Yatri-grade (ward-equivalent zones + no request/pickup split)** | weight_0.5 first | **0.40** |

Two findings, neither hypothetical:

1. **The point-estimate-vs-bootstrap-majority disagreement.** At rungs
   0-3, the single nominal (un-resampled) fit picks `weight_0.5`, but the
   bootstrap *majority* picks `weight_1.5` (60-70% of draws) — the exact
   failure mode this project's "report the distribution, not a point
   estimate" methodology exists to catch, caught for real in this
   project's own results, not hypothetically.
2. **The ranking flips at the most severe, most realistic rung.** Rung 4
   — built specifically to match real published ward-aggregate
   resolution — is the *only* rung, across three separate coarsening
   runs, where a ranking actually moves rather than just narrowing or
   widening within the same winner. The bootstrap majority swings to
   agree with the nominal fit (`weight_0.5`, 60%). A plausible mechanism,
   consistent with §5.2: destroying both spatial resolution (5 zones → 3
   wards) *and* collapsing time to a single daily bin at once squeezes
   both axes B2's value function could exploit, consistent with the
   higher-`value_weight` variant losing its edge — not directly isolated
   by this run.

Takeaway for reporting: the earlier four rungs support "this close pair's
already-thin margin doesn't obviously erode further under moderate
coarsening" — that statement needed qualifying, not retracting. It holds
up through rung 3, but not at Namma Yatri's actual real-world resolution.

### 5.5 P1 — validation against real data

Against pre-registered thresholds (committed 2026-08-04, before any
real-data attempt): 14 January-2024 Manhattan weekdays, HV0003 (Uber)
only, 12:00-18:00 window, first 10 days for calibration, last 4 held out
(`analysis/nyc_p1_validation.py`).

| Check | Threshold | Measured | Result |
|---|---|---|---|
| Wait-time KS distance | ≤ 0.10 | 0.7355 | **FAIL** |
| Hour-of-day cosine similarity | ≥ 0.90 | 0.9865 | PASS |

**Not adjusted after seeing the result.** The arrival-timing *shape* is
well captured; the wait-time *distribution* is not, and not marginally —
at the loss-minimizing, properly-converged fleet size (4,500; a first
attempt hit the search boundary at 2,000, caught by the `hit_boundary`
diagnostic — see §9), simulated median wait is 65s against a real 187s.

**Two real bugs found and fixed along the way, before this result was
even reachable**:

- A genuine clock-alignment bug: this codebase's simulated clock always
  starts at midnight (`SimulationEngine._hour_of` wraps to `(time //
  3600) % 24`; `generate_scenario` samples arrivals over `[0,
  horizon_minutes)`), but every fitted model bins by each timestamp's
  *absolute* hour-of-day. Fitting directly against real 12:00-18:00
  timestamps would have put every fitted rate in hour-of-day bins the
  simulated clock never visits — arrivals would have silently generated
  at rate 0 everywhere. Fixed by shifting timestamps so window-start
  (12:00) maps to simulated midnight before fitting.
- The fleet-size search boundary bug (§9), which this real run re-caught
  for real, not just in a unit test — widened the candidate range from
  `[100..2000]` to `[100..12500]` until it actually converged.

**A follow-up rules out the most tempting explanation**: that the
calibration objective was asking for two incompatible things (absolute
scale and distributional shape) and settling for a bad compromise on
both. Recalibrating fleet size against the P90/median *ratio* alone
(`ratio_loss`, ignoring absolute scale — `analysis/nyc_p1_ratio_calibration_check.py`)
picks the identical fleet size (4,500), and at that fleet size the
simulated ratio (1.979) already matches the real one (1.963-1.978) almost
exactly — yet KS distance is unchanged to four decimal places, because a
correctly-*shaped* distribution sitting at the wrong absolute *location*
still fails a KS test exactly as hard. The mismatch is a genuine ~3x
*scale* gap in the reactive baseline's dispatch mechanism
(`NearestIdlePolicy`'s unconstrained, no-batching-radius,
no-acceptance-friction matching — real Uber dispatch runs proprietary
batching, ETA prediction, and driver-side acceptance, producing a
tighter, higher distribution), not a calibration artifact, and not
closeable by adjusting fleet size at all.

**Consequence**: no real-NYC confirmatory study (P3 at scale, C3) was
attempted. Per the pre-registration's own risk table ("if validation
fails, write it up as the primary finding; C2 becomes secondary"),
running a confirmatory study against a twin that just failed its own
validation and presenting the output as real-world-grounded would be
exactly the failure mode this project's methodology exists to prevent.
Every synthetic-data result in this report describes this simulator's own
dynamics; it is not yet demonstrated to describe any real market's.

#### Validation V2 — pre-specification (2026-09-28, before the full run)

A re-examination of the V1 failure found a different root cause from the
one stated above. `TravelTimeModel` hard-coded every same-zone move to
~60s, and at a calibrated fleet of 4,500 nearly every pickup is same-zone,
so simulated waits clustered at ~65s whatever the fleet size. Separately,
the real wait (median 180s) has two observable parts that NYC reports via
`on_scene_datetime`: request → driver on scene (median 116s) and on scene
→ rider aboard (median 45s). The simulator modelled no boarding at all.

V2 changes the model, not the test:

- **Boarding** — fitted directly from the calibration days
  (`pickup − on_scene`, empirical distribution), fixed per request in the
  scenario like patience.
- **Same-zone pickup time** — a latent lognormal, median calibrated
  jointly with fleet size by grid search on the calibration days only
  (KS to the calibration days' waits); sigma taken from real same-zone
  trip durations. Grid: fleet {1500…10000} × median {45…300}s
  (`configs/nyc.yaml`).
- Everything else unchanged: same scope, same calibration/held-out split,
  same nearest-idle reference policy, same patience (300s), same
  thresholds (KS ≤ 0.10, cosine ≥ 0.90), evaluated once on the held-out
  days.

Disclosure: a 2×2 smoke test of the pipeline (fleet {3000, 4000} × median
{90, 150}s, one replication) ran before this note was written and gave a
held-out KS of 0.097. The grid and thresholds above were already fixed in
`configs/nyc.yaml` at that point and are not changed in response to it.

#### Validation V2 — result

`uv run dispatch-eval validate` (W&B run `nyc-validation`; output
`results/nyc_validation.json`). 80-point supply grid on the calibration
days, then 4 replications at the best point against the 4 held-out days
(178,948 real trips vs 138,476 simulated completions):

| Check | Threshold | V1 | **V2** |
|---|---|---|---|
| Wait-time KS distance | ≤ 0.10 | 0.7355 FAIL | **0.0308 PASS** |
| Hour-of-day cosine similarity | ≥ 0.90 | 0.9865 PASS | **0.9862 PASS** |
| Approach-time KS (diagnostic, not pre-registered) | — | — | 0.0763 |

| Wait quantile | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| Real, held-out | 92s | 133s | 187s | 264s | 367s |
| Simulated | 87s | 128s | 189s | 268s | 379s |

The residual mismatch sits in the fastest pickups: real approach times
have a longer short tail (p10 36s vs 53s simulated), i.e. drivers who are
already metres from the rider, which a zone-level model with a single
same-zone travel distribution cannot represent.

**What the calibration can and cannot identify.** The best grid point,
(10,000 vehicles, 120s), lies on the grid's upper edge, and the KS
surface is flat above ~6,500 vehicles (0.058 → 0.052 → 0.050 at 6,500 /
8,000 / 10,000). Extending the grid on the calibration days only
(`analysis/nyc/calibration_ridge_check.py`; held-out data untouched) gives
0.048 at both 12,500 and 16,000. Once supply is plentiful nearly every
pickup comes from inside the rider's own zone, so waits stop depending on
fleet size: **wait-time data bounds the fleet from below (every fleet of
~4,000+ fits within KS 0.10; 3,500 does not) but not from above.** A
Little's-law count of cars committed to riders (`dispatch-eval eda`)
gives a floor of ~3,100 busy at the 17:00 peak, consistent with that lower
bound. The policy study (§5.9) therefore treats fleet size as a latent
parameter and spans the data-consistent range rather than trusting one
point.

### 5.6 C3 — decision currency and B5's clairvoyant bound

Run end-to-end against the synthetic confirmatory study at the
pre-registered target scale (B=200, R=5,
`analysis/decision_currency_run.py`, reusing `run_study.run_pipeline`
in-memory rather than reading the parquet back, since C3 needs the live
`RankingFlipResult`/`FittedModels` a flattened table doesn't carry) — the
real-data version is withheld per §5.5.

B5's clairvoyant (offline, perfect-information) bound, after fixing two
real bugs found in its own construction:

- An overly pessimistic exit-pin on vehicle availability (fixed into a
  genuine earliest-feasible-release relaxation).
- A discretization bin width too coarse for the pre-registered
  abandonment specification (`mean_patience_seconds=300.0` is a third of
  one 15-minute bin), which silently excluded 67.5% of requests as
  "unservable" before the solver ever saw them (fixed by using a finer
  bin; made visible going forward via
  `ClairvoyantResult.fraction_excluded_by_discretization`).

Both fixes together, re-measured at B=200: B1/B3/B4 close 90-93% of the
clairvoyant gap, B2 closes 95%, all properly below 1.0 (a pre-fix run had
every policy "beating" the bound by 5-10%, the tell that something was
wrong), with 9.4% of requests still excluded by discretization at the bin
width used — reported alongside the gap-closed figure per the
pre-registration's own commitment, not hidden.

**C3's headline: B2 is worth +256.8 vehicles relative to B0's own
calibrated fleet (60), 95% confidence interval [+221.7, +309.0]** — i.e.
B0 would need to more than quintuple its fleet to match B2's wait-time
performance at the same fleet size. The self-check (B0 against itself)
centers on zero (+2.2, CI crossing zero), as it must. Δdriver-hours is
exact (`fleet_size * horizon_seconds / 3600`); Δ$/day is a documented flat
average-fare-per-trip approximation, since the simulator's `Request`
doesn't carry a fare.

### 5.7 C4 — compute parity

Real wall-clock latency (`analysis/compute_parity_sweep.py`), not
assumed: every policy in the ladder fits comfortably within a
multi-second production dispatch-cycle budget, even at a synthetic
600×600 (simultaneous open requests × idle vehicles) batch size well
beyond what a real 5-second tick would realistically accumulate.

| policy | 10×10 | 50×50 | 100×100 | 300×300 | 600×600 |
|---|---|---|---|---|---|
| B0 nearest idle | 0.03ms | 0.48ms | 1.78ms | 16.2ms | 64.2ms |
| B1 batched Hungarian | 0.05ms | 0.98ms | 3.9ms | 34.9ms | 142.1ms |
| B2 value-corrected | 0.08ms | 1.7ms | 6.8ms | 61.8ms | 248.0ms |
| B3 fluid balancing | 0.10ms | 1.8ms | 7.1ms | 62.7ms | 253.0ms |
| B4 sampling lookahead | 0.18ms | 2.2ms | 7.5ms | 62.9ms | 251.8ms |

The theoretically-expected crossover (cubic Hungarian-solver scaling
eventually costing optimal-in-principle methods their latency advantage)
is real and visible in the scaling trend (B2-B4 cost ~4x what B0 costs at
the same size, all of B1-B4 scale roughly cubically) but doesn't bind at
any realistic batch size against a multi-second budget — it would bind
well before 600×600 against a tighter one (~100ms budget), a checkable
claim this measurement makes concrete rather than asserted.

**What wasn't built**: the plan's "degrade the ones that don't fit and
re-measure quality" half. `matching_radius_seconds` looks like the
obvious knob but isn't one — confirmed empirically
(`tests/test_compute_parity.py::test_matching_radius_does_not_shrink_the_solved_cost_matrix`
spies on the exact matrix shape passed to `scipy`'s solver at two very
different radii and finds it identical): it masks out-of-radius pairs
with a large cost, but the matrix actually solved is always
`max(n_requests, n_idle_vehicles)` square regardless. Sweeping the radius
changes assignment *quality*, not *latency*. A real candidate-graph-
truncation mechanism (e.g. restrict each request to its k nearest idle
vehicles before building the cost matrix) doesn't exist in this codebase
and is out of scope for this pass.

### 5.8 C5 — Census/BBMP ward crosswalk

Real data engineering, kept as a standalone contribution rather than a
paper claim (no equity analysis is built on top of it in this report):
`src/dispatch_eval/geo/crosswalk.py` implements area-weighted
interpolation between incompatible ward boundary vintages —
`areal_interpolate` for extensive/count variables (population, SC/ST,
summed) and `areal_interpolate_weighted_average` for intensive/rate
variables (household amenity percentages, where summing would be
meaningless) — plus a validator that checks an interpolated column
against an authoritative reference.

Applied to Bengaluru's 2011 Census wards → 369-ward GBA (2025) reallocation
(`analysis/ward_crosswalk.py`, `analysis/ward_amenities_crosswalk.py`):
population/SC/ST and household-amenity-rate crosswalks built and
validated against the GBA's own official reallocation, with measured
error rates. Karnataka's literacy/worker-participation Census variables
specifically resisted scripted access (data.gov.in's PCA file is a
client-side SPA) — population/SC/ST and household amenities are already
crosswalked without it.

## 6. Literature novelty check (P0.1)

A systematic check for whether this project's central methodological
claim — combining bootstrap-resampled input-model uncertainty,
common-random-numbers paired simulation, and ranking-flip/indifference-set
reporting, applied to ride-hailing dispatch — already exists in the
literature. Three passes of increasing rigor:

1. The plan's five required search strings against general web search.
2. A targeted 2025-2026 candidate check. One new close-sounding paper
   found — *Data-driven optimization for ride-sourcing vehicle
   dispatching and relocation under demand and travel time uncertainty*
   (Transportation Research Part C, June 2025) — and ruled out: it's
   approximate dynamic programming with functional-data-analysis
   travel-time estimation feeding a *robust optimization* model —
   uncertainty is a distributional input for deriving one better single
   policy, not something bootstrapped and propagated to compare multiple
   policies' rankings.
3. A genuine forward-citation-graph traversal via OpenAlex's public API
   (no authentication needed, unlike Semantic Scholar's Graph API which
   rate-limited every attempt, or ScienceDirect which returned 403):
   every reachable methodological anchor paper in the input-uncertainty /
   ranking-and-selection (R&S) literature, checked by pulling every
   citing work via `filter=cites:<id>` and keyword-matching titles
   (regex word-boundaries, after an initial false positive was caught and
   fixed — a naive substring match for "AMoD" matched inside
   "met**amod**el").

| Anchor paper | Citing papers checked | Matches |
|---|---|---|
| Song, Nelson, Hong (2015) — *Input uncertainty and indifference-zone ranking & selection* | 20 | 0 |
| Barton, Nelson, Xie (2013) — *Quantifying Input Uncertainty via Simulation Confidence Intervals* | 128 | 0 |
| Fan, Hong, Zhang — *Robust Selection of the Best* (2019 journal + 2013 WSC versions) | 45 + 22 | 0 |
| Lam (2022) — *A Cheap Bootstrap Method for Fast Inference* | 5 | 0 |
| Wu, Wang, Zhou (2022) — *Data-Driven Ranking and Selection Under Input Uncertainty* | 20 | 0 |
| Wu, Zhou (2017/2019) — budget allocation / fixed confidence R&S | 12 + 7 | 0 |
| Shi, Gao, Xiao, Chen (2019) — worst-case constrained R&S | 8 | 0 |
| Kumar, Tiwari (2026) — closest applied ride-hailing paper found | 0 forward citations | — |

One correction along the way: an earlier pass conflated two different
papers under one citation — "Data-Driven Ranking and Selection Under
Input Uncertainty" (2022) is by **Wu, Wang, and Zhou**, not Song and
Nelson, who have a similarly-titled but distinct 2015 paper (row 1 above).

**267 citing papers checked across 10 nodes, zero ride-hailing/dispatch
/fleet/mobility-on-demand keyword matches**, supplemented by two
independent full-text searches directly against OpenAlex's index (110 +
7 total results, no genuine match — the closest, an "operation-agnostic
stochastic user equilibrium model for mobility-on-demand networks," is
equilibrium/congestion modeling, not policy comparison under input
uncertainty).

**Coverage caveat, stated plainly**: OpenAlex is a large, real citation
graph, not provably identical in coverage to Scopus/Web of
Science/Scholar, and this is one citation hop, not a full transitive
closure. A reviewer with institutional access re-checking the same
anchors before submission is cheap insurance, not expected to change the
finding.

The gap this leaves: the input-uncertainty/R&S methodology literature is
application-agnostic (general finite-alternative simulation-optimization,
no spatial fleet, no dispatch policy, no zones); the ride-hailing dispatch
literature almost universally treats input models as fixed when comparing
policies, and where it does use "uncertainty" language, it means
distributionally-robust or robust optimization over a single derived
policy, not bootstrap-propagated estimation error compared across
independent policies via CRN.

## 7. Limitations

Stated together, not scattered through the results:

1. **The twin does not currently validate against real wait-time data**
   (§5.5). Every synthetic-data result in this report describes this
   simulator's own dynamics, not demonstrably real market dynamics. The
   arrival-process fit is validated; the dispatch/wait-time mechanism is
   not, and a follow-up check isolated this to a genuine ~3x scale gap in
   the reactive baseline's dispatch mechanism, not a fixable calibration
   artifact.
2. **Bengaluru applicability data was not obtained**, despite a
   substantive, documented attempt. `https://nammayatri.in/open/` is the
   real, official open-data portal; a plain fetch returns only the SPA
   shell. Pulling the page's own ~26MB minified JS bundle
   (`HomeRoute.bs.*.js`) and grepping it directly found real, concrete
   evidence that ward-level data files exist server-side — a switch
   statement keyed on city ID builds paths like
   `funnel_cumulative_ward_new_key.json`, `trends_cumulative_ward_new_key.json.gz`,
   `driver_eda_wards_new_key.json`, separately for Bengaluru, Gulbarga,
   Mysore, Tumkuru, with `_purple`/`_openmarket` variants — but the base
   URL these paths are relative to wasn't recoverable by static analysis
   (traced to a minified local variable reassigned across too many
   closures in a 26MB bundle to resolve by grep alone). Two plausible
   bases were tried directly and ruled out (a `storage.googleapis.com`
   bucket — 403; several same-origin guesses — 404). Two Google Sheets
   referenced in the same bundle were checked directly and are real but
   irrelevant (an internal product roadmap; single-city customer-rating
   aggregates, not ward-level trip data). Two Kaggle mirrors were found
   but need an authenticated account. The §5.4 coarsening-ladder result
   measures the *cost* of this resolution gap without the data itself,
   which is what this report actually needed, but the C5 crosswalk's
   assumed 369-ward GBA scheme remains unverified against Namma Yatri's
   own if that data is ever obtained. The concrete unblock: a browser
   session's Network tab on the live page (a ~30-second check) or a
   Kaggle account.
3. **The MDE curve's 1/n_days scaling assumption is measurably wrong** —
   real across-theta variance shrinks faster than 1/n as n_days drops
   (only 18% of the 1/n-predicted variance at n_days=3, in a 24-day
   synthetic B0-vs-B1 check at n_bootstrap=50), in the conservative
   direction (the curve over-estimates data needed, not under-estimates
   it — its output is `z * sqrt(within + across/n_ratio)`, so a smaller
   true across-theta variance than assumed means the real MDE at small
   n_days is *better* than the curve reports). A component ablation
   (refitting one of the four sub-models — arrival, OD, travel-time, fare
   — at a time from less data, holding the other three at their nominal
   fit) rules out the leading candidate cause: arrival's unsmoothed
   zero-fallback (asymmetric against the OD fit's Dirichlet smoothing) is
   not meaningfully worse than the other three sub-models, which do
   smooth — all four individually reproduce essentially the same
   deviation magnitude (ratios 0.19-0.34 at n_days=3, converging to 1.0
   by n_days=24) as the full joint refit (0.18 at n_days=3). The
   better-supported remaining explanation points at the synthetic
   data-generating process itself — e.g. within-day demand correlation
   making a small bootstrap resample of days less variable across draws
   than a naive i.i.d.-days assumption predicts, for any estimator fit
   from those days — not yet isolated directly (checking within-day vs.
   across-day variance components of the raw synthetic trip counts is the
   natural next step).
4. **Novelty-check coverage is OpenAlex, not institutional Scopus/Scholar**
   (§6) — large and real, not provably identical coverage.
5. **C4's quality-degradation half and any equity analysis on top of the
   C5 crosswalk are out of scope entirely** — not attempted, not claimed.

## 8. Reproducibility

### The NYC study, end to end

```bash
uv sync
uv run dispatch-eval validate          # calibrate + held-out validation  → results/nyc_validation.json (~15 min)
uv run dispatch-eval eda               # exploratory tables                → results/eda/
uv run dispatch-eval study --regime tight --workers 8   # one supply regime → results/nyc_study_tight.{json,parquet}
uv run dispatch-eval report            # every figure + docs/dashboard.html, from results/ only
```

Scope, split, grids, regimes and budgets all live in `configs/nyc.yaml`.
The first run downloads the January 2024 TLC file once and caches the
study window under `data/cache/`.

**Experiment tracking.** Every validation, calibration, EDA and study run
is a Weights & Biases run in project `fleet-dispatch-eval` (config, seeds,
per-draw progress, calibration grids, tuning histories, result tables).
`DISPATCH_WANDB=0` disables tracking; the test suite always runs with it
off.

**Kaggle.** The four study regimes (~1,000 bootstrap cells each) ran as
parallel Kaggle CPU kernels: `kaggle/push.sh` uploads a private dataset
(package wheel, config, cached NYC window, validation result) and pushes
one script kernel per regime; W&B runs offline there, and
`kaggle/pull.sh <regime>` downloads the outputs into `results/` and syncs
the W&B run from a machine with credentials.

### Test coverage

All tests are in `tests/` and pass (`uv run pytest -q`, ~35s). Every test
is synthetic-data-only — nothing depends on real NYC/Bengaluru/Delhi NCR
files being present.

| File | Covers |
|---|---|
| `test_zone_index.py` | The zone-level reductions are exact: greedy picks identical to brute force (ties included); batched matching and the transport LP reach the brute-force optimum. |
| `test_boarding.py` | Boarding fit from `on_scene_ts`, sampling range, wait = approach + boarding in the engine, same-zone pickup parameters. |
| `test_nyc_study.py` | Clock alignment, wait/approach extraction, same-zone sigma, joint supply calibration and its boundary flag. |
| `test_policy_study.py` | Ladder construction, equal tuning budgets, paired summaries, per-draw vehicle-equivalents and clipping. |
| `test_dashboard.py`, `test_tracking.py` | Dashboard payload embedding; tracking disabled is a no-op. |
| `test_models.py` | `generate_arrival_minutes` never produces an arrival outside its requested window. |
| `test_engine.py` | Event-loop correctness, undersupply → abandonment, repositioning state transitions, deterministic replay; the `policy_travel_time_model` split; `unresolved_at_horizon` finite-horizon boundary accounting. |
| `test_scenario.py` | The CRN fix — scenario generation is deterministic, order-independent, full-trace regression. |
| `test_calibration.py` | Every fitted model recovers known ground truth within tolerance; `arrival_sparsity_report`; OD fit never assigns exactly zero probability to any zone; `calibrate_fleet_size`'s `hit_boundary` flag (interior vs. forced-outside-range); `ratio_loss` vs. `absolute_moment_loss` preferring genuinely different fleet sizes on a constructed example. |
| `test_tuning.py` | `random_search` recovers a known optimum, reproducible, respects `minimize`. |
| `test_batched_hungarian.py` | B1 finds the globally optimal assignment where greedy provably wouldn't; radius-cutoff behavior. |
| `test_value_function.py` | Terminal boundary zero, busier zone strictly lower cost-to-go, monotonic decay. |
| `test_value_corrected_hungarian.py` | Cost correction flips a tied pickup-cost choice toward better value; `value_weight=0` recovers B1 exactly. |
| `test_fluid_zone_balancing.py` | Apportionment rounding, zero-demand-zone handling, delegation, end-to-end run. |
| `test_sampling_lookahead.py` | Sampled-lookahead repositioning, edge cases, never repositioning to own zone. |
| `test_clairvoyant.py` | Same-zone service, infeasible-request handling, multi-request chaining, the pinning-approximation regression, discretization-exclusion diagnostic. |
| `test_ranking_flip.py` | The full CRN loop; parallel draws reproduce serial results exactly; secondary metrics; identical policies give identical per-cell metrics; `variance_decomposition` and `variance_decomposition_three_term` recover known variance structure (and agree exactly at a single spec); `indifference_set`; MDE curve; `compute_clairvoyant`'s output. |
| `test_forecast_degradation.py` | Each degrading function's exact arithmetic; `build_policy_ladder` wiring; end-to-end sweep. |
| `test_decision_currency.py` | `FleetWaitCurve.invert` interpolation/extrapolation; `decision_currency`'s sign convention. |
| `test_compute_parity.py` | Latency measurement shapes; the empirical proof `matching_radius_seconds` doesn't shrink the solved matrix. |
| `test_coarsening_ladder.py` | Each coarsening transform's exact effect, including zone aggregation and request/pickup-split stripping; `run_coarsening_ladder`'s `zone_map`/`policies_by_rung` override path. |
| `test_crosswalk.py` | `areal_interpolate`'s area-weighted split/conservation; weighted-average behavior; CRS/empty-input error handling. |
| `test_sources.py` | Every real-source adapter against schema-valid fixtures; required-column enforcement. |

### Analysis scripts

Every `analysis/*.py` script is independently runnable
(`uv run python analysis/<name>.py`) and caches its own output under
`analysis/cache/` (gitignored — regenerated by running the script, never
committed). None require real data except where noted.

| Script | What it produces |
|---|---|
| `policy_tuning_run.py` | Identical-budget random-search tuning for B1-B4; writes `cache/policy_tuning_results.json`, consumed by most other scripts. |
| `tuned_ranking_flip_run.py` | §5.1 run 1 — B0-B4, fleet=30. |
| `fleet_size_sensitivity_run.py` | §5.1 runs 2-3 — same, fleet=15 and 60. |
| `b3_vs_b4_run.py` | §5.1 run 4 — the closest structurally-different pair. |
| `value_weight_sweep_run.py` | §5.1's within-mechanism indifference set. |
| `forecast_degradation_sweep.py` | §5.2 — the 9-level sweep. |
| `b2_mechanism_diagnostic.py` | §5.2 — `var_across_zones`, the radius ablation, retuning check. |
| `abandonment_sweep_run.py` | §5.3 — the abandonment sweep + three-term decomposition. |
| `coarsening_ladder_sweep.py` | §5.4 — B0 vs. B1, large-effect pair, 4 rungs. |
| `coarsening_ladder_close_pair_run.py` | §5.4 — the close pair, 4 rungs. |
| `c6_bengaluru_grade_run.py` | §5.4 — the 5th, ward-grade rung. |
| `nyc_reference_comparison.py` | §2's shape-plausibility check (fetches/caches one day of real NYC data). **Requires network access.** |
| `nyc_p1_validation.py` | §5.5 — validation V1 (superseded by `dispatch-eval validate`, kept as the record of the failed run). **Requires network access**, fetches ~14 days of NYC TLC data. |
| `nyc_p1_ratio_calibration_check.py` | §5.5's V1 ratio-loss follow-up (reuses `nyc_p1_validation.py`'s cached data). |
| `nyc/calibration_ridge_check.py` | §5.5 V2 — extends the supply grid on calibration days only. |
| `decision_currency_run.py` | §5.6 — C3 end-to-end at B=200/R=5. |
| `compute_parity_sweep.py` | §5.7 — the latency table. |
| `mde_scaling_component_ablation.py` | §7's MDE component ablation. |
| `ward_crosswalk.py` | §5.8 — population/SC/ST crosswalk. |
| `ward_amenities_crosswalk.py` | §5.8 — household-amenity-rate crosswalk. |

### Running the confirmatory study

```bash
uv sync              # installs polars, numpy, scipy, pytest, ruff, networkx
uv run pytest -q     # 158 tests
uv run ruff check .  # lint

uv run python run_study.py --n-bootstrap 40 --n-replications 4   # ~4.5 min; results/study_results.parquet
uv run python run_study.py --n-bootstrap 200 --n-replications 5  # ~26 min, the pre-registered target scale
```

To generate a small synthetic dataset and poke at it interactively:

```python
from datetime import datetime
from dispatch_eval.sources.synthetic import generate_synthetic_trips

df = generate_synthetic_trips(
    n_days=5, zones=["A", "B", "C"], start_date=datetime(2026, 1, 5), fleet_size=40
)
```

### Fetching real data

Nothing large is checked into this repository — `analysis/cache/` and
`results/` are gitignored, and every script that needs real data fetches
and caches it itself on first run:

- **NYC TLC data** (`nyc_reference_comparison.py`, `nyc_p1_validation.py`):
  public, unauthenticated parquet files at
  `https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_YYYY-MM.parquet`
  and the zone lookup at
  `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`. No
  account needed; `nyc_p1_validation.py` pulls ~14 days of Manhattan
  HV0003 trips (~34MB cached).
- **Census/BBMP data** (`ward_crosswalk.py`, `ward_amenities_crosswalk.py`):
  fetched from data.gov.in and BBMP's own published ward boundary files;
  see the script docstrings for exact URLs.
- **Namma Yatri (Bengaluru)**: not fetchable from this environment — see
  Limitation 2 above for exactly what was found and what's still needed.

## 9. Silent-exclusion audit

Prompted by a real discretization bug found in B5 (§5.6: 67.5% of
requests silently dropped as "unservable" from bin-width rounding, not
real infeasibility): a check for siblings of that failure mode across
four other candidate spots — each got either an explicit exclusion
counter in its output or a test proving it can't happen.

1. **Fleet-size calibration — real bug, fixed.** `calibrate_fleet_size`
   is a plain grid search; if the true optimum is outside the candidate
   range, it silently returns the largest (or smallest) candidate with no
   indication it never converged — indistinguishable from a genuine
   interior optimum. **This already happened in this project's own
   `run_study.py` runs** (`fleet_size=60`, the literal max of the
   candidates tried, never checked at the time) and again in the first
   NYC P1 attempt (`fleet_size=2000`, again the literal max — §5.5).
   Fixed: `FleetSizeCalibrationResult.hit_boundary: bool`, true whenever
   the returned value is the min or max of the candidates tried.
2. **NHPP arrival fit — not a bug, but a real, measured asymmetry.** A
   `(zone, day_type, bin_index)` cell with zero observed requests falls
   back to rate 0.0 with no smoothing — correct MLE for a genuinely
   covered cell, but unlike `fit_od_model` (Dirichlet-smoothed
   specifically so sparse zero-counts aren't mistaken for structural
   impossibility), there's no protection against a short fitting window
   producing a false zero. `arrival_sparsity_report` makes the fraction
   of the fitted grid resting on this fallback visible. Investigated
   further in §7 as a candidate (and ultimately ruled out) explanation
   for the MDE curve's scaling deviation.
3. **OD smoothing — checked, confirmed correct, proven not just
   claimed.** `fit_od_model`'s smoothing covers every zone, including
   ones never observed as a destination for a specific cell — proven
   directly on a real fit (`test_fit_od_model_never_assigns_exactly_zero_to_any_zone`),
   not resting on the docstring's own claim.
4. **Scenario generation / horizon truncation — one real gap, closed.**
   `generate_arrival_minutes` never produces an out-of-window arrival —
   checked directly, already correct by construction. A real gap: a
   request arriving near the end of the simulation horizon, with a
   patience window extending past it, never gets its own abandonment
   event processed — left permanently unresolved, counted in
   `total_requests` but in neither `completed_requests` nor
   `abandoned_requests`, with `fraction_served`/`mean_wait_seconds`
   silently computed as if this never happened. Not a bug (an inherent
   finite-horizon boundary effect of any discrete-event simulation with a
   cutoff), but was invisible. Added
   `SimulationResult.unresolved_at_horizon`; measured at 2.2% in a
   representative ample-supply run.

Every metric this project reports off `SimulationResult` or a fitted
input model now has a way to check whether it silently rests on excluded
or fallback data: `ClairvoyantResult.fraction_excluded_by_discretization`,
`FleetSizeCalibrationResult.hit_boundary`, `arrival_sparsity_report`,
`SimulationResult.unresolved_at_horizon`. None are checked automatically
as a hard gate — they're diagnostics to report alongside any headline
number, the same discipline applied to every other documented
approximation in this project.
