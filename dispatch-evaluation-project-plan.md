# Project Plan: What Do We Actually Know About Dispatch Policy Rankings?

**Working title:** Input-Uncertainty-Aware Evaluation of Ride-Hailing Dispatch Policies
**Deliverable:** open-source evaluation harness + technical report + executive brief
**Target venue:** Winter Simulation Conference (methodology track) — check current deadline; typical cycle is spring submission for a December conference. Fallback: arXiv + strong repo.

---

## 0. The thesis

The literature reports dispatch policy improvements of 1–10% on simulators calibrated from partial data. Nobody propagates the uncertainty in that calibration into the comparison. The claim to test:

> At the effect sizes reported in the ride-hailing dispatch literature, policy rankings are not statistically identified once input-model estimation error is propagated. This project quantifies the identification threshold, reports the indifference set rather than a winner, and expresses every result in a decision currency (vehicle-equivalents) so that algorithmic gains and capacity gains are directly comparable.

Two properties make this a good thesis:

1. **It is falsifiable and informative either way.** If rankings are stable, the deliverable is the minimum detectable effect curve — how much data you need to resolve a 2% claim. If rankings are unstable, the deliverable is a correction to how a whole subfield reports results.
2. **It licenses its own simplicity.** If the argument is that policy differences are swamped by input uncertainty, you do not need a microscopic road-network simulator. You need an honest one with quantified uncertainty. This kills the main failure mode of the original project shape (spending four months building simulator infrastructure that contributes nothing).

---

## 1. Contributions, in dependency order

| ID | Contribution | Depends on | Kill risk |
|----|---|---|---|
| C1 | **Variance decomposition.** Split the variance of a policy-comparison estimate into intrinsic simulation noise, input-model estimation error, and structural ambiguity. Report the ratio. | P1, P2 | Low |
| C2 | **Ranking-flip experiment.** Bootstrap input models × common random numbers × k policies. Report the distribution over rankings, not a winner. Report the indifference set at level α. | C1 | Low — this is the centrepiece |
| C3 | **Decision currency.** Express every policy delta as Δfleet-equivalent, Δ$/day, Δdriver-hours, with intervals. | C2 | Low |
| C4 | **Compute-parity frontier.** Quality vs. decision-latency budget, at a fixed real-time dispatch cycle. Charge every policy for its own latency. | P2 | Medium |
| C5 | **Distributional accounting.** Ward-level decomposition of the mean gain; Simpson checks; join to Census of India / BBMP ward demographics. Who pays for the average improvement. | C2 | Low |
| C6 | **Coarsening ladder.** Start from Delhi NCR's synthetic trip-level resolution and coarsen it step-by-step toward what Bengaluru's real open data actually offers (15-min rounding → ward-level OD → ward-only aggregate counts, no timestamps at all). Measure how much of the ranking survives at each rung. | C2 | Low, cheap, high value |

C2 is the paper. C1 is the setup. C3–C6 are what turn it from a statistics note into something an operator or a reviewer finds decision-relevant.

---

## 2. Data

There is no Indian-city equivalent of NYC TLC's High Volume FHV records — no regulator publishes trip-level ride-hailing data for any Indian city with a `request_datetime`/`pickup_datetime` split. That fact is itself load-bearing for this project (see the note at the end of this section), not a gap to paper over. Two sources fill the primary/secondary roles instead, and — critically — they are **inverted in resolution** relative to the original NYC(fine)/Chicago(coarse) plan: the only *real* source is coarse, and the only *fine-grained* source is synthetic.

> **AMENDMENT, 2026-08-04 — data-source roles restructured, made before any real-data validation attempt.** The assignment below (written before any of these sources was actually inspected) made Bengaluru's Namma Yatri aggregates the *primary* validation source and Delhi NCR's Kaggle set a *secondary, synthetic* one. Two things learned since then make that untenable, not just imperfect:
>
> 1. Namma Yatri's aggregates have **no `request_datetime`/`pickup_datetime` split at all** (confirmed against the actual export shape, not assumed) — wait time, this project's primary metric and the quantity P1's own pre-registered KS-distance threshold is defined against, is not observable from this source under any circumstance. A validation criterion that can never be evaluated against its intended source isn't a validation criterion.
> 2. Delhi NCR's Kaggle set remains of **unverified, plausibly-synthetic provenance** (see below) — presenting it as real operational data, in a paper whose central claim is about statistical honesty, is exactly the kind of thing a reviewer would (rightly) flag.
>
> Meanwhile `sources/nyc_tlc.py` is a **verified** adapter (checked against the actual live parquet schema, not guessed) with real trip-level `request_datetime`, already fully wired into this codebase (`analysis/nyc_reference_comparison.py`) and giving an actually-falsifiable P1 validation.
>
> **Restructured roles:**
> - **NYC TLC data becomes the primary methodological study.** P1 validation, C1 (variance decomposition), C2 (the ranking-flip experiment), C3 (decision currency), and C6's "start from real trip-level data" direction all run against real NYC data — this is what makes the project's central statistical claim checkable against something real, not merely plausible-sounding on synthetic data.
> - **Bengaluru (Namma Yatri) becomes a dedicated applicability study, not a calibration source.** Its ward-aggregate resolution can't support P1/C2, but it is exactly the right target for C6's actual question: given real, currently-available Indian open mobility data at its *actual* resolution, what can honestly be concluded? That is a genuine, useful finding about Indian open mobility data on its own — no longer downstream of the main NYC-based study, a separate contribution in its own right.
> - **Delhi NCR is dropped from every result.** The schema adapter (`sources/delhi_ncr.py`) and its tests stay in the codebase as a working capability — useful if a verified-provenance Indian trip-level source ever surfaces — but no claimed result in this project rests on it, and it is not run as part of the confirmatory study.
>
> This amendment is made **before** any real-data validation has been attempted against any source — the correct time to revise a data plan in light of what the sources actually are, not after seeing whether a validation attempt succeeds or fails. `docs/pre_registration.md` carries the corresponding dated update to the validation thresholds themselves.

### Primary (real, coarse): Bengaluru — Namma Yatri Open Data
[nammayatri.in/open](https://nammayatri.in/open/) publishes operational statistics from Namma Yatri, a live, open-source, commission-free auto-rickshaw dispatch platform (Direct-to-Driver, built on the ONDC/Beckn protocol). This is real operational data from an actual running dispatch system, which is why it anchors the project — but it is **aggregated at the ward level**, not trip-level:

- Ride requests, bookings, completed trips, cancellations, fare estimates, and driver earnings, aggregated per ward and time window.
- No `request_datetime`/`pickup_datetime` split at all — wait time is **not directly observable**, not even conditionally on matched trips. This is a step worse than NYC's censoring problem: NYC censors unmatched requests but still gives trip-level timestamps for matched ones; Bengaluru's open data gives neither.
- No confirmed OD matrix — published figures appear to be per-ward marginals (requests, completions, cancellations by ward), not ward-to-ward flows. Treat OD structure as unavailable from this source until verified against the actual export.
- A scraped snapshot exists as a Kaggle mirror (`arshdkhan/namma-yatri-bengaluru-ward-wise-ride-open-data`); treat it as a convenience copy of the same aggregates, not an independent source.

Use this source for: validating the simulator's *aggregate* outputs (ward-level trip volume, cancellation rate, fare range) against something real, and for C5's demographic join (Census of India / BBMP ward boundaries and demographics, in place of ACS tracts).

### Secondary (synthetic, fine): Delhi NCR — Kaggle ride-booking dataset
`shayanzk/ola-ride-bookings-dataset` (Kaggle) contains trip-level rows with Delhi NCR locations (Okhla, Barakhamba Road, Cyber Hub, Saket, etc.), booking timestamps, pickup/drop locations, distances, fares, and trip status. **Treat this as synthetic, not real operational data**, until proven otherwise — Ola/Uber do not publish real trip microdata for India, the dataset carries no data-provenance disclosure, and its shape matches the generic templated datasets Kaggle users generate for BI/SQL portfolio practice, not a real platform export.

Role: it has the right *shape* (trip-level rows, timestamps, OD, fare) to exercise the full simulator + calibration + bootstrap harness end-to-end. Use it as a structural test-bed and as the top rung of the C6 coarsening ladder — **never as evidence about real Delhi ride-hailing**, and never let a headline result rest on it alone. Any claim calibrated against it must be labeled synthetic-data in every figure and table it touches.

### The inversion, and why it's a finding, not just a workaround
The original plan's C6 ("coarsen NYC to Chicago's resolution and measure the shift") assumed you start from a real fine-grained baseline and degrade it. Here, the real data (Bengaluru) *is already at the degraded end* — coarser than even Chicago's 15-minute rounding, since it has no trip-level timestamps or OD at all. So C6 becomes a ladder rather than a single before/after comparison: start from Delhi NCR's (synthetic) trip-level resolution, coarsen in steps — 15-minute rounding, ward-level OD, ward-only aggregate counts with no timestamps — and find where in that ladder the ranking stops being identifiable. The bottom rung of the ladder is exactly what Bengaluru's real open data provides. If identifiability collapses before you reach that rung, it says something concrete and citable about whether this class of study is even possible on the public data that actually exists for Indian cities — which is a stronger and more honest version of the original point than "does it replicate in a second city."

### Network
Same principle as the original plan: do **not** start with an OSMnx road graph. Build a zone-to-zone (ward-to-ward), hour-of-day travel-time matrix from whichever source has trip-level `trip_time`/timestamps — currently only the Delhi NCR synthetic set qualifies, so the network model is provisional on synthetic data until a real trip-level source is found or purchased. Flag every figure derived from it accordingly. Road network is a Phase 5 extension only if C4 needs it.

### Engineering consequence
Because neither source can be authenticated or downloaded automatically from this environment (Kaggle requires a personal API token; Namma Yatri's aggregates require a scrape), the codebase is built against one **canonical internal trip-record schema**, with a source-specific adapter per dataset and a synthetic generator that matches the same schema for development and tests. This means P1 can be built, tested, and validated end-to-end now; swapping in the real Delhi NCR CSV or a fresh Bengaluru scrape later is a matter of writing one adapter each, not restructuring the simulator. See `docs/observability_table.md` for the full quantity-by-quantity observability audit against this data landscape.

---

## 3. Phases

### P0 — Novelty gate and observability audit (week 1–2)

Two deliverables, both short, both go/no-go.

**P0.1 Novelty check.** Before building anything, confirm the gap. Search strings to run against Google Scholar, arXiv, INFORMS journals, and Winter Simulation Conference proceedings:

- `"input uncertainty" ride-hailing simulation`
- `"ranking and selection" ride-hailing OR mobility-on-demand OR dispatch`
- `bootstrap "input model" dispatch policy comparison`
- `"probability of correct selection" fleet OR dispatch OR mobility`
- `"common random numbers" ride-hailing simulation policy`
- forward-citation sweep on Barton/Nelson/Xie 2014, Song & Nelson, Fan/Hong/Zhang robust selection, Lam's cheap bootstrap — looking for any transportation application

Outcome: a one-page memo listing the closest 5 papers and the precise sentence describing what each does *not* do. If someone has already done C2 for ride-hailing, pivot to C4+C3 (compute-parity + decision currency), which is a separate and still-open contribution.

*Done, provisional GO — see `docs/p0_novelty_check.md`: a web-search pass (not a full Scholar/WSC-archive citation traversal) across all five strings plus the named forward-citation sweep found no paper combining bootstrap-over-input-models, CRN, and a ranking-flip/indifference-set report across dispatch policies for ride-hailing. Closest five papers and what each doesn't do are listed there. No pivot triggered — but redo this properly (real Scholar citation chains, direct WSC/INFORMS search) before submitting anywhere; this pass is good enough to keep building on, not good enough to cite as confirmed.*

**P0.2 Observability table.** One table, three columns: quantity / observable in public data / how it enters the simulator. Rows: request arrivals, OD distribution, trip duration, trip distance, fare, driver pay, rider wait, pickup distance, fleet size, driver online/offline, acceptance rate, cancellation, abandonment, repositioning. Most of the bottom half will be *latent*. That table is the honest foundation of the whole argument, and it is also Figure 1 of the paper.

### P1 — Minimal digital twin + calibration (week 3–5)

Deliberately minimal. Event-driven, `heapq`-based, ward-level (zone-level), single operator, no road graph.

**State:** vehicles (idle / en-route-to-pickup / occupied / repositioning, each with a zone and an available-at timestamp), open requests (origin zone, destination zone, request time, patience clock).

**Events:** request arrival, dispatch decision tick, pickup, dropoff, reposition arrival, abandonment.

**Input models to fit:**

| Input | Model | Notes |
|---|---|---|
| Request arrivals | Non-homogeneous Poisson per zone, piecewise-constant rate by (zone, day-type, 15-min bin) | Test the Poisson assumption — index of dispersion by zone. Overdispersion is common and belongs in the ambiguity set. Fittable against Bengaluru ward counts (real) or Delhi NCR synthetic rows |
| Destination | Row-normalised OD matrix per (origin zone, time bin) | Smoothing/shrinkage needed for sparse zone pairs. **Only fittable from the Delhi NCR synthetic set** — no confirmed OD structure in Bengaluru's published aggregates |
| Travel time | Lognormal per (origin, destination, hour) fitted to `trip_time` | Keep the variance, not just the mean. **Synthetic-data-only until a real trip-level source exists** — flag every downstream number derived from it |
| Fare / driver pay | Regression on distance, duration, time bin | Needed for C3. Fit against Delhi NCR synthetic rows; sanity-check the fare *range* against Bengaluru's published fare estimates |
| Fleet size | **Latent — calibrated** | See below |
| Abandonment hazard | **Latent — structural ambiguity** | See below |

**Calibration of latent quantities.** Fleet size and the abandonment hazard are not identified separately by matched-trip data alone, and — with this data landscape — the wait-time distribution itself is only observable in the synthetic Delhi NCR set (via its VTAT-style pickup-wait field, itself unverified). Handle this honestly, and treat it as one level more uncertain than the original NYC-based plan assumed:

- Calibrate fleet size to match the *observed wait-time distribution* (not just its mean — match the median and the 90th percentile) under the nearest-idle baseline, using the Delhi NCR synthetic set. Label every fleet-size figure downstream of this as synthetic-calibrated until a real trip-level source replaces it.
- Where a real anchor exists (Bengaluru ward-level trip counts and cancellation rates), use it as an independent sanity check on the simulator's aggregate output, not as a calibration target for wait time.
- Treat the abandonment hazard as a structural axis: run the whole study under 3–5 abandonment specifications spanning plausible behaviour. Any conclusion that holds across all of them is robust; any that does not is a finding.

**Validation, held-out days:** wait-time distribution (KS distance, not mean-only) against held-out Delhi NCR synthetic rows; ward-level trip counts and cancellation rate against Bengaluru's published aggregates; trips served per vehicle-hour, empty-mile fraction, revenue per vehicle-hour, hour-of-day shape. Pre-register the acceptance thresholds *before* running validation. If validation fails badly, that is a publishable result about the identifiability of these simulators from public Indian ride-hailing data, not a project failure — write it up.

### P2 — Baseline ladder with enforced parity (week 6–7)

The parity condition is the methodological point. Every policy gets the **same tuning budget** and is charged for its **own decision latency**.

Ladder:

- **B0** Nearest idle vehicle (first dispatch)
- **B1** Batched bipartite matching, Hungarian, with tuned batch window Δ and matching radius r. Note that analytic guidance exists for optimal Δ and r under given supply-demand conditions — implement that as B1', because it is a much stronger baseline than the greedy default most papers compare against
- **B2** Min-cost flow over the bipartite graph with edge costs including a value-function correction (the learning-and-planning structure)
- **B3** B2 + fluid/zone-balancing repositioning
- **B4** B3 + sampling-based lookahead repositioning (sample future requests from the fitted arrival model, solve an assignment over samples)
- **B5** *Clairvoyant* min-cost flow with full future knowledge — an offline upper bound, not a policy

**B5 is important.** It lets you report every policy as *fraction of the clairvoyant gap closed*, which normalises across simulators and cities in a way that "3.2% better than greedy" does not. That normalisation is itself a small interpretive contribution and makes your numbers comparable to other people's.

**Tuning:** each of B1–B4 has 2–6 continuous parameters (Δ, r, reposition threshold, lookahead horizon, value discount). Give each an identical Bayesian optimisation budget (e.g. 200 simulator evaluations, same acquisition, same seeds). Log the tuning curves — the gap between untuned and tuned baselines is itself a result worth reporting, because it bounds how much of the literature's claimed RL advantage could be tuning asymmetry.

### P3 — The ranking-flip experiment (week 8–9) — **the centrepiece**

Procedure:

```
for b in 1..B:                          # B ≈ 200–500
    D*_b ← bootstrap resample of the real trip data
    θ*_b ← refit all input models on D*_b
    for each policy π in ladder:
        for r in 1..R:                  # R replications, CRN across policies
            y[π, b, r] ← simulate(π, θ*_b, seed_r)
    rank_b ← ordering of policies by mean_r y[π, b, r]
```

Common random numbers across policies within (b, r) is non-negotiable — it is what makes the paired comparison sharp and it is what most of the literature omits.

**Outputs:**

- `P(π ranked first)` for each policy — a bar chart. If this is spread across three policies, the point-estimate literature is unfounded.
- Kendall's τ between each bootstrap ranking and the nominal ranking — a histogram.
- **Indifference set** at level α: the set of policies that cannot be separated. Report this instead of a winner.
- **Variance decomposition (C1):** total variance of the pairwise difference, split into within-θ (intrinsic) and across-θ (input uncertainty) components. Report the ratio. The headline number is something like "input uncertainty contributes Nx the variance of simulation noise."
- **Minimum detectable effect curve:** as a function of days of real data used for fitting, the smallest true effect whose sign is resolved at 95%. This is the practically useful artefact — it tells a future researcher how much data they need before a 2% claim means anything.

Computational note: B×R×|policies| simulations is the budget driver. Use a metamodel-assisted variant if it blows up — fit a cheap surrogate over θ-space and reserve full simulation for a designed subset. Start with a short horizon (a 6-hour weekday peak over the central-Bengaluru or Delhi NCR study area) to make the full factorial feasible, then extend.

### P4 — Decision translation (week 10–12)

**C3 — Exchange rate.** Sweep fleet size in the simulator to get the wait-time-vs-fleet curve. Invert it: any policy's wait-time improvement becomes "worth X vehicles." Propagate the C2 uncertainty through, so the answer is an interval: *"the lookahead policy is worth 34 vehicles, 95% CI [−12, 81]."* An interval that straddles zero communicates more than any percentage improvement in the literature. Also report Δ$/day and Δdriver-hours.

*Built: `src/dispatch_eval/decision_currency.py` (tested, `tests/test_decision_currency.py`). The fleet-size curve is built once on nominal data (mirrors the scope decision `ranking_flip.py` already makes for fleet size itself — recalibrating a full curve per bootstrap draw would multiply the simulation budget by the curve's own point count), then inverted per bootstrap draw using C2's already-computed `metric_by_policy`, at zero extra simulation cost beyond the curve sweep itself. Handles the case where a policy's wait time falls outside the swept fleet-size range via linear extrapolation (flagged per-draw), rather than silently clipping and understating the answer. Δdriver-hours is exact (`vehicle_hours` scales linearly with fleet size in the engine); Δ$/day is a documented approximation — a flat fare-per-trip from the input data times a simulated trips-per-vehicle-per-day rate, since the simulator's `Request` entity never carries a fare (fares are fit for calibration only, never assigned to simulated trips).*

**C4 — Compute parity.** Fix a real-time dispatch cycle (production systems run on cycles of a couple of seconds). For each policy, measure wall-clock decision time as a function of open requests and idle vehicles. Then: at each fleet scale, which policies can actually complete a decision inside the budget? Degrade the ones that cannot (truncate the candidate graph, coarsen the radius) and re-measure quality. Plot quality vs. decision-latency budget. Expect some optimal-in-principle methods to lose once charged for their latency — and expect the crossover point to move with fleet size. This is a clean, underreported result.

*Measurement half built (see `docs/compute_parity.md`): `src/dispatch_eval/compute_parity.py` times every B0-B4 policy's `dispatch()`/`reposition()` at problem sizes from 10x10 up to 600x600. Real result: every policy fits a multi-second production cycle even at 600x600, the O(n³) Hungarian-solver scaling the plan expects is visible in the trend (B1-B4 cost roughly cubically; B2-B4 run ~4x B0 at matched size) but only crosses a tight 100ms budget at the most extreme size tested. The "degrade + re-measure quality" half is **not** built: B1/B2's `matching_radius_seconds` looked like the plan's "truncate the candidate graph" lever but turns out to be cost-masking, not graph-size reduction — confirmed empirically (the solved matrix is the same shape regardless of radius, `tests/test_compute_parity.py`) — so it changes assignment quality, not latency. A real candidate-graph-truncation mechanism doesn't exist in this codebase yet; building one is the prerequisite for that half.*

**C5 — Distributional accounting.** Decompose the mean wait-time gain by zone/ward. Check for Simpson reversals (mean improves, majority of zones worsen). Join wards to Census of India / BBMP demographic data via a crosswalk and report the gain distribution across income and social-category quantiles (the ACS income/race split has no direct Indian-census analogue — use whatever socioeconomic strata the ward-level census or SECC data actually supports, and say so explicitly rather than forcing a US-shaped category system onto Indian data). State clearly that this is an association in a simulation, not a causal claim about a real platform.

*Data check done, both crosswalks built (see `docs/census_bbmp_data.md`): the demographic side is real and downloadable — Census 2011 ward-level population, SC/ST share, and household-amenities data via OpenCity — but it's tied to ~198-ward boundaries from 2011, and Bengaluru's wards have since been redrawn at least three times, most recently by the September 2025 BBMP→Greater Bengaluru Authority restructuring (now 369 wards). The needed GIS crosswalk between boundary eras is implemented as two functions (`src/dispatch_eval/geo/crosswalk.py`, tested): `areal_interpolate` for extensive/count variables (population, SC/ST — applied in `analysis/ward_crosswalk.py`) and `areal_interpolate_weighted_average` for intensive/rate variables (household amenities — applied in `analysis/ward_amenities_crosswalk.py`, since summing two wards' percentages is meaningless). Validated: citywide population/SC/ST totals match the GBA's own official ward-level reallocation to within 0.3%, but per-ward area-weighting error is substantial (~25% median); amenity citywide means hold within a point after crosswalking except treated-water access (5.6-point shift, flagged for extra caution). Literacy/worker-participation rates are still not obtained — data.gov.in's Karnataka PCA file specifically resisted scripted access (its modern portal is a client-side-rendered SPA: 403 from `WebFetch`, empty shell from `curl`), a real documented obstacle, not a cursory miss. Still unverified: whether Namma Yatri's actual ward data uses this same 369-ward scheme — check before reporting C5 results. SECC's caste data was never released (only SC/ST is available); Karnataka's own 2015 caste survey exists but isn't a stable public dataset as of this check. Census 2027 won't publish usable tables for years.*

**C6 — Coarsening ladder.** Re-run P3 with input models fitted to the Delhi NCR synthetic data artificially coarsened step by step toward Bengaluru's real published resolution (15-min time rounding → ward-level OD only → ward-only aggregate counts with no timestamps, fare rounding at each step). Report the ranking shift at each rung. Cheap, and it directly answers "can you even do this study on the public data that actually exists for an Indian city?" — with the answer graded rather than binary.

*Built and run twice (see `docs/coarsening_ladder.md`): `src/dispatch_eval/coarsening_ladder.py` re-runs C2's exact bootstrap-CRN loop unchanged on four progressively coarsened versions of the same data (only two small, well-tested transforms needed — fare/distance collapsed to route-level means, and destination shuffled to destroy true OD structure — plus reusing `fit_all_models`'s own bin-width parameters, now threaded through `run_ranking_flip_experiment`). First run, B0-vs-B1 (14 days synthetic, B=20, R=4): ranking survives all four rungs with zero erosion — expected, since that pair's effect size already sits well outside C2's own indifference zone, not a strong test. The sharper follow-up needed a genuinely close pair from an actual indifference set — found via a separate search (`docs/indifference_search.md`: `ValueCorrectedHungarianPolicy` at `value_weight` in `{0.5, 0.75, 1.0, 1.5}` are all statistically indistinguishable from each other) — and re-run on `weight_0.5` vs. `weight_1.5`. That run found something real: the single nominal (un-resampled) fit's point estimate disagrees with the bootstrap-majority winner at every rung (P(weight_1.5 ranked first) = 0.60-0.70 throughout while the nominal fit alone favors weight_0.5) — a concrete instance of exactly the failure mode "report the distribution, not a point estimate" exists to catch. Coarsening itself didn't obviously worsen the uncertainty for this pair across the four rungs tested. B2-B4 weren't included in the B0-vs-B1 run: their value function is fit once at nominal resolution and reused unchanged across rungs, which would muddy a coarsening result specifically for them until it's refit per rung too (cheap, but a separate follow-up) — moot for the close-pair run, which compares two `value_weight` settings of the same already-nominal-fitted value function directly.*

*Note on the 2026-08-04 data-role amendment above: everything just described coarsens this project's own **synthetic** data, not real Bengaluru data — a methodology check (does the coarsening machinery work, does it find real effects) rather than the actual applicability study C6 is meant to be. The real version — coarsening real NYC trip-level data down through the ladder, and separately checking what real Namma Yatri ward aggregates alone would support — is the concrete next step the amendment points to, not yet run.*

### P5 — RL entry (optional, week 13+)

Only after P3 is done. Gymnasium environment wrapping the same simulator, central dispatcher, PPO.

The honest framing: RL is **one entry in the comparison**, not the point. Two things to report that the literature usually does not:

- **Seed variance.** Train 5+ seeds. If the spread across training seeds exceeds the gap to the best tuned heuristic, say so plainly. That is a result.
- **Budget parity.** Report the total simulator-evaluation count for RL training against the BO budget given to the baselines. If RL used 100× the compute, that belongs in the abstract.

Expect the outcome that RL sits inside the indifference set. That is a fine outcome and it is the one that makes the paper interesting.

---

## 4. Engineering notes

- **Language:** Python. `polars` for the trip data (20M+ rows/month), `heapq` for the event queue, `scipy.optimize.linear_sum_assignment` for Hungarian, `networkx` or `ortools` min-cost flow, `scikit-optimize`/`optuna` for BO, `gymnasium` + `stable-baselines3` for P5.
- **Scale control:** a single central-city ward cluster (Bengaluru) or the Delhi NCR synthetic study area, single operator, weekday, 6-hour peak window for the main study. Extend to full-city and full-day for the final robustness section only.
- **Reproducibility discipline:** seed everything; store `(policy, θ_bootstrap_index, replication_seed) → metrics` in a single parquet results table; never recompute a headline number from a notebook cell. One `run_study.py` that regenerates every figure from raw data.
- **Pre-registration:** write down validation acceptance thresholds, the α for the indifference set, and the primary metric *before* running P1 validation and P3. Commit the file with a timestamp. This is the process correction that mattered most in the superconductivity work and it matters more here, because the whole paper is a claim about statistical discipline.

  *Done: `docs/pre_registration.md`, committed 2026-08-04 — KS-distance and hour-of-day-shape thresholds for P1, the abandonment-hazard sweep specifications, mean wait time as the primary P3 metric, α=0.05 for the indifference set, a minimum n_bootstrap≥40 (evidence-based, from the MDE-scaling-validation work's own finding that n_bootstrap=15 was unstable), and a B5 gap-closed reporting commitment (from the finding that mean_patience_seconds=300s makes B5 not a real upper bound). Explicitly scoped to the confirmatory study against real data, not the exploratory/methodology-development work already done — see that doc for why pre-registering after that work but before the confirmatory run is the correct order.*

---

## 5. Risks and responses

| Risk | Response |
|---|---|
| Rankings turn out stable | Publish the MDE curve and the threshold. Still a contribution: "here is when you're allowed to believe a 2% claim." Reframe title toward the identification threshold |
| Simulator cannot match the observed wait distribution | This is a *bigger* result about identifiability from public data. Write it up as the primary finding; C2 becomes secondary |
| Someone has already published C2 for ride-hailing | P0.1 catches this in week 1. Pivot to C4 + C3, which are separately open |
| Compute blows up on B×R×policies | Shorten horizon, reduce B with a metamodel-assisted bootstrap, run the full factorial only for the top-4 policies |
| Scope creep into simulator-building | Hard rule: no road network, no pooling, no pricing, no multi-operator until P3 is complete and written up |
| Time pressure from other commitments | P0–P3 is the minimum publishable unit. P4–P5 are extensions. Structure the writing so the paper is submittable at the end of P3 + C3 |

---

## 6. Sequencing against the original six stages

The original Stage 1–6 structure maps onto this, but reordered so the contribution comes early rather than last:

| Original | Here | Change |
|---|---|---|
| S1 Digital twin | P1 | Deliberately minimal; zone-level, no road graph |
| S2 Calibration | P0.2 + P1 | Elevated: the observability audit becomes a contribution, not a preliminary |
| S3 Classical baselines | P2 | Parity condition added; clairvoyant bound added |
| S4 Optimisation / RL | P2 (BO for baselines) + P5 (RL, demoted) | RL moved to optional and reframed as one entry |
| S5 Evaluation | **P3 — promoted to the centrepiece** | CRN + multiple replications retained; bootstrap over input models added |
| S6 Decision analysis | P4 | Pareto retained; vehicle-equivalent currency and compute-parity frontier added |

---

## 7. What the finished artefact looks like

1. **`dispatch-eval`** — a pip-installable package: minimal DES, the baseline ladder, the bootstrap-CRN harness, and the reporting functions (indifference set, MDE curve, vehicle-equivalent conversion). Usable by someone else on their own simulator. This is the reusable contribution.
2. **Technical report / paper** — 8–10 pages, WSC format.
3. **Executive brief** — two pages, no equations, for the decision-analysis framing: here is what the fleet operator should actually do, here is the confidence attached to it, here is what more data would buy.
4. **A single figure** that carries the whole argument: the ranking-flip distribution next to the reported effect sizes from the literature, on the same axis.
