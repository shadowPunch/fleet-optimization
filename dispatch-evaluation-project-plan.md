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
| C5 | **Distributional accounting.** Zone-level decomposition of the mean gain; Simpson checks; join to ACS demographics. Who pays for the average improvement. | C2 | Low |
| C6 | **Coarsening experiment.** Artificially degrade NYC data to Chicago's resolution (15-min rounding, tract aggregation) and measure how much measurement coarsening alone moves the ranking. | C2 | Low, cheap, high value |

C2 is the paper. C1 is the setup. C3–C6 are what turn it from a statistics note into something an operator or a reviewer finds decision-relevant.

---

## 2. Data

### Primary: NYC TLC High Volume FHV trip records
`fhvhv_tripdata_YYYY-MM.parquet`, published monthly with roughly a two-month lag.

Why this and not Chicago as primary: it contains `request_datetime` and `pickup_datetime`, which means **rider wait time is directly observable at the trip level**. That is the single most important validation target and it is what makes calibration falsifiable rather than decorative.

Fields that matter:

- `request_datetime`, `pickup_datetime`, `dropoff_datetime` → wait time, trip duration
- `PULocationID`, `DOLocationID` → 263 taxi zones, OD structure
- `trip_miles`, `trip_time` → travel-time matrix estimation
- `base_passenger_fare`, `driver_pay`, `tolls`, `congestion_surcharge`, `cbd_congestion_fee` → revenue and cost sides
- `shared_request_flag`, `shared_match_flag` → pooling, if you extend
- `hvfhs_license_num` → operator; **filter to a single operator**, otherwise you are simulating a market, not a fleet

Caveats to state explicitly in the report, not bury:

- `on_scene_datetime` is only reliably populated for accessible vehicles per the data dictionary. Use `request → pickup`, not `request → on_scene`.
- **Only matched trips appear.** Abandoned and unserved requests are absent. Every wait distribution you fit is conditional on eventual service. This is a censoring problem, not a nuisance — it biases calibration toward optimistic supply and it belongs in the structural ambiguity set (C1's third component).
- Supply is unobserved. Fleet size is a latent parameter, not a known input.

### Secondary: Chicago TNP Trips
Census-tract OD, times rounded to 15 minutes, fares rounded. Two uses:

1. External validity — does the C2 result replicate in a second city?
2. C6 — Chicago's documented coarsening is a natural experiment. Coarsen NYC to match it and measure the ranking shift attributable to measurement resolution alone.

### Network
Do **not** start with an OSMnx road graph. Start with a zone-to-zone, hour-of-day travel-time matrix estimated directly from `trip_time` in the trip records. It is calibratable against the same data that produced it, it is fast, and it avoids the free-flow-speed fiction. Road network is a Phase 5 extension only if C4 needs it.

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

**P0.2 Observability table.** One table, three columns: quantity / observable in public data / how it enters the simulator. Rows: request arrivals, OD distribution, trip duration, trip distance, fare, driver pay, rider wait, pickup distance, fleet size, driver online/offline, acceptance rate, cancellation, abandonment, repositioning. Most of the bottom half will be *latent*. That table is the honest foundation of the whole argument, and it is also Figure 1 of the paper.

### P1 — Minimal digital twin + calibration (week 3–5)

Deliberately minimal. Event-driven, `heapq`-based, zone-level, single operator, no road graph.

**State:** vehicles (idle / en-route-to-pickup / occupied / repositioning, each with a zone and an available-at timestamp), open requests (origin zone, destination zone, request time, patience clock).

**Events:** request arrival, dispatch decision tick, pickup, dropoff, reposition arrival, abandonment.

**Input models to fit:**

| Input | Model | Notes |
|---|---|---|
| Request arrivals | Non-homogeneous Poisson per zone, piecewise-constant rate by (zone, day-type, 15-min bin) | Test the Poisson assumption — index of dispersion by zone. Overdispersion is common and belongs in the ambiguity set |
| Destination | Row-normalised OD matrix per (origin zone, time bin) | Smoothing/shrinkage needed for sparse zone pairs |
| Travel time | Lognormal per (origin, destination, hour) fitted to `trip_time` | Keep the variance, not just the mean |
| Fare / driver pay | Regression on distance, duration, time bin | Needed for C3 |
| Fleet size | **Latent — calibrated** | See below |
| Abandonment hazard | **Latent — structural ambiguity** | See below |

**Calibration of latent quantities.** Fleet size and the abandonment hazard are not identified separately by matched-trip data alone. Handle this honestly:

- Calibrate fleet size to match the *observed wait-time distribution* (not just its mean — match the median and the 90th percentile) under the nearest-idle baseline, since first-dispatch is closest to what generated the data.
- Treat the abandonment hazard as a structural axis: run the whole study under 3–5 abandonment specifications spanning plausible behaviour. Any conclusion that holds across all of them is robust; any that does not is a finding.

**Validation, held-out days:** wait-time distribution (KS distance, not mean-only), trips served per vehicle-hour, empty-mile fraction, revenue per vehicle-hour, hour-of-day shape. Pre-register the acceptance thresholds *before* running validation. If validation fails badly, that is a publishable result about the identifiability of these simulators, not a project failure — write it up.

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

Computational note: B×R×|policies| simulations is the budget driver. Use a metamodel-assisted variant if it blows up — fit a cheap surrogate over θ-space and reserve full simulation for a designed subset. Start with a short horizon (a 6-hour Manhattan weekday peak) to make the full factorial feasible, then extend.

### P4 — Decision translation (week 10–12)

**C3 — Exchange rate.** Sweep fleet size in the simulator to get the wait-time-vs-fleet curve. Invert it: any policy's wait-time improvement becomes "worth X vehicles." Propagate the C2 uncertainty through, so the answer is an interval: *"the lookahead policy is worth 34 vehicles, 95% CI [−12, 81]."* An interval that straddles zero communicates more than any percentage improvement in the literature. Also report Δ$/day and Δdriver-hours.

**C4 — Compute parity.** Fix a real-time dispatch cycle (production systems run on cycles of a couple of seconds). For each policy, measure wall-clock decision time as a function of open requests and idle vehicles. Then: at each fleet scale, which policies can actually complete a decision inside the budget? Degrade the ones that cannot (truncate the candidate graph, coarsen the radius) and re-measure quality. Plot quality vs. decision-latency budget. Expect some optimal-in-principle methods to lose once charged for their latency — and expect the crossover point to move with fleet size. This is a clean, underreported result.

**C5 — Distributional accounting.** Decompose the mean wait-time gain by zone. Check for Simpson reversals (mean improves, majority of zones worsen). Join taxi zones to ACS tract demographics via a crosswalk and report the gain distribution across income and race quantiles. State clearly that this is an association in a simulation, not a causal claim about a real platform.

**C6 — Coarsening.** Re-run P3 with input models fitted to NYC data artificially coarsened to Chicago's published resolution (15-min time rounding, tract-level aggregation, fare rounding). Report the ranking shift attributable to measurement resolution alone. Cheap, and it directly answers "can you even do this study on Chicago data?"

### P5 — RL entry (optional, week 13+)

Only after P3 is done. Gymnasium environment wrapping the same simulator, central dispatcher, PPO.

The honest framing: RL is **one entry in the comparison**, not the point. Two things to report that the literature usually does not:

- **Seed variance.** Train 5+ seeds. If the spread across training seeds exceeds the gap to the best tuned heuristic, say so plainly. That is a result.
- **Budget parity.** Report the total simulator-evaluation count for RL training against the BO budget given to the baselines. If RL used 100× the compute, that belongs in the abstract.

Expect the outcome that RL sits inside the indifference set. That is a fine outcome and it is the one that makes the paper interesting.

---

## 4. Engineering notes

- **Language:** Python. `polars` for the trip data (20M+ rows/month), `heapq` for the event queue, `scipy.optimize.linear_sum_assignment` for Hungarian, `networkx` or `ortools` min-cost flow, `scikit-optimize`/`optuna` for BO, `gymnasium` + `stable-baselines3` for P5.
- **Scale control:** Manhattan-only, single operator, weekday, 6-hour peak window for the main study. Extend to full-borough and full-day for the final robustness section only.
- **Reproducibility discipline:** seed everything; store `(policy, θ_bootstrap_index, replication_seed) → metrics` in a single parquet results table; never recompute a headline number from a notebook cell. One `run_study.py` that regenerates every figure from raw data.
- **Pre-registration:** write down validation acceptance thresholds, the α for the indifference set, and the primary metric *before* running P1 validation and P3. Commit the file with a timestamp. This is the process correction that mattered most in the superconductivity work and it matters more here, because the whole paper is a claim about statistical discipline.

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
