# The forecast-degradation axis: real result

**Status: built and run across 9 levels, including the maximally severe
single-axis test. Result: the ranking does not collapse anywhere — B2
wins cleanly even when its arrival forecast is flattened across both
time *and* space. This is the opposite of the working hypothesis,
reported honestly rather than reframed after the fact.**

## The hypothesis this was built to check

Every rigorous comparison in `docs/indifference_search.md` found B2
(value-corrected) cleanly beating the rest of the ladder, with no
indifference between *mechanisms* — only within B2's own `value_weight`.
A real, sharp critique of that finding: B2's value function, and B3/B4's
reposition logic, were built from models fit the *same way*, on the *same
kind of data*, as the model that generates the true simulated world — an
oracle forecast no real deployment has. The working hypothesis: this
inflates B2-B4's advantage, and the "no indifference between mechanisms"
finding is (at least partly) that artifact rather than genuine evidence
these mechanisms differ this much.

`src/dispatch_eval/forecast_degradation.py` makes this checkable: the
true world stays fixed at a correctly-specified, bootstrap-varying fit
(exactly as every other P3 run in this project already does), while every
policy's own *belief* — value function, reposition reference, per-tick
travel-time estimate via the new `policy_travel_time_model` split — is
built from a deliberately degraded model instead.

## What was run

`analysis/forecast_degradation_sweep.py`, 9 levels, B=25/R=4 each, same
tuned hyperparameters throughout (only forecast quality varies, not
tuning — see the module's own docstring for why that matters):

| Level | What's degraded |
|---|---|
| `true_forecast` | baseline — no degradation |
| `n_days_1` / `n_days_3` / `n_days_5` | policy's forecast fit on only 1 / 3 / 5 of the dataset's 10 days |
| `homogeneous_arrival_only` | time-of-day shape flattened; each zone keeps its own average rate |
| `spatially_uniform_arrival_only` | **strictly more severe** — a single rate shared by every zone *and* every time bin, added after the first pass (below) found zero erosion from anything that left spatial structure intact |
| `marginal_od_only` | destination choice collapsed to one citywide marginal, ignoring origin and time |
| `flat_travel_time_only` | travel time treated as a deterministic point estimate, no variance |
| `fully_degraded` | 1 day of data + spatially-uniform arrival + marginal OD + flat travel time, simultaneously — the closest analog to a real deployment's crude, hand-built forecast |

## Result

**Identical at every single level, including the most severe one.** B2
wins 100% of bootstrap draws (`P(ranked first) = 1.0`) and is alone in
its own indifference set at α=0.05, from `true_forecast` all the way to
`fully_degraded` — `spatially_uniform_arrival_only` included, the one
built specifically to remove "which zone is busier" from the policy's
belief entirely, not just "when." The only variation anywhere in the full
JSON (`analysis/cache/forecast_degradation_sweep_results.json`) is a
same-P(ranked-first)=0 swap between B3 and B4's *nominal* ranking
position in `fully_degraded` — neither ever wins a single bootstrap draw
at any level, so this is noise between two already-losing policies, not
a finding.

**This does not confirm the working hypothesis.** The prediction was that
separation would collapse somewhere along this axis, and specifically
that removing spatial knowledge would be the sharpest test of it; it
didn't collapse even there. Reported as found, not reframed as a
near-miss.

## What this means

The straightforward spatial-signal explanation from the first pass at
this experiment (below) is now ruled out — `spatially_uniform_arrival_only`
was built to test exactly that and didn't move the result. What's left is
a more structural possibility, genuinely open rather than confirmed:
**B2's advantage may not come mainly from forecast content at all.** Two
candidate mechanisms this experiment can't distinguish between, both
consistent with the data:

1. **A residual temporal signal survives even spatial flattening.**
   `compute_value_function`'s backward induction has a finite-horizon
   boundary condition — cost-to-go is pinned to zero at the terminal bin
   regardless of demand — so `C(zone, time_bin)` still varies *by time
   bin* (decaying toward the horizon's end) even when every zone's rate
   is identical. B2's cost correction could still be doing real work
   through *when* a trip ends, not *where* it ends — a real forecast
   signal, just not a spatial one.
2. **The advantage isn't really about the value function's content at
   all.** B2 was tuned with its own `matching_radius_seconds` (independent
   of `value_weight`), and B1 was tuned separately with its own. If B2's
   tuned radius happens to suit this scenario better than B1's, some or
   all of the measured gap could be a tuning-search artifact unrelated to
   value-function forecasting, degraded or not.

Distinguishing these needs a further, more surgical experiment — e.g.
B1 with B2's exact tuned radius but `value_weight=0`, against B2 under
`spatially_uniform_arrival_only` — not run here. Flagged as the genuine
open question this result raises, not resolved by it.

**What was ruled out along the way (the first pass, before
`spatially_uniform_arrival_only` existed)**: `homogenize_arrival_model`,
`marginalize_od_model`, and `flatten_travel_time_model` in isolation, and
`n_days` limited to 1/3/5 — none of these moved the result either, which
is *why* the maximally severe spatial test was added rather than stopping
at "no collapse found yet."

## What this means for C1

The conceptual point from the critique stands regardless of this result:
`variance_decomposition`'s `input_uncertainty_ratio` still only ever
captures parameter-estimation error *within a correctly-specified
family* — structural misspecification is still exactly zero by
construction in the main bootstrap loop, because that loop was never
touched here (only a separate, held-out forecast-degradation sweep was
run alongside it; see the caveat now in `variance_decomposition`'s own
docstring). This module is what makes structural ambiguity checkable at
all, which C1 didn't have before; it isn't yet folded into
`variance_decomposition`'s own arithmetic as a third term, and doing that
— rather than running it as a separate sweep — is a real follow-up, not
attempted here.

What *is* now empirically grounded, for this specific ladder: giving
every policy a substantially worse forecast — down to one day of data,
zero spatial knowledge, zero temporal shape, marginal-only routes, and
deterministic travel time, all at once — did not change which policy
wins. That's a real, checked finding, not an assumption, and the
follow-up question it raises (is B2's edge really about forecasting, or
about something else the tuning search found) is now a sharper, more
specific one than "is this an oracle-forecast artifact."
