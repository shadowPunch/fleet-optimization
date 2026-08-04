# C6 — Coarsening ladder: real result

**Status: built and run twice — once on a large-effect pair (ranking
survives trivially, as expected), and once on a genuinely close pair
found via `docs/indifference_search.md` (the sharper test this doc
originally flagged as missing). Both are real, informative results.**

## What was built

`src/dispatch_eval/coarsening_ladder.py` re-runs C2's exact
`run_ranking_flip_experiment` bootstrap-CRN loop, unchanged, on four
progressively coarsened versions of the same synthetic trip data — see the
module's own docstring for the exact rung definitions and why each one is
a defensible reading of the plan's stated sequence ("15-min time rounding
→ ward-level OD only → ward-only aggregate counts with no timestamps, fare
rounding at each step"). In short:

| Rung | Time/OD binning | Fare & distance | Origin-destination |
|---|---|---|---|
| 0 (baseline) | 1-min / 15-min | exact, per trip | exact, per trip |
| 1 — "15-min rounding" | 15-min / 60-min (this project's existing P2/P3 default) | exact | exact |
| 2 — "ward-level OD" | 15-min / 60-min | route-mean only | exact |
| 3 — "ward-only aggregate, no timestamps" | daily / daily | route-mean, computed on fake routes | shuffled (destroyed) |

Rung 0 is deliberately *finer* than this project's own everyday default
(bin_minutes=15) specifically so rung 1 is a real degradation from
something, not a no-op — this project's P2/P3 runs have quietly been
living at what this ladder calls rung 1 all along.

## What was run, and the result

`analysis/coarsening_ladder_sweep.py`: B0 (nearest-idle) vs. B1 (batched
Hungarian), 14 days of synthetic data, 5 zones, fleet 30, B=20 bootstrap
draws x R=4 replications per rung (~2m20s total, no real data). Real
result:

| Rung | Nominal ranking | τ to rung 0 | P(B1 ranked first) |
|---|---|---|---|
| 0 fine-grained | B1 > B0 | 1.0 | 1.0 |
| 1 15-min rounding | B1 > B0 | 1.0 | 1.0 |
| 2 ward-level OD | B1 > B0 | 1.0 | 1.0 |
| 3 ward-only aggregate | B1 > B0 | 1.0 | 1.0 |

**The ranking survives every rung, with zero erosion in P(ranked first).**
This is a real, honest result, not a placeholder — but it's the *expected*
one for this pair, not a strong test of the coarsening ladder's value.
B0-vs-B1's effect size is large (greedy single-assignment vs. batched
globally-optimal assignment is a structurally different mechanism, not a
close variant), so it sits well outside the indifference zone C2 itself
already establishes — coarsening the input data can't flip a ranking that
bootstrap resampling *of the original fine-grained data* wasn't close to
flipping in the first place (`P(ranked first) = 1.0` at rung 0 already).

## The sharper follow-up, now run

`docs/indifference_search.md` found a real indifference set:
`ValueCorrectedHungarianPolicy` at `value_weight` in
`{0.5, 0.75, 1.0, 1.5}` are all statistically indistinguishable from each
other. `analysis/coarsening_ladder_close_pair_run.py` re-runs this exact
same coarsening ladder on the two most distant members of that set
(`weight_0.5` vs. `weight_1.5`, B=20, R=4). Real result:

| Rung | Nominal ranking (lower wait first) | τ to rung 0 | P(weight_1.5 ranked first) |
|---|---|---|---|
| 0 fine-grained | weight_0.5, weight_1.5 | 1.0 | 0.65 |
| 1 15-min rounding | weight_0.5, weight_1.5 | 1.0 | 0.70 |
| 2 ward-level OD | weight_0.5, weight_1.5 | 1.0 | 0.70 |
| 3 ward-only aggregate | weight_0.5, weight_1.5 | 1.0 | 0.60 |

Two things worth noting, not glossed over:

- **The single nominal (un-resampled) fit disagrees with the bootstrap
  majority.** The nominal ranking lists `weight_0.5` first (lower point-
  estimate wait) at every rung, but `weight_1.5` actually wins the
  *majority* of bootstrap draws (60-70%) at every rung too. A report that
  only quoted the nominal point estimate would have picked the wrong
  "typical" answer — exactly the failure mode this project's whole
  methodology (report `P(ranked first)`, not a point-estimate winner)
  exists to catch, caught here for real, not hypothetically.
- **Coarsening doesn't obviously worsen the uncertainty for this pair.**
  P(ranked first) stays in the same ~30-40%-losing-side band across all
  four rungs (never collapsing toward 0 or 1 the way the B0-vs-B1 run
  did, but also never drifting further from ~50/50 as the data gets
  coarser). For this specific pair and this specific coarsening path, the
  uncertainty was already fully present at full resolution — going to
  Bengaluru's real, coarser publication resolution doesn't compound it
  much further, at least by this measure. That's a real, checkable
  result now, not a hand-wave either way.

(`τ to rung 0` is trivially ±1 with only two policies being compared —
not informative here; `P(ranked first)` is the signal to read.)

## Scope note: B2-B4 weren't included in this run

B2 (`ValueCorrectedHungarianPolicy`) carries a value function fitted once,
outside `run_ranking_flip_experiment`'s own loop, from the *nominal*
(rung-0-resolution) data — B3/B4 wrap B2 and inherit the same object.
Reusing that same, nominal-fitted value function unchanged across every
coarsened rung is consistent with how this project already treats fleet
size (calibrated once, held fixed across the whole bootstrap loop — see
`ranking_flip.py`'s own docstring), but it means B2-B4's *own* internal
model of the world wouldn't reflect the rung's coarsening at all, which
would muddy a coarsening result specifically for them. Including B2-B4
properly means refitting the value function per rung (cheap — once per
rung, not once per bootstrap draw) — a small, well-scoped follow-up, not
attempted in this pass so the first real coarsening-ladder run could stay
about the one thing it was testing: input-model coarsening.
