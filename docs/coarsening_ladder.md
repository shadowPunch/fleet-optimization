# C6 — Coarsening ladder: real result

**Status: built and run. Result for the pair tested: the ranking survives
every rung — but that's the expected outcome for this specific pair, and
the natural sharper follow-up (below) hasn't been run yet.**

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

## The sharper follow-up this points to, not yet run

The genuinely informative version of this experiment pairs policies that
C2's own indifference-set output (`ranking_flip.indifference_set`) already
found to be *close* — e.g. two policies that already land in the same
indifference set under full-resolution bootstrap resampling are exactly
the ones a real analyst would worry could flip under coarser real-world
data too. That run needs an actual P3 result at meaningful scale to know
which pair is close enough to be interesting (the project's own README
notes P3 hasn't been run at its planned real scale yet) — worth doing once
that exists, using this same `coarsening_ladder.py` machinery unchanged.

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
