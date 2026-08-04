# P0.1 — Novelty check

**Status: done, provisional GO.** This project's central claim — that bootstrapping
input models, propagating that uncertainty through common-random-numbers
simulation, and reporting a ranking-flip distribution / indifference set
across *dispatch policies* is missing from the ride-hailing literature —
was not contradicted by this search. No paper combining all of that for
ride-hailing/mobility-on-demand turned up. Proceed with C2 as planned.

**Caveat on method, stated plainly:** this was a web-search-based literature
scan (general web search plus targeted queries against the plan's exact
search strings), not a true Google Scholar / Scopus / Web of Science
citation-graph traversal, and not a direct search of INFORMS PubsOnline or
the WSC archive's own interfaces. It will miss paywalled or poorly-indexed
papers and anything using different terminology than what was tried. Treat
this as the fast, cheap version of P0.1, not the final word — before
submitting anywhere, redo this with actual Scholar "cited by" chains on the
five papers below and a direct WSC-archive search.

## Searches run

The plan's five required strings, verbatim:

1. `"input uncertainty" ride-hailing simulation`
2. `"ranking and selection" ride-hailing OR mobility-on-demand OR dispatch`
3. `bootstrap "input model" dispatch policy comparison`
4. `"probability of correct selection" fleet OR dispatch OR mobility`
5. `"common random numbers" ride-hailing simulation policy`

Plus a forward-citation sweep (via web search, not Scholar) on the four
papers the plan names, and four additional targeted queries combining
"indifference set," "variance decomposition," and recent (2024-2026)
ride-hailing/fleet simulation work. Eleven searches total.

## Closest five papers, and what each does not do

1. **Song, E., Nelson, B.L., Hong, L.J. — "Input uncertainty and
   indifference-zone ranking & selection"** (WSC 2017; extended as
   "Data-Driven Ranking and Selection Under Input Uncertainty," *Operations
   Research*, 2022+). This is the closest methodological match — it's
   where "indifference zone" and input-uncertainty-aware R&S meet. **Does
   not**: touch ride-hailing, dispatch, or any spatial/queueing system —
   stays in the general finite-alternative simulation-optimization setting,
   with no notion of a spatial fleet, zones, or a dispatch policy at all.

2. **Barton, R.R., Nelson, B.L., Xie, W. (2014) — "Quantifying Input
   Uncertainty via Simulation Confidence Intervals."** *INFORMS Journal on
   Computing* 26(1):74-87. The metamodel-assisted bootstrap this project's
   own computational fallback is named after. **Does not**: compare
   *multiple* competing systems or policies at all — it builds a confidence
   interval for **one** system's mean performance under input uncertainty;
   no ranking, no indifference set, no ride-hailing application.

3. **Fan, W., Hong, L.J., Zhang, X. (2013) — Robust Selection of the Best
   (RSB).** Formulates input uncertainty as a discrete ambiguity *set* and
   selects the best worst-case alternative. **Does not**: use a bootstrap
   over real data to characterize uncertainty (it assumes a given finite
   set of plausible input models rather than resampling to generate one);
   does not use common random numbers to sharpen a paired ranking-flip
   distribution; no ride-hailing or mobility application.

4. **Lam, H. — "Cheap Bootstrap for Input Uncertainty Quantification."**
   WSC 2022 (+ *A Cheap Bootstrap Method for Fast Inference*, arXiv:2202.00090).
   The computational-efficiency angle most relevant if this project's own
   B×R×|policies| budget becomes the bottleneck the plan's risk table
   anticipates. **Does not**: address multi-alternative comparison at
   all — it's single-system CI construction with reduced outer-sample
   count; no dispatch, fleet, or spatial application.

5. **Kumar, I., Tiwari, A. — "Regime-Calibrated Fleet Repositioning with a
   Spatial Queue-Regret Decomposition."** arXiv, 2026 (evaluated on eight
   NYC scenarios). The closest *applied* ride-hailing paper found — it
   compares repositioning approaches in a common simulator and uses
   "simulator wait under frozen seeds" as a selection criterion, which
   sounds like common random numbers on the surface. **Does not**:
   quantify or propagate *input-model estimation uncertainty* — the frozen
   seeds are used for internal reproducibility when selecting among the
   paper's own retrieval/calibration variants, not as a CRN device paired
   with bootstrapped input models to compare independent dispatch
   policies; reports point-estimate mean wait per scenario (82.3s vs.
   85.3s vs. 85.8s), not a ranking-flip distribution, P(ranked first), or
   an indifference set.

## What this means for the project

No pivot needed per the plan's own decision rule ("if someone has already
done C2 for ride-hailing, pivot to C4+C3"). The gap between the
input-uncertainty/R&S methodology literature (papers 1-4, all
application-agnostic) and the ride-hailing dispatch literature (which
almost universally treats input models as fixed/known when comparing
policies, per paper 5 and the broader search) looks real. C2 remains the
paper. Re-run the more rigorous version of this check (real Scholar
citation chains, WSC archive, INFORMS PubsOnline) before submission —
this pass is good enough to keep building on, not good enough to cite as
"confirmed novel" in a paper.
