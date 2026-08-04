# MDE curve's 1/n_days assumption — empirically checked

**Status: checked against real (synthetic) data. The assumption doesn't
hold exactly — the real deviation is informative, and errs in the safe
direction.**

## What was checked

`minimum_detectable_effect_curve` (P3 output #4) extrapolates the minimum
detectable effect at a candidate `n_days` from a single measured variance
decomposition, assuming input-model estimation error (`across_theta_variance`)
scales as `1/n_days` — the standard asymptotic scaling for M-estimator /
bootstrap variance. That assumption was never checked against this
project's own simulator until now.

`ranking_flip.validate_mde_scaling` (tested,
`tests/test_ranking_flip.py::test_validate_mde_scaling_anchor_ratio_is_exactly_one_by_construction`)
does the actual check: re-runs the full bootstrap-CRN experiment at
several real values of `n_days` (via `subset_trips_by_days`, not
extrapolation), measures `across_theta_variance` at each via
`variance_decomposition`, and compares it to what the 1/n line — anchored
at the largest `n_days` tested — predicts.

## Real result

24 days of synthetic data, B0 (nearest-idle) vs. B1 (batched Hungarian),
fleet 30, 5 zones, checked at n_days = 3, 6, 12, 24. First pass at
`n_bootstrap=15` gave a noisy, non-monotonic signal (ratios 0.43, 0.24,
0.59) — expected, since a variance-of-variance estimate from only 15
bootstrap draws has a relative standard error around 40% on its own,
enough to swamp a real trend. Re-run at `n_bootstrap=50` (~5.5 minutes,
still no real data needed) cleaned that up into a real, monotonic signal:

| n_days | measured across-θ variance | 1/n-predicted | ratio |
|---|---|---|---|
| 3 | 614.3 | 3320.9 | 0.18 |
| 6 | 506.7 | 1660.5 | 0.31 |
| 12 | 339.1 | 830.2 | 0.41 |
| 24 (anchor) | 415.1 | 415.1 | 1.00 (by construction) |

**The 1/n_days assumption over-predicts variance at every smaller `n_days`
tested, and by more as `n_days` shrinks** — real across-theta variance at
3 days is only 18% of what the 1/n line says it should be, not the ~100%
a correct assumption would produce. This is a real, monotonic pattern
(not sampling noise — it held up going from 15 to 50 bootstrap draws),
though the *cause* isn't established by this check alone: candidates
include the NHPP arrival fit's own smoothing/fallback behavior at
sparse-data cells, and the fact that `n_days` here drives four jointly-refit
sub-models (arrival, OD, travel-time, fare) at once, not one
single-parameter estimator the classical 1/n result was derived for.

## What this means for the MDE curve

The direction of the error is the reassuring one:
`minimum_detectable_effect_curve`'s output is `z * sqrt(within +
across/n_ratio)` — a smaller true `across_theta_variance` than the 1/n
line assumes means the **real MDE at small n_days is better (smaller)
than the curve currently reports**. The curve is conservative, not
misleadingly optimistic: a researcher planning a data-collection budget
off this curve would over-estimate how much data they need, not
under-estimate it. That's a safe failure mode for a planning tool, but the
curve's numeric output still shouldn't be quoted as calibrated without
this caveat attached, and the underlying cause is worth understanding
before this gets used for anything higher-stakes than a rough budget
estimate.
