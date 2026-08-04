# Does this project's baseline ladder ever produce a real indifference set?

**Status: searched across four separate rigorous bootstrap-CRN
experiments. Answer, so far: no — not once, not even for the closest
methodological pair in the ladder.**

## Why this question, and why it matters

The plan's central thesis is that policy rankings are often *not*
statistically identified once input-model estimation error is propagated
— C2's whole machinery exists to detect exactly that. The plan's own risk
table treats the opposite outcome as a legitimate, still-valuable result,
not a failure: *"Rankings turn out stable → Publish the MDE curve and the
threshold. Still a contribution: 'here is when you're allowed to believe a
2% claim.' Reframe title toward the identification threshold."* Before
reframing anything, the question needed an actual answer, not an
assumption either way — this is that search.

## What was run

Four full bootstrap-CRN experiments, all using the tuned B1-B4 parameters
from `analysis/policy_tuning_run.py` (see `docs/` — no hand-picked
defaults):

1. **B0-B4, fleet_size=30** (`tuned_ranking_flip_run.py`, B=40): B2
   (value-corrected) wins every bootstrap draw. Indifference set at
   α=0.05: `{B2}` only.
2. **B0-B4, fleet_size=15** (tight fleet) (`fleet_size_sensitivity_run.py`,
   B=25): same — B2 wins every draw, indifference set `{B2}`.
3. **B0-B4, fleet_size=60** (generous fleet) (same script): same again.
4. **B3 vs. B4 head-to-head, fleet_size=30** (`b3_vs_b4_run.py`, B=50) —
   the closest pair *methodologically possible* in this ladder: both wrap
   the identical tuned B2 dispatch logic and differ only in repositioning
   heuristic (B3's smooth fluid target allocation vs. B4's one-sample
   Monte Carlo lookahead). Still no indifference: B4 wins all 50 draws,
   indifference set `{B4}` only.

Run (4) is the most informative one — its own variance decomposition
(`within_theta_variance=102.2`, `across_theta_variance=27.6`,
`input_uncertainty_ratio=0.27`) shows *intrinsic simulation noise actually
exceeds input-model uncertainty* for this comparison, and the ranking
**still** never flips across a single bootstrap draw. The B3-vs-B4 gap is
large enough to swamp both noise sources at once, not narrowly surviving
one of them.

## What this means

Not a refutation of the project's methodology — the opposite: C2's
machinery is doing exactly its job, correctly reporting high confidence
when the underlying gap really is large. The informative finding is
*why* it's always large here: this project's baseline ladder (B0-B4)
compares structurally different **dispatch mechanisms** — greedy
single-assignment vs. globally-optimal batched assignment vs.
value-function-corrected assignment vs. two different repositioning
heuristics layered on top. That's a bigger jump than what the ride-hailing
literature typically reports margins between. P0.1's novelty check found
a real applied paper reporting "82.3s vs. 85.3s vs. 85.8s" — a few percent
apart, plausibly *tuning variants of one approach*, not different
mechanisms. This project's own ladder, once each policy is actually tuned
to its own best configuration (not just given an arbitrary fixed
parameter), doesn't produce comparisons that close.

**This reframes, rather than undermines, where input uncertainty is a
real risk for this project specifically**: not for "does batched
assignment beat greedy," which is answered confidently here, but for (a)
close variants *within* one mechanism (e.g., B2 at different
`value_weight` settings — not yet tried), (b) the literature's own
reported comparisons, which this project cannot re-run directly but whose
margins look closer than anything found here, and (c) what happens under
real, coarser Bengaluru-resolution data (C6) or a smaller real-world
effect size than this synthetic ladder produces. The MDE curve
(`minimum_detectable_effect_curve`, empirically checked in
`docs/mde_scaling_validation.md`) is the tool for exactly that last
question, and is arguably now the more central deliverable for this
specific ladder than the ranking-flip result itself — consistent with the
plan's own contingency for this outcome.

## What wasn't tried

A within-mechanism sweep — e.g. B2 at several `value_weight` values
against each other, or B1 at several `matching_radius_seconds` values
against each other — is the natural next place to look for a genuinely
close pair, since it directly varies *how much* of one mechanism's
strength is applied rather than comparing different mechanisms outright.
Not run in this pass.
