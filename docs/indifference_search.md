# Does this project's baseline ladder ever produce a real indifference set?

**Status: found. Not between different dispatch mechanisms (checked four
ways, always no) — between close settings of the *same* mechanism.**

**Update**: the "no indifference between mechanisms" result below was
checked against a real, sharp alternative explanation — that B2-B4's
apparent advantage is really an oracle-forecast artifact, since their
value function/reposition logic was built from models fit the same way
the true world is generated. `docs/forecast_degradation.md` rules that
out across nine degradation levels, including one removing all spatial
demand knowledge from the forecast entirely — the separation below holds
up, not explained away by that confound (though it opens its own, still
open question about *why* it holds up).

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

Run (4) is the most informative of the four — its own variance
decomposition (`within_theta_variance=102.2`, `across_theta_variance=27.6`,
`input_uncertainty_ratio=0.27`) shows *intrinsic simulation noise actually
exceeds input-model uncertainty* for this comparison, and the ranking
**still** never flips across a single bootstrap draw. The B3-vs-B4 gap is
large enough to swamp both noise sources at once, not narrowly surviving
one of them.

5. **A within-mechanism sweep: B2 at five `value_weight` settings against
   *each other*** (`value_weight_sweep_run.py`, B=30,
   `value_weight ∈ {0.0, 0.5, 0.75, 1.0, 1.5}`, `matching_radius_seconds`
   held fixed at B2's own tuned value so weight is the only thing that
   varies). **This one found it.** `weight_0.0` (no value correction at
   all) is clearly worse and excluded, but `{0.5, 0.75, 1.0, 1.5}` — a 3x
   range, including the actually-tuned value (0.75-ish) — are **all** in
   the indifference set at α=0.05, with P(ranked first) split
   `{0.5: 0.23, 0.75: 0.20, 1.0: 0.30, 1.5: 0.27}`, not concentrated on
   any one of them.

## What this means

Runs (1)-(4) are not a refutation of the project's methodology — the
opposite: C2's machinery is doing exactly its job, correctly reporting
high confidence when the underlying gap really is large. Those four all
compared structurally different **dispatch mechanisms** — greedy
single-assignment vs. globally-optimal batched assignment vs.
value-function-corrected assignment vs. two different repositioning
heuristics layered on top — a bigger jump than the ride-hailing literature
typically reports margins between (P0.1's novelty check found a real
applied paper reporting "82.3s vs. 85.3s vs. 85.8s," plausibly *tuning
variants of one approach*, not different mechanisms).

Run (5) is the confirmation that the search itself was sound, not just
unlucky: indifference **does** show up in this project's simulator, right
where the hypothesis said to look — inside one mechanism's own strength
dial, once it's doing "enough" correction. `value_weight` above ~0.5
apparently saturates for this scenario; below that, correcting for future
value at all matters, but exactly how strongly doesn't, at least not at a
level this experiment's precision can resolve.

**This gave C6 the close pair it was missing, and that follow-up is now
run too.** `docs/coarsening_ladder.md`'s original run (B0 vs. B1) was
cleanly separated at every rung — not a strong test. Re-running the exact
same ladder on `weight_0.5` vs. `weight_1.5` found the sharper result:
the bootstrap-majority winner disagrees with the single nominal fit's
point estimate at *every* rung (P(weight_1.5 ranked first) = 0.60-0.70
throughout, while the nominal fit alone points the other way) — a real,
not hypothetical, instance of exactly the failure mode this project's
whole "report the distribution, not a point estimate" methodology exists
to catch. See `docs/coarsening_ladder.md` for the full table.
