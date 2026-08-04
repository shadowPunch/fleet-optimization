# C4 — Compute parity: measured, not assumed

**Status: measurement half done; the "degrade + re-measure quality" half
needs a mechanism this codebase doesn't have yet (see below).**

## What was measured

`analysis/compute_parity_sweep.py` builds real B0-B4 policy instances
(fitted on synthetic data, same pattern as every other test in this
project — no real data needed, see `src/dispatch_eval/compute_parity.py`)
and times each one's `dispatch()` (+ `reposition()`, for B3/B4) call at
five synthetic problem sizes, from a light 10-request/10-vehicle tick up
to a deliberately extreme 600x600 one. Real numbers, this machine:

| policy | 10×10 | 50×50 | 100×100 | 300×300 | 600×600 |
|---|---|---|---|---|---|
| B0 nearest idle | 0.03ms | 0.48ms | 1.78ms | 16.2ms | 64.2ms |
| B1 batched Hungarian | 0.05ms | 0.98ms | 3.9ms | 34.9ms | 142.1ms |
| B2 value-corrected | 0.08ms | 1.7ms | 6.8ms | 61.8ms | 248.0ms |
| B3 fluid balancing | 0.10ms | 1.8ms | 7.1ms | 62.7ms | 253.0ms |
| B4 sampling lookahead | 0.18ms | 2.2ms | 7.5ms | 62.9ms | 251.8ms |

Against a real production-style cycle budget (the plan's own framing:
"production systems run on cycles of a couple of seconds"), **every
policy in the ladder fits comfortably**, even at 600 simultaneous open
requests and 600 idle vehicles — a batch size well beyond what a single
5-second dispatch tick would realistically accumulate. B0 stays under
budget even at a much tighter 100ms cycle; B1-B4 only cross 100ms at the
most extreme (600×600) synthetic size tested. **The crossover the plan
expects ("some optimal-in-principle methods lose once charged for their
latency") is real and visible in the scaling trend (B2-B4 cost ~4x what B0
costs at the same size, and all of B1-B4 scale roughly cubically — the
Hungarian solver's O(n³) is doing exactly what it says), but doesn't bind
at any realistic batch size against a multi-second budget.** It would
bind well before 600×600 against a tighter budget (say 100ms, plausible
for a system serving many cities from one dispatch process) — the table
above is what makes that a checkable claim instead of a guess.

## What wasn't built, and why

The plan's C4 also calls for: *"Degrade the ones that cannot [complete a
decision in budget] (truncate the candidate graph, coarsen the radius) and
re-measure quality. Plot quality vs. decision-latency budget."*

B1 and B2 already expose a `matching_radius_seconds` constructor
parameter that looks like exactly this knob. It **is not** — reading
`policies/batched_hungarian.py` shows it masks out-of-radius
(vehicle, request) pairs with an enormous cost, but the matrix
`scipy.optimize.linear_sum_assignment` actually solves is always
`max(n_requests, n_idle_vehicles)` square, regardless of the radius.
Confirmed empirically, not just by reading the code:
`tests/test_compute_parity.py::test_matching_radius_does_not_shrink_the_solved_cost_matrix`
spies on the exact matrix shape passed to the solver at two very different
radii and finds it identical. **Sweeping the radius changes assignment
quality, not wall-clock latency**, in this implementation.

A real "truncate the candidate graph" mechanism — one that actually
shrinks the matrix solved, e.g. restricting each request to its k nearest
idle vehicles before building the cost matrix — doesn't exist in this
codebase yet. Building it is a prerequisite for the "degrade + re-measure
quality" half of C4; `compute_parity.py`'s measurement infrastructure
(`measure_decision_latency`, `latency_frontier`, `feasible_within_budget`)
is ready to consume it once it exists — pair a degraded-vs-full-graph
policy pair with a `run_simulation` quality measurement, the same pattern
`decision_currency.py` (C3) already uses to pair a latency measurement
with a quality one.
