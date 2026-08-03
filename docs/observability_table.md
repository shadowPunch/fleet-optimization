# P0.2 — Observability audit

Every quantity the digital twin (P1) needs, against what the two available data
sources actually give us. This is Figure 1 of the paper: most of the bottom
half of this table is latent, and that is the honest foundation the rest of
the project rests on.

Two sources, deliberately kept apart because they are not interchangeable —
see [`dispatch-evaluation-project-plan.md`](../dispatch-evaluation-project-plan.md#2-data)
for why neither is an Indian-city equivalent of NYC TLC data:

- **Bengaluru** — Namma Yatri Open Data (nammayatri.in/open). Real, live,
  operational. Ward-level aggregates only.
- **Delhi NCR** — Kaggle `shayanzk/ola-ride-bookings-dataset`. Trip-level rows
  with the right *shape*, but treated as **synthetic** — unverified provenance,
  no disclosed data-generating process.

| Quantity | Observable — Bengaluru (real) | Observable — Delhi NCR (synthetic) | How it enters the simulator |
|---|---|---|---|
| Request arrivals | Ward-level counts per time window | Per-row booking timestamps | Fitted arrival-rate model (P1); Bengaluru gives a real marginal check, Delhi NCR gives the only bin-level rate curve |
| OD distribution | **Not observed** — published figures are per-ward marginals, no ward-to-ward flow | Pickup/drop location per row | OD matrix fit **only** from Delhi NCR; flagged synthetic-only in every downstream figure |
| Trip duration | **Not observed** | Implied by timestamps (unverified whether it's a true per-trip duration or a synthetically sampled field) | Feeds the travel-time model; synthetic-only |
| Trip distance | **Not observed** | Distance field per row | Feeds fare/pay regression; synthetic-only |
| Fare | Published fare *estimates* (aggregate, not per-trip) | Fare field per row | Regression fit on Delhi NCR; Bengaluru's aggregate range used as a plausibility check only |
| Driver pay | **Not observed** (Namma Yatri's "100% to driver" model means fare ≈ pay, but no per-trip figure is published) | **Not observed directly** — no separate driver-pay field disclosed in the dataset description | Approximated as a fraction of fare; treat as a modeling assumption, not a fitted quantity — state this explicitly in the report |
| Rider wait (request→pickup) | **Not observed at all** — no timestamp split exists in the published aggregates | Possibly proxied by a VTAT-style pickup-wait field, itself unverified as real vs. synthetically sampled | The single most important validation target in the original NYC-based plan; here it is either absent or of uncertain provenance — this is the headline entry in this table |
| Pickup distance | **Not observed** | Not directly disclosed; would need to be derived from pickup location + assigned-vehicle location, which the raw data doesn't carry | Not fittable from either source; treat as simulator-internal (computed from the ward/travel-time model), not calibrated |
| Fleet size | **Not observed** (platform-side, proprietary) | **Not observed** (dataset has no driver-supply field) | **Latent — calibrated** against whatever wait-time proxy is available (Delhi NCR only); see P1 calibration notes |
| Driver online/offline | **Not observed** | **Not observed** | Not modeled explicitly; folded into the fleet-size latent calibration |
| Acceptance rate | **Not observed** as a distinct stage (no "driver declined" field disclosed) | Cancellation fields exist but conflate several stages (customer cancel, driver cancel, incomplete) | Modeled coarsely via the cancellation taxonomy below; not separately identified |
| Cancellation | Aggregate cancellation count per ward | Row-level cancellation status + reason (Completed / Cancelled by Customer / Cancelled by Driver / Incomplete) | Cancellation taxonomy fit from Delhi NCR; Bengaluru's aggregate cancellation *rate* used as a real-world sanity check |
| Abandonment | **Not observed** — indistinguishable from "Cancelled by Customer" even in the synthetic set, and absent entirely from Bengaluru's aggregates | Folded into "Cancelled by Customer," not separately labeled | **Latent — structural ambiguity axis.** Run under 3–5 abandonment-hazard specifications per the original plan; this data landscape makes that axis more load-bearing, not less |
| Repositioning | **Not observed** by either source (no idle-vehicle movement data published or disclosed) | **Not observed** | Purely simulator-internal policy behavior (B3/B4), never a calibration target |

## Reading this table

Compared to the original NYC-TLC-based version of this audit, two things get
worse and none get better:

1. **Wait time — the single most important validation target — moves from
   "observable, censored" (NYC) to "not observable at all" (Bengaluru) or
   "observable only in an unverified synthetic proxy" (Delhi NCR).** This is
   the single biggest methodological consequence of the city switch and it
   must be stated in the report's limitations section, not discovered by a
   reviewer.
2. **OD structure, travel time, and trip distance are synthetic-only.** Every
   figure derived from them carries a synthetic-data flag through P1–P4.

This is consistent with the project's own thesis: if the argument is that
published effect sizes don't survive honest propagation of input-model
uncertainty, then starting from a data landscape where the inputs are *even
less* identified than the US case is a stronger test of that argument, not a
weaker one. The C6 coarsening ladder (see the plan) turns this into a
measured result — where, exactly, on the path from Delhi NCR's synthetic
trip-level resolution down to Bengaluru's real ward-aggregate resolution does
ranking identifiability collapse — rather than leaving it as a caveat.
