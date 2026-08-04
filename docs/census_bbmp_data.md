# Census of India / BBMP data for C5 (distributional accounting)

**Bottom line:** real, ward-level demographic data for Bengaluru exists and
is directly downloadable — but it's from the 2011 Census (15 years stale
as of 2026, and the next round won't publish usable tables for years), and
it's tied to ward boundaries that have since been redrawn **at least three
times**, most recently by a full governance restructuring that replaced
BBMP outright in September 2025. Joining it to any current ward-level trip
data (Namma Yatri's included) needs a real GIS crosswalk, not an ID join.

**Update: the crosswalk is now built** — `analysis/ward_crosswalk.py`,
backed by `src/dispatch_eval/geo/crosswalk.py` (tested,
`tests/test_crosswalk.py`). Two findings changed the plan from what was
anticipated below:

1. The current-era (369-ward) boundary turned out to be readily available
   after all (OpenCity's official December-2025 GBA delimitation KML) —
   the "may not be readily available yet" concern this doc raised
   previously did not materialize.
2. That same GBA file already carries **official total/SC/ST population
   re-tabulated onto the new 369 wards** (`TOT_P`/`SC_P`/`ST_P`), which
   matches the 2011 Census citywide totals to within 0.3% — almost
   certainly the legally-required delimitation process's own reallocation.
   **C5 should use those columns directly for population/SC/ST**, not this
   project's own areal interpolation. The interpolation crosswalk is still
   needed — and was validated against those same official columns — for
   the Census variables the GBA file does *not* carry (household
   amenities, literacy, worker categories): see "Crosswalk results" below
   for its measured, non-trivial per-ward error on those.

## What's actually available

**Census 2011, ward level (the only usable demographic source):**

- [OpenCity — Bengaluru Ward-wise Census Data 2011](https://data.opencity.in/dataset/bengaluru-census-2011):
  directly downloadable CSVs —
  - total population, SC (Scheduled Caste), and ST (Scheduled Tribe)
    population per ward, sex-disaggregated
  - household assets/amenities (electricity, water, sanitation, vehicle
    ownership) — usable as a socioeconomic-status proxy, since Census
    doesn't publish income directly
  - housing/houselisting data (house type, utilities)
  - ST household amenities disaggregated to ward/panchayat/tehsil
- [data.gov.in — Village/Town-wise Primary Census Abstract 2011, Karnataka](https://www.data.gov.in/catalog/villagetown-wise-primary-census-abstract-2011-karnataka):
  broader state file (needs filtering to Bengaluru), adds age-group
  population, literacy status, and worker-category breakdowns not in the
  OpenCity extract
- [Census India NADA — District Census Handbook, Bangalore (Series 30, Part XII A)](https://censusindia.gov.in/nada/index.php/catalog/584):
  the most authoritative/detailed official source, but PDF/table-based
  rather than a clean CSV

**Not usable, for different reasons:**

- **SECC 2011 caste data** — never publicly released, by policy decision;
  only the SC/ST flags (already in the regular Census) are available. No
  finer caste breakdown exists in any official, released dataset.
- **Karnataka Socio-Economic and Education Survey 2015** (the state's own
  "caste census") — completed in 2024/tabled to Cabinet April 2025, but
  its public release has been politically contentious; a "fresh survey"
  was reportedly directed in 2025 rather than releasing the existing one
  cleanly. Not a stable public dataset to build on right now.
- **Census 2027** (the delayed "2021 round") — house-listing just started
  (April-September 2026), population-enumeration reference date is March
  2027; published, usable tabulations are realistically years away. Not a
  near-term option.

## The ward-boundary crosswalk problem

Bengaluru's ward boundaries have been redrawn repeatedly since 2011, and
the local governance structure itself changed during this project's own
timeline:

| Era | Wards | Governing body |
|---|---|---|
| ~2011 Census | ~198 | BBMP (earlier boundary revision) |
| 2022 delimitation | 243 | BBMP |
| 2023 delimitation | 225 | BBMP |
| **2025 restructuring (current)** | **369** | **Greater Bengaluru Authority (GBA)**, replacing BBMP outright — 5 City Corporations (East/West/Central/North/South), fully operational since 2 Sept 2025 |

Namma Yatri's ward-level open data almost certainly reports against
whatever boundary is current at scrape time — likely the 369-ward GBA
structure now, not the ~198-ward geometry the 2011 Census was tabulated
against. **A direct ward-ID join between Namma Yatri's data and Census
2011 data would silently join the wrong geography to itself** — ward
number 42 in 2011 and ward number 42 today are not the same polygon.

This turned out to be solvable, not fatal — the boundary files exist and
the crosswalk is built. What actually worked, concretely:

- **2011-era geometry (198 wards)**: Datameet's
  [Municipal_Spatial_Data](https://github.com/datameet/Municipal_Spatial_Data)
  GitHub repo, `Bangalore/BBMP_oldWards.geojson`. Its own embedded
  `POP_TOTAL` field sums to 5.84M — it does **not** match the Census 2011
  citywide total (8.44M) and must not be used; only its polygons are used,
  joined to the authoritative OpenCity Census CSV via `WARD_NO` = `Ward
  Num` (clean 198/198 match, confirmed).
- **Current-era geometry (369 wards)**: found on OpenCity as the ["GBA
  Wards Delimitation 2025 - Final Wards Map"](https://data.opencity.in/dataset/gba-wards-delimitation-2025)
  dataset, `gba-369-wards-december-2025.kml` — the official December-2025
  GBA delimitation, not a third-party scrape. This was flagged above as
  possibly not readily available yet; it was in fact already published.
  **Gotcha**: the KML's `ward_id` field is per-corporation, not global — it
  restarts at 1 in each of the 5 city corporations, so 329 of 369 rows
  share a `ward_id` with a same-numbered ward elsewhere in the city. The
  real unique key is `Corporation-ward_id` (`src/dispatch_eval/geo` builds
  this explicitly rather than trusting `ward_id` alone).
- **The BBMP/GBA GISViewer itself** (`bbmp.gov.in/gisviewer`) was
  unreachable (connection failure) — not needed in the end, since the
  OpenCity KML above supplied the current boundary directly.

The [BBMP old wards geojson](https://github.com/datameet/Municipal_Spatial_Data)
and the GBA KML both load cleanly with `geopandas` (the KML via
`driver="KML"`; its numeric fields are typed as strings in the KML schema
and need `pd.to_numeric` before any arithmetic — a silent-string-concat
bug caught during development, not shipped).

### Crosswalk results

`analysis/ward_crosswalk.py` reallocates every 2011 Census numeric column
(population, sex, SC/ST, all sex-disaggregated) from the 198 old-ward
polygons onto the 369 current-ward polygons by standard area-weighting —
each source ward's value split across overlapping target wards in
proportion to intersection area, i.e. assuming uniform population density
within each 2011 ward. Two validations were run, not just asserted:

1. **Geometry coverage** — do the old and current boundaries actually
   cover the same ground? Yes: the current 369-ward union covers 717.2
   km², the 2011 198-ward union covers 711.6 km², and on average 99.6% of
   each current ward's own area falls inside the 2011 boundary (worst
   case: 52%, zero wards below 50%). So per-ward error below is *not*
   from missing source coverage (e.g. newly annexed land with no 2011
   analog) — GBA is essentially a re-delimitation of the same city, not a
   territorial expansion, at least by area.
2. **Accuracy vs. the GBA's own official reallocation** — compare this
   project's interpolated Population/SC-Population/ST-Population against
   the GBA KML's own `TOT_P`/`SC_P`/`ST_P` (see the finding above: those
   columns are themselves a Census-2011-consistent reallocation, so this
   is a real ground truth, not a second guess). At the **citywide** level
   the two agree closely (interpolated 8,398,719 vs. official 8,402,887,
   0.05% off) — confirming both datasets describe the same underlying
   population. At the **per-ward** level, area-weighting is noticeably
   worse: median absolute error 24.4% for population (52.8%/43.1% mean
   for SC/ST, which are smaller, patchier counts more sensitive to the
   uniform-density assumption). The worst-error wards are explainable, not
   noise — they're dense, built-up central neighborhoods (Indiranagar,
   Malleshwaram, Rajajinagar) that got split into several new wards
   covering unequal actual population despite roughly comparable area,
   exactly where "uniform density within the old ward" fails hardest.

**Conclusion**: area-weighting recovers the right citywide total but is a
genuinely crude per-ward estimator (~25% typical error) — good enough to
use *only* for the Census variables that have no official reallocation
(household amenities, literacy, worker categories), and every C5 result
built on those must carry this measured error band, not a vague caveat.
For population/SC/ST specifically, use the GBA KML's own official columns
directly and skip the interpolation error entirely. Full numeric detail is
in `analysis/cache/ward_crosswalk_validation.json` (gitignored — rerun
`uv run python analysis/ward_crosswalk.py` to regenerate; requires the raw
boundary/census files, not committed, links above).

### Household amenities: a second crosswalk, a genuinely different kind

`analysis/ward_amenities_crosswalk.py` extends this to household
amenities, using OpenCity's **"Bengaluru Housing and Houselisting Data,
Census 2011"** resource (same dataset page as the population CSV). Two
real findings from building it, not assumed going in:

1. **A same-page resource with a misleading name.** OpenCity also lists
   "Household Assets - Bangalore, Census 2011," which sounds like the
   obvious source for electricity/vehicle-ownership rates — checked
   directly, and it's actually tabulated at **district/tehsil/village**
   level (one row for "District - Bangalore," others for individual rural
   villages like "Gopalapura"), not BBMP wards at all. Not usable for this
   crosswalk; not used. The housing/houselisting file, by contrast, has a
   real `Ward No` column and genuinely is ward-level — verified by
   checking the actual data, not inferred from the resource's title.
2. **Every column in the usable file is a percentage**, not a count
   (`HH condition - total` = 100 for every ward). Population/SC/ST are
   *extensive* quantities — summing fragments across a reallocation is
   meaningful. A percentage is *intensive* — summing two wards' "80%
   electrified" and "40% electrified" produces a meaningless 120%. This
   needs an area-weighted **average**, not an area-weighted **sum**:
   `geo.crosswalk.areal_interpolate_weighted_average`, a new sibling to
   `areal_interpolate` (tested, `tests/test_crosswalk.py`), weighted by
   each overlap's share of the *target* ward's covered area rather than
   the source ward's own area.

A third finding, smaller but worth flagging: this file's BBMP ward
numbers run 1-209 (208 wards, gap at 201) — more than the population
CSV's 198. The extra ~10 wards have no matching polygon in the 2011
boundary geometry this crosswalk uses and are dropped, not guessed at
(logged explicitly by the script: `dropping 10 wards with no 2011
boundary geometry: [199, 200, 202, ..., 209]`). Plausibly these are areas
added to BBMP between the original ~198-ward delimitation and whenever
this housing file was compiled, still within the "2011 Census" vintage —
not confirmed against an authoritative source, stated as an open question
rather than resolved.

Real result, a curated subset of amenity rates (electricity, treated tap
water, two-wheeler/car ownership, latrine access, housing condition),
citywide mean before vs. after interpolation:

| Amenity | Source (198-ward) | Interpolated (369-ward) |
|---|---|---|
| Electrified | 98.4% | 98.3% |
| Treated tap water | 76.3% | 70.7% |
| Two-wheeler ownership | 47.1% | 46.3% |
| Car/jeep ownership | 19.2% | 19.1% |
| Latrine within premises | 96.7% | 96.6% |
| Housing condition "good" | 79.2% | 79.0% |
| Housing condition "dilapidated" | 1.1% | 1.1% |

Most amenities barely move (≤0.8 points) — but treated-tap-water access
shifts by 5.6 points, the one variable in this set with real spatial
heterogeneity (electricity is near-universal at ~98% everywhere, leaving
little room for an interpolation artifact to show up; water
infrastructure genuinely varies block to block). Worth treating that
specific variable's crosswalked numbers with more caution than the
others, not applying one blanket error bar to all of them.

## What this actually gives C5

With both crosswalks built, the Census 2011 extract supports:

- Total population and SC/ST population share per ward, at the current
  369-ward scheme, at official-reallocation accuracy (via the GBA KML's
  own columns) — the closest available proxy for the plan's original
  "race/income quantile" framing, though SC/ST status is not equivalent to
  the US race categories that framing was written around, and shouldn't be
  presented as if it were a direct analog.
- Household amenities (electricity, piped water, vehicle ownership,
  latrine access, housing condition) as a socioeconomic-status proxy,
  since Census doesn't publish household income directly — via the
  weighted-average crosswalk above; citywide means hold up well except for
  treated-water access, which should carry a wider error bar than the
  rest.
- Literacy rate and worker-participation rate as additional SES proxies
  are **still not obtained** — they're in the data.gov.in Karnataka PCA
  file specifically (not OpenCity, which was checked directly and doesn't
  carry them), and that portal's modern interface is a client-side-rendered
  SPA: `WebFetch` returned HTTP 403 on the catalog page, and `curl`
  against both the page itself and a guessed CKAN-style API endpoint
  returned only the SPA's empty HTML shell, no embedded data. A person
  with a browser could very likely still get this file by navigating the
  UI directly; scripted access specifically did not work in this
  environment after a real attempt, not a cursory one.

What it does **not** support: any caste breakdown finer than SC/ST, any
income figure, or anything from 2025/2027 — the plan's C5 section should
be scoped around 2011-vintage, SC/ST-and-amenities-based demographics, not
a richer profile it can't actually get.

## Recommendation

1. Use the OpenCity Census-2011 ward CSVs as the demographic source for C5.
2. ~~Build the spatial crosswalk~~ — done, twice over:
   `src/dispatch_eval/geo/crosswalk.py` has both `areal_interpolate`
   (extensive/count variables — population, SC/ST) and
   `areal_interpolate_weighted_average` (intensive/rate variables —
   household amenities), each tested. `analysis/ward_crosswalk.py` and
   `analysis/ward_amenities_crosswalk.py` apply them to Bengaluru's
   specific 2011-to-369-ward case. For population/SC/ST, skip the
   interpolation and use the GBA KML's own columns directly (see above).
3. Report the crosswalk's own approximation error alongside the C5
   results, not as a footnote — the measured numbers to cite are in
   "Crosswalk results" above: ~0.05-0.3% at the citywide level, ~25-53%
   median per-ward for population/SC/ST; citywide amenity means generally
   hold within a point except treated-water access (5.6-point shift,
   flagged above as the one variable needing a wider error bar).
4. **Still not obtained**: literacy and worker-participation rates
   (data.gov.in's Karnataka PCA file specifically — a real, documented
   access obstacle for scripted tools, not a cursory miss; see above).
5. **Still unverified**: which ward scheme Namma Yatri's actual open data
   uses. Everything above assumes it's the current 369-ward GBA scheme
   (the reasonable default, and what this crosswalk targets), but that
   assumption should be checked directly against a fresh scrape before
   C5's results are reported — if Namma Yatri turns out to still report
   against an older scheme (243 or 225 wards), the target side of this
   crosswalk needs to be rebuilt against that geometry instead.
