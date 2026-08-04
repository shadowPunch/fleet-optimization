# Census of India / BBMP data for C5 (distributional accounting)

**Bottom line:** real, ward-level demographic data for Bengaluru exists and
is directly downloadable — but it's from the 2011 Census (15 years stale
as of 2026, and the next round won't publish usable tables for years), and
it's tied to ward boundaries that have since been redrawn **at least three
times**, most recently by a full governance restructuring that replaced
BBMP outright in September 2025. Joining it to any current ward-level trip
data (Namma Yatri's included) needs a real GIS crosswalk, not an ID join.
That crosswalk is buildable — the boundary files exist — but it's a
genuine, nontrivial piece of work, not a formality.

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

This is solvable, not fatal: ward boundary shapefiles exist for multiple
eras —
[Datameet's municipal spatial data project](http://projects.datameet.org/Municipal_Spatial_Data/bangalore/),
the [OpenBangalore GitHub shapefiles](https://github.com/openbangalore/bangalore/blob/master/bangalore/GIS/bbmpwards/bbmpwards.shp),
and [BBMP/GBA's own GISViewer](https://bbmp.gov.in/gisviewer/) — enough to
build a proper spatial crosswalk (area-weighted or population-weighted
reallocation of 2011 ward attributes onto current ward polygons) rather
than guessing at a correspondence. That crosswalk itself is a source of
additional approximation error and should be reported as one, consistent
with how every other data limitation in this project is handled (see
`docs/observability_table.md`) — not smoothed over as a clean join.

## What this actually gives C5

Once the crosswalk exists, the Census 2011 extract supports:

- Total population and SC/ST population share per ward — the closest
  available official proxy for the plan's original "race/income quantile"
  framing, though SC/ST status is not equivalent to the US race categories
  that framing was written around, and shouldn't be presented as if it
  were a direct analog.
- Household amenities (electricity, piped water, vehicle ownership) as a
  socioeconomic-status proxy, since Census doesn't publish household
  income directly.
- Literacy rate and worker-participation rate as additional SES proxies.

What it does **not** support: any caste breakdown finer than SC/ST, any
income figure, or anything from 2025/2027 — the plan's C5 section should
be scoped around 2011-vintage, SC/ST-and-amenities-based demographics, not
a richer profile it can't actually get.

## Recommendation

1. Use the OpenCity Census-2011 ward CSVs as the demographic source for C5.
2. Build the spatial crosswalk from 2011-era ward boundaries to whichever
   boundary Namma Yatri's data currently reports against — check this
   directly against a fresh scrape before assuming it's the 369-ward GBA
   structure.
3. Report the crosswalk's own approximation error alongside the C5
   results, not as a footnote.
