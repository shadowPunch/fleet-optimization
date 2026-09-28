"""C5 follow-up: crosswalk household-amenity rates (not covered by the
population/SC/ST crosswalk in `ward_crosswalk.py`) onto the current
369-ward GBA structure.

Source: OpenCity's "Bengaluru Housing and Houselisting Data, Census 2011"
(`analysis/cache/housing_houselisting_2011.csv`) — a genuinely ward-level
file (has a real `Ward No` column, unlike the sibling "Household Assets"
resource on the same OpenCity dataset page, which turned out to be
district/tehsil/village-level despite its name — checked directly, not
assumed; see TECHNICAL_REPORT.md).

**A different variable type than `ward_crosswalk.py` handles, requiring
different interpolation logic, found while building this**: every column
in this file is a *percentage* (`HH condition - total` is 100 for every
ward), not a count. Population/SC/ST are extensive quantities (summing
them across reallocated area fragments is meaningful); a percentage is
intensive (summing two wards' "80% electrified" and "40% electrified"
would produce a meaningless 120%). This needs an area-weighted *average*,
not an area-weighted *sum* — `geo.crosswalk.areal_interpolate_weighted_average`,
a new sibling function to the count-summing `areal_interpolate` that
`ward_crosswalk.py` uses.

**A ward-numbering mismatch, also found here, not assumed**: this file's
BBMP rows run 1-209 (208 wards excluding a Ward No=0 district-total row,
gap at 201), while the Census population CSV / 2011 boundary geometry
`ward_crosswalk.py` already uses only goes up to 198. Wards 199-209
(9 wards) have no matching geometry in the 2011 boundary file used here —
almost certainly areas added to BBMP after the original 198-ward
delimitation but before this housing file was compiled. Those 9 wards are
dropped (documented, not silently zero-filled) rather than guessing at
geometry for them.

Usage: uv run python analysis/ward_amenities_crosswalk.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dispatch_eval.geo.crosswalk import areal_interpolate_weighted_average

CACHE = Path(__file__).parent / "cache"
HOUSING_CSV = CACHE / "housing_houselisting_2011.csv"
OLD_WARDS_GEOJSON = CACHE / "ward_boundaries" / "bbmp_oldwards_2012.geojson"
GBA_WARDS_KML = CACHE / "ward_boundaries" / "gba_369_wards_2025.kml"
OUTPUT_CROSSWALK = CACHE / "ward_amenities_crosswalk_2011_to_gba369.csv"
OUTPUT_VALIDATION = CACHE / "ward_amenities_crosswalk_validation.json"

# A curated subset of the housing file's 145 columns, matching the plan's
# stated interest: electricity, piped water, vehicle ownership as SES
# proxies, plus latrine access and housing condition.
AMENITY_VALUE_COLS = [
    "Lighting - Electricity",
    "Drinking Water - Tapwater from treated source",
    "HH - Scooter/ Motorcycle/Moped",
    "HH - Car/ Jeep/Van",
    "Number of households having latrine facility within the premises",
    "HH condition - Good",
    "HH Condition - Dilapidated",
]


def load_2011_amenity_wards() -> gpd.GeoDataFrame:
    housing = pd.read_csv(HOUSING_CSV)
    bbmp = housing[housing["Area Name"].str.contains("BBMP") & (housing["Ward No"] > 0)].copy()

    geometry = gpd.read_file(OLD_WARDS_GEOJSON)[["WARD_NO", "geometry"]]
    geometry["WARD_NO"] = geometry["WARD_NO"].astype(int)

    matched = set(geometry["WARD_NO"]) & set(bbmp["Ward No"])
    dropped = sorted(set(bbmp["Ward No"]) - matched)
    if dropped:
        print(f"  dropping {len(dropped)} wards with no 2011 boundary geometry: {dropped}")
    bbmp = bbmp[bbmp["Ward No"].isin(matched)]

    merged = geometry.merge(bbmp, left_on="WARD_NO", right_on="Ward No", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)


def load_gba_wards() -> gpd.GeoDataFrame:
    gba = gpd.read_file(GBA_WARDS_KML, driver="KML")
    gba["gba_ward_key"] = gba["Corporation"] + "-" + gba["ward_id"].astype(str)
    return gba[["gba_ward_key", "ward_name", "Corporation", "geometry"]]


def main() -> None:
    for path in (HOUSING_CSV, OLD_WARDS_GEOJSON, GBA_WARDS_KML):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    print("Loading 2011 ward-level amenity data and boundaries...")
    wards_2011 = load_2011_amenity_wards()
    wards_gba = load_gba_wards()
    print(f"  {len(wards_2011)} source wards (2011), {len(wards_gba)} target wards (GBA 2025)")

    print("Area-weighted-averaging amenity rates onto the 369-ward GBA structure...")
    interpolated = areal_interpolate_weighted_average(
        source=wards_2011,
        target=wards_gba,
        value_cols=AMENITY_VALUE_COLS,
        source_id_col="WARD_NO",
        target_id_col="gba_ward_key",
    )
    crosswalk = interpolated.merge(
        wards_gba[["gba_ward_key", "ward_name", "Corporation"]], on="gba_ward_key", how="left"
    )

    OUTPUT_CROSSWALK.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUTPUT_CROSSWALK, index=False)

    validation = {
        "n_source_wards_2011": len(wards_2011),
        "n_target_wards_gba": len(wards_gba),
        "n_target_wards_with_any_overlap": len(interpolated),
        "citywide_mean_by_amenity": {
            col: float(crosswalk[col].mean()) for col in AMENITY_VALUE_COLS
        },
        "source_citywide_mean_by_amenity": {
            col: float(wards_2011[col].mean()) for col in AMENITY_VALUE_COLS
        },
    }
    OUTPUT_VALIDATION.write_text(json.dumps(validation, indent=2))

    print(f"\nWrote {len(crosswalk)} crosswalked wards to {OUTPUT_CROSSWALK}")
    print(f"Wrote validation metrics to {OUTPUT_VALIDATION}\n")
    for col in AMENITY_VALUE_COLS:
        print(
            f"  {col}: source citywide mean {validation['source_citywide_mean_by_amenity'][col]:.1f}%"
            f" -> interpolated citywide mean {validation['citywide_mean_by_amenity'][col]:.1f}%"
        )


if __name__ == "__main__":
    main()
