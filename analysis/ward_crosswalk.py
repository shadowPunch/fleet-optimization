"""Build the C5 ward-boundary crosswalk: 2011 Census wards -> 2025 GBA wards.

See TECHNICAL_REPORT.md for the full background. In short: the only
usable ward-level demographic data for Bengaluru is tied to the ~198-ward
geometry the 2011 Census was tabulated against, but Namma Yatri's open data
almost certainly reports against the current 369-ward Greater Bengaluru
Authority (GBA) scheme (effective September 2025) — three boundary
redrawings later. Joining the two on ward ID would silently join the wrong
polygon to itself, so this script builds a real spatial crosswalk using
`dispatch_eval.geo.crosswalk.areal_interpolate`.

Two things this script establishes, not assumed going in:

1. The GBA's own 2025 delimitation file already carries total/SC/ST
   population *re-tabulated onto the new 369 wards* (columns TOT_P, SC_P,
   ST_P) — almost certainly the official reallocation done as part of the
   legally-required delimitation process, not something this script needs
   to compute. Those columns are used directly, and also used as a ground
   truth to validate this script's own areal-interpolation method (which is
   the only option for every Census variable the GBA file does *not*
   carry — e.g. household amenities, literacy, worker categories).
2. The GBA file's `ward_id` field is NOT globally unique — it restarts at 1
   within each of the 5 city corporations (East/West/Central/North/South).
   The unique target key used throughout is `Corporation-ward_id`.

Data sources (all public, no auth, see TECHNICAL_REPORT.md for URLs):
  - Census 2011 ward CSV: OpenCity "Bengaluru Ward-wise Census Data 2011"
  - 2011-era ward geometry (198 wards): Datameet Municipal_Spatial_Data,
    BBMP_oldWards.geojson — geometry only; its own embedded POP_TOTAL does
    NOT match the Census CSV (5.84M vs 8.44M) and must not be used, only
    the polygons, joined to the CSV via WARD_NO = Ward Num.
  - Current ward geometry (369 wards): OpenCity "GBA Wards Delimitation
    2025 - Final Wards Map", gba-369-wards-december-2025.kml

Usage: uv run python analysis/ward_crosswalk.py
(expects the four files above already downloaded into analysis/cache/ —
see TECHNICAL_REPORT.md for exact URLs; not auto-fetched by this
script because the KML source requires following OpenCity's dataset page
rather than a stable direct link)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dispatch_eval.geo.crosswalk import areal_interpolate, interpolation_error

CACHE = Path(__file__).parent / "cache"
CENSUS_CSV = CACHE / "bengaluru_ward_census_2011.csv"
OLD_WARDS_GEOJSON = CACHE / "ward_boundaries" / "bbmp_oldwards_2012.geojson"
GBA_WARDS_KML = CACHE / "ward_boundaries" / "gba_369_wards_2025.kml"
OUTPUT_CROSSWALK = CACHE / "ward_crosswalk_2011_to_gba369.csv"
OUTPUT_VALIDATION = CACHE / "ward_crosswalk_validation.json"

# Every numeric Census variable to reallocate. Assembly constituency and
# Ward Name are categorical/identifying, not reallocated.
CENSUS_VALUE_COLS = [
    "Population",
    "Male",
    "Female",
    "SC Population",
    "SC male",
    "SC Female",
    "ST Population",
    "ST Male",
    "ST Female",
]


def load_2011_wards() -> gpd.GeoDataFrame:
    """2011 ward polygons, attributed with the Census CSV (not the geojson's own figures)."""
    geometry = gpd.read_file(OLD_WARDS_GEOJSON)[["WARD_NO", "geometry"]]
    geometry["WARD_NO"] = geometry["WARD_NO"].astype(int)

    census = pd.read_csv(CENSUS_CSV)

    matched = set(geometry["WARD_NO"]) & set(census["Ward Num"])
    if len(matched) != len(census):
        raise ValueError(
            f"expected every Census ward to have a matching 2011 boundary polygon, "
            f"got {len(matched)}/{len(census)} matched"
        )

    merged = geometry.merge(census, left_on="WARD_NO", right_on="Ward Num", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)


def load_gba_wards() -> gpd.GeoDataFrame:
    """Current (369-ward) GBA polygons, with a globally unique ward key and numeric population fields."""
    gba = gpd.read_file(GBA_WARDS_KML, driver="KML")

    if gba["ward_id"].nunique() == len(gba):
        raise AssertionError(
            "ward_id is now globally unique in the GBA source — the "
            "Corporation-ward_id composite key this script builds is no "
            "longer necessary; simplify back to plain ward_id."
        )
    gba["gba_ward_key"] = gba["Corporation"] + "-" + gba["ward_id"].astype(str)
    if gba["gba_ward_key"].nunique() != len(gba):
        raise ValueError("Corporation-ward_id is not a unique key for the GBA ward set")

    for col in ("TOT_P", "TOT_M", "TOT_F", "SC_P", "SC_M", "SC_F", "ST_P", "ST_M", "ST_F"):
        gba[col] = pd.to_numeric(gba[col], errors="coerce")

    return gba[
        ["gba_ward_key", "ward_name", "Corporation", "zone_name", "TOT_P", "SC_P", "ST_P", "geometry"]
    ]


def geometry_coverage(wards_2011: gpd.GeoDataFrame, wards_gba: gpd.GeoDataFrame) -> dict:
    """What fraction of each current ward's area is covered by *any* 2011 ward.

    Distinguishes the two possible causes of interpolation error: incomplete
    source coverage (some GBA ward sits partly outside the old boundary
    entirely, so its "missing" population can never be recovered by area
    weighting) versus non-uniform density within a fully-covered source
    ward (area weighting's actual assumption failing). Low coverage would
    mean the former; this project's data has neither — see the docstring.
    """
    old_proj = wards_2011.to_crs("EPSG:32643")
    gba_proj = wards_gba.to_crs("EPSG:32643")
    old_union = old_proj.union_all()

    gba_area = gba_proj.geometry.area
    covered_area = gba_proj.geometry.intersection(old_union).area
    coverage_frac = covered_area / gba_area

    return {
        "old_wards_total_area_km2": float(old_proj.geometry.area.sum() / 1e6),
        "gba_wards_total_area_km2": float(gba_area.sum() / 1e6),
        "mean_coverage_fraction": float(coverage_frac.mean()),
        "min_coverage_fraction": float(coverage_frac.min()),
        "n_wards_below_50pct_covered": int((coverage_frac < 0.5).sum()),
    }


def build_crosswalk() -> tuple[pd.DataFrame, dict]:
    wards_2011 = load_2011_wards()
    wards_gba = load_gba_wards()

    interpolated = areal_interpolate(
        source=wards_2011,
        target=wards_gba,
        value_cols=CENSUS_VALUE_COLS,
        source_id_col="WARD_NO",
        target_id_col="gba_ward_key",
    )

    reference = wards_gba[["gba_ward_key", "ward_name", "Corporation", "zone_name", "TOT_P", "SC_P", "ST_P"]]
    crosswalk = interpolated.merge(reference, on="gba_ward_key", how="left")

    validation = {
        "population_vs_official_TOT_P": interpolation_error(
            interpolated, reference, "Population", "TOT_P", "gba_ward_key"
        ),
        "sc_population_vs_official_SC_P": interpolation_error(
            interpolated, reference, "SC Population", "SC_P", "gba_ward_key"
        ),
        "st_population_vs_official_ST_P": interpolation_error(
            interpolated, reference, "ST Population", "ST_P", "gba_ward_key"
        ),
        "geometry_coverage": geometry_coverage(wards_2011, wards_gba),
        "n_source_wards_2011": len(wards_2011),
        "n_target_wards_gba": len(wards_gba),
        "n_target_wards_with_any_overlap": len(interpolated),
    }
    return crosswalk, validation


def main() -> None:
    for path in (CENSUS_CSV, OLD_WARDS_GEOJSON, GBA_WARDS_KML):
        if not path.exists():
            raise SystemExit(
                f"Missing required input: {path}. See TECHNICAL_REPORT.md for download URLs."
            )

    print("Building areal-interpolation crosswalk (2011 wards -> 369 GBA wards)...")
    crosswalk, validation = build_crosswalk()

    OUTPUT_CROSSWALK.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUTPUT_CROSSWALK, index=False)
    OUTPUT_VALIDATION.write_text(json.dumps(validation, indent=2))

    print(f"\nWrote {len(crosswalk)} crosswalked wards to {OUTPUT_CROSSWALK}")
    print(f"Wrote validation metrics to {OUTPUT_VALIDATION}")

    cov = validation["geometry_coverage"]
    print(
        f"\nGeometry coverage: mean {cov['mean_coverage_fraction']:.1%} of each current ward's "
        f"area falls inside the 2011 boundary (min {cov['min_coverage_fraction']:.1%}, "
        f"{cov['n_wards_below_50pct_covered']} wards below 50%) — old={cov['old_wards_total_area_km2']:.1f} "
        f"km2 vs current={cov['gba_wards_total_area_km2']:.1f} km2. High coverage means per-ward error "
        f"below is from non-uniform density within source wards, not missing source area."
    )
    print(f"\n{validation['n_target_wards_with_any_overlap']}/{validation['n_target_wards_gba']} "
          f"current wards have any overlap with the 2011 boundary.")

    for key in (
        "population_vs_official_TOT_P",
        "sc_population_vs_official_SC_P",
        "st_population_vs_official_ST_P",
    ):
        m = validation[key]
        print(
            f"\n=== {key} ===\n"
            f"  matched {m['n_wards_matched']} wards, "
            f"total interpolated={m['total_interpolated']:.0f} vs official={m['total_reference']:.0f}\n"
            f"  mean abs % error={m['mean_abs_pct_error']:.1%}, "
            f"median={m['median_abs_pct_error']:.1%}, max={m['max_abs_pct_error']:.1%}"
        )


if __name__ == "__main__":
    main()
