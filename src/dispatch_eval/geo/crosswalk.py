"""Areal-interpolation crosswalk between administrative boundary eras.

Ward boundaries are redrawn periodically (see docs/census_bbmp_data.md for
Bengaluru's specific history: ~198 wards at the 2011 Census, 243 in a 2022
delimitation, 225 in 2023, 369 under the September 2025 Greater Bengaluru
Authority restructuring). Any dataset tabulated against an older boundary
scheme needs its attributes reallocated onto whichever scheme a newer
dataset uses before the two can be joined on ward id — ward "42" in two
different eras is not the same polygon, so an ID join silently joins the
wrong geography to itself.

This module implements standard area-weighted ("areal") interpolation: each
source polygon's attribute value is split across the target polygons it
overlaps, in proportion to the fraction of the source polygon's *area* each
overlap covers. This assumes the attribute (e.g. population) is uniformly
distributed within each source polygon, which is never exactly true — it is
a documented approximation, not an exact reallocation, and its error should
be reported alongside any result that depends on it (see
`interpolation_error`, and `docs/census_bbmp_data.md`).

Where an authoritative official reallocation already exists for a variable
(as it does for Bengaluru's total/SC/ST population — the GBA's own 2025
delimitation file appears to already carry these figures re-tabulated onto
the new 369-ward scheme), prefer it directly and use this module only for
variables the official source lacks.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

DEFAULT_PROJECTED_CRS = "EPSG:32643"  # UTM zone 43N — covers Bengaluru / south India


def areal_interpolate(
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    value_cols: list[str],
    source_id_col: str,
    target_id_col: str,
    projected_crs: str = DEFAULT_PROJECTED_CRS,
) -> pd.DataFrame:
    """Reallocate `value_cols` from `source` polygons onto `target` polygons.

    Both GeoDataFrames must carry a CRS; both are reprojected to
    `projected_crs` so area is measured in real units (m^2), not degrees —
    computing area directly in lat/lon would silently distort every weight.
    The default is appropriate for Bengaluru/south India; pass a different
    UTM zone (or other equal-area/conformal projected CRS) for other
    regions.

    Returns one row per `target_id_col` with each of `value_cols` summed
    over all intersecting source polygons, weighted by
    (intersection area / source polygon area). Target polygons with no
    source overlap are absent from the result, not zero-filled.
    """
    if source.crs is None or target.crs is None:
        raise ValueError("source and target GeoDataFrames must both have a CRS set")
    if not value_cols:
        raise ValueError("value_cols must be non-empty")

    src = source[[source_id_col, *value_cols, "geometry"]].to_crs(projected_crs).copy()
    tgt = target[[target_id_col, "geometry"]].to_crs(projected_crs).copy()

    src["_source_area"] = src.geometry.area
    if (src["_source_area"] <= 0).any():
        raise ValueError("source contains zero- or negative-area geometries after reprojection")

    pieces = gpd.overlay(src, tgt, how="intersection", keep_geom_type=True)
    weight = pieces.geometry.area / pieces["_source_area"]

    for col in value_cols:
        pieces[col] = pieces[col] * weight

    return pieces.groupby(target_id_col, as_index=False)[value_cols].sum()


def interpolation_error(
    interpolated: pd.DataFrame,
    reference: pd.DataFrame,
    interpolated_col: str,
    reference_col: str,
    id_col: str,
) -> dict[str, float]:
    """Sanity-check `areal_interpolate`'s output against an authoritative reference.

    Used to validate the areal-interpolation *method* on a variable where an
    official reallocation already exists, before trusting it for variables
    that have no such reference. `id_col` must be present, under the same
    name, in both frames.
    """
    merged = interpolated[[id_col, interpolated_col]].merge(
        reference[[id_col, reference_col]], on=id_col, how="inner"
    )
    if merged.empty:
        raise ValueError(f"no overlapping '{id_col}' values between interpolated and reference")

    diff = merged[interpolated_col] - merged[reference_col]
    pct_err = (diff / merged[reference_col].replace(0, pd.NA)).abs()

    return {
        "n_wards_matched": len(merged),
        "total_interpolated": float(merged[interpolated_col].sum()),
        "total_reference": float(merged[reference_col].sum()),
        "mean_abs_pct_error": float(pct_err.mean()),
        "median_abs_pct_error": float(pct_err.median()),
        "max_abs_pct_error": float(pct_err.max()),
        "rmse": float((diff**2).mean() ** 0.5),
    }
