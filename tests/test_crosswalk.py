from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from dispatch_eval.geo.crosswalk import (
    areal_interpolate,
    areal_interpolate_weighted_average,
    interpolation_error,
)

CRS = "EPSG:32643"


def _gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, crs=CRS)


@pytest.fixture
def source() -> gpd.GeoDataFrame:
    # Two adjacent 1x1 unit squares, "uniform density" of 100 and 200 per unit area.
    return _gdf(
        [
            {"ward_id": "A", "population": 100.0, "geometry": box(0, 0, 1, 1)},
            {"ward_id": "B", "population": 200.0, "geometry": box(1, 0, 2, 1)},
        ]
    )


def test_splits_value_proportionally_to_overlap_area(source):
    target = _gdf(
        [
            # straddles A/B evenly: half of A's 100 + half of B's 200
            {"new_ward": "X", "geometry": box(0.5, 0, 1.5, 1)},
            # fully inside A
            {"new_ward": "Y", "geometry": box(0, 0, 0.5, 1)},
        ]
    )

    result = areal_interpolate(source, target, ["population"], "ward_id", "new_ward")
    result = result.set_index("new_ward")

    assert result.loc["X", "population"] == pytest.approx(150.0)
    assert result.loc["Y", "population"] == pytest.approx(50.0)


def test_conserves_total_value_when_target_fully_covers_source(source):
    # Two target polygons that together exactly tile the two source squares.
    target = _gdf(
        [
            {"new_ward": "P", "geometry": box(0, 0, 1.5, 1)},
            {"new_ward": "Q", "geometry": box(1.5, 0, 2, 1)},
        ]
    )
    result = areal_interpolate(source, target, ["population"], "ward_id", "new_ward")
    assert result["population"].sum() == pytest.approx(300.0)


def test_target_with_no_overlap_is_absent_from_result(source):
    target = _gdf([{"new_ward": "Z", "geometry": box(10, 10, 11, 11)}])
    result = areal_interpolate(source, target, ["population"], "ward_id", "new_ward")
    assert result.empty


def test_requires_crs_on_both_frames(source):
    target = gpd.GeoDataFrame(
        [{"new_ward": "X", "geometry": box(0, 0, 1, 1)}]
    )  # no crs set
    with pytest.raises(ValueError, match="CRS"):
        areal_interpolate(source, target, ["population"], "ward_id", "new_ward")


def test_requires_nonempty_value_cols(source):
    target = _gdf([{"new_ward": "X", "geometry": box(0, 0, 1, 1)}])
    with pytest.raises(ValueError, match="value_cols"):
        areal_interpolate(source, target, [], "ward_id", "new_ward")


def test_interpolation_error_zero_when_reference_matches_exactly():
    interpolated = pd.DataFrame({"new_ward": ["X", "Y"], "population": [150.0, 50.0]})
    reference = pd.DataFrame({"new_ward": ["X", "Y"], "official_pop": [150.0, 50.0]})

    metrics = interpolation_error(interpolated, reference, "population", "official_pop", "new_ward")

    assert metrics["n_wards_matched"] == 2
    assert metrics["mean_abs_pct_error"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_interpolation_error_reports_nonzero_error_for_mismatch():
    interpolated = pd.DataFrame({"new_ward": ["X", "Y"], "population": [160.0, 50.0]})
    reference = pd.DataFrame({"new_ward": ["X", "Y"], "official_pop": [150.0, 50.0]})

    metrics = interpolation_error(interpolated, reference, "population", "official_pop", "new_ward")

    assert metrics["max_abs_pct_error"] == pytest.approx(10.0 / 150.0)
    assert metrics["total_interpolated"] == pytest.approx(210.0)
    assert metrics["total_reference"] == pytest.approx(200.0)


def test_interpolation_error_raises_when_no_ids_overlap():
    interpolated = pd.DataFrame({"new_ward": ["X"], "population": [150.0]})
    reference = pd.DataFrame({"new_ward": ["Z"], "official_pop": [150.0]})
    with pytest.raises(ValueError, match="no overlapping"):
        interpolation_error(interpolated, reference, "population", "official_pop", "new_ward")


# --- areal_interpolate_weighted_average ----------------------------------------


@pytest.fixture
def source_rates() -> gpd.GeoDataFrame:
    # Two adjacent 1x1 unit squares at different electrification rates.
    return _gdf(
        [
            {"ward_id": "A", "electrified_pct": 80.0, "geometry": box(0, 0, 1, 1)},
            {"ward_id": "B", "electrified_pct": 40.0, "geometry": box(1, 0, 2, 1)},
        ]
    )


def test_weighted_average_is_area_weighted_not_summed(source_rates):
    target = _gdf(
        [
            # straddles A/B evenly -> simple average of 80 and 40
            {"new_ward": "X", "geometry": box(0.5, 0, 1.5, 1)},
        ]
    )
    result = areal_interpolate_weighted_average(
        source_rates, target, ["electrified_pct"], "ward_id", "new_ward"
    )
    assert result.set_index("new_ward").loc["X", "electrified_pct"] == pytest.approx(60.0)


def test_weighted_average_matches_source_when_target_wholly_inside_one_source(source_rates):
    target = _gdf([{"new_ward": "Y", "geometry": box(0, 0, 0.5, 1)}])  # fully inside A
    result = areal_interpolate_weighted_average(
        source_rates, target, ["electrified_pct"], "ward_id", "new_ward"
    )
    assert result.set_index("new_ward").loc["Y", "electrified_pct"] == pytest.approx(80.0)


def test_weighted_average_weights_by_overlap_area_not_equal_split(source_rates):
    # 3/4 of this target's area is over A (80%), 1/4 over B (40%):
    # weighted average = 0.75*80 + 0.25*40 = 70, not the simple mean (60).
    target = _gdf([{"new_ward": "Z", "geometry": box(0.25, 0, 1.25, 1)}])
    result = areal_interpolate_weighted_average(
        source_rates, target, ["electrified_pct"], "ward_id", "new_ward"
    )
    assert result.set_index("new_ward").loc["Z", "electrified_pct"] == pytest.approx(70.0)


def test_weighted_average_requires_crs(source_rates):
    target = gpd.GeoDataFrame([{"new_ward": "X", "geometry": box(0, 0, 1, 1)}])
    with pytest.raises(ValueError, match="CRS"):
        areal_interpolate_weighted_average(
            source_rates, target, ["electrified_pct"], "ward_id", "new_ward"
        )


def test_weighted_average_requires_nonempty_value_cols(source_rates):
    target = _gdf([{"new_ward": "X", "geometry": box(0, 0, 1, 1)}])
    with pytest.raises(ValueError, match="value_cols"):
        areal_interpolate_weighted_average(source_rates, target, [], "ward_id", "new_ward")
