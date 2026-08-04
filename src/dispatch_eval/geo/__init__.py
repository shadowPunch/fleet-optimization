"""Geospatial data-preparation tools (dev-time only, not simulator runtime).

Nothing in the simulator or calibration path needs a GIS library. This
sub-package exists for preparing real-world reference data — currently, C5's
ward-boundary crosswalk (see `crosswalk.py`) — before it enters the rest of
the package as plain tables. `geopandas`/`shapely`/`pyproj` are therefore
dev dependencies only (see pyproject.toml), not core runtime dependencies.
"""
