from __future__ import annotations

import json
from datetime import time

import polars as pl

from dispatch_eval.studies.dashboard import build_dashboard


def test_dashboard_embeds_results_as_json(tmp_path):
    eda = tmp_path / "eda"
    eda.mkdir()
    pl.DataFrame({"slot": [time(12, 0)], "mean_requests": [10.0], "min_requests": [8],
                  "max_requests": [12]}).write_parquet(eda / "demand_by_slot.parquet")
    pl.DataFrame({"hour": [12], "trips": [5], "mean_wait_s": [190.0], "mean_approach_s": [130.0],
                  "mean_boarding_s": [60.0]}).write_parquet(eda / "wait_components_by_hour.parquet")
    (eda / "summary.json").write_text(json.dumps({"trips": 5}))
    (tmp_path / "nyc_validation.json").write_text(json.dumps({
        "wait_ks_distance": 0.03, "hour_of_day_cosine": 0.98, "quantiles": [],
        "calibration_grid": [],
    }))
    (tmp_path / "nyc_study_tight.json").write_text(json.dumps({"fleet_size": 4000, "note": "</script>"}))

    html = build_dashboard(tmp_path, tmp_path / "out.html").read_text()

    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert data["validation"]["wait_ks_distance"] == 0.03
    assert data["eda"]["demand"][0]["slot"] == "12:00"
    assert list(data["studies"]) == ["tight"]
    assert "</script>\"" not in payload  # embedded strings can't close the script tag
