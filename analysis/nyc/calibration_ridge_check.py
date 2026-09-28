"""Is fleet size identified by wait times above ~6,500 vehicles?

The V2 supply calibration's best point sat on the grid's upper edge
(10,000). This extends the grid upward on the *calibration days only* —
the held-out days are not touched — to show whether the KS surface keeps
falling (the grid was too narrow) or is flat (fleet size is not identified
from wait data once supply is plentiful).

Usage: uv run python analysis/nyc/calibration_ridge_check.py
"""

import json
from pathlib import Path

import yaml

from dispatch_eval.calibration.pickup import calibrate_supply, same_zone_log_sigma
from dispatch_eval.studies.nyc_data import wait_seconds
from dispatch_eval.studies.validation import build_twin
from dispatch_eval.tracking import tracked_run

cfg = yaml.safe_load(Path("configs/nyc.yaml").read_text())
twin = build_twin(cfg)
sigma = same_zone_log_sigma(twin.data.calibration)
fleets, medians = [12500, 16000], [90, 120, 150]

with tracked_run("calibration-ridge-check", "calibration", {"fleets": fleets, "medians": medians},
                 tags=["nyc", "calibration"]) as run:
    result = calibrate_supply(
        wait_seconds(twin.data.calibration),
        lambda fleet, params: twin.simulate(fleet, params, cfg["validation"]["seed"]).wait_times,
        fleets, medians, sigma, on_point=print,
    )
    run.log_table("ridge_grid", result.grid)

out = Path("results/nyc_calibration_ridge.json")
out.write_text(json.dumps(result.grid, indent=2))
print(f"wrote {out}")
