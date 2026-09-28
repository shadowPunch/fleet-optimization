"""Kaggle entry point: run one regime of the NYC policy study.

Pushed as a script kernel by `kaggle/push.sh`. Inputs come from the private
dataset built by the same script (package wheel, config, cached NYC data,
validation result). W&B runs offline here; `kaggle/pull.sh` downloads the
outputs and syncs the runs from a machine that has W&B credentials.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REGIME = "__REGIME__"  # substituted by push.sh
N_BOOTSTRAP = "__N_BOOTSTRAP__"
INPUT = Path(glob.glob("/kaggle/input/*dispatch-eval-inputs*")[0])
WORK = Path("/kaggle/working")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


wheel = glob.glob(str(INPUT / "*.whl"))[0]
run([sys.executable, "-m", "pip", "install", "-q", wheel])

(WORK / "data" / "cache").mkdir(parents=True, exist_ok=True)
for f in INPUT.glob("*.parquet"):
    shutil.copy(f, WORK / "data" / "cache" / f.name)
shutil.copy(INPUT / "nyc_taxi_zone_lookup.csv", WORK / "data" / "cache")

os.chdir(WORK)
os.environ["WANDB_MODE"] = "offline"
workers = os.cpu_count() or 4
print(f"cpu_count={workers}", flush=True)
cmd = [
    "dispatch-eval", "study",
    "--config", str(INPUT / "nyc.yaml"),
    "--validation", str(INPUT / "nyc_validation.json"),
    "--regime", REGIME,
    "--output-dir", str(WORK / "results"),
    "--workers", str(workers),
]
if N_BOOTSTRAP:
    cmd += ["--n-bootstrap", N_BOOTSTRAP]
run(cmd)
