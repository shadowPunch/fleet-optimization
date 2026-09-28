#!/usr/bin/env bash
# Download a finished study kernel's outputs into results/ and sync its
# offline W&B run. Usage: kaggle/pull.sh <regime>
set -euo pipefail
cd "$(dirname "$0")/.."
USER=$(kaggle config view | awk '/username/ {print $3}')
OUT="results/kaggle/$1"
mkdir -p "$OUT"
kaggle kernels output "$USER/fleet-dispatch-study-$1" -p "$OUT"
cp "$OUT"/results/nyc_study_"$1".* results/
for run in "$OUT"/wandb/offline-run-*; do uv run wandb sync "$run"; done
