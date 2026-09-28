#!/usr/bin/env bash
# Build the inputs dataset and push one study kernel per regime to Kaggle.
# Usage: kaggle/push.sh [n_bootstrap]   (default: the config's n_bootstrap)
# Env: REGIMES="tight mid" to push only some regimes; SKIP_DATASET=1 to reuse
# the already-uploaded inputs dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

USER=$(kaggle config view | awk '/username/ {print $3}')
DATASET="dispatch-eval-inputs"
N_BOOTSTRAP="${1:-}"
BUILD=$(mktemp -d)

# 1. Inputs dataset: package wheel, config, cached NYC window, validation result.
if [[ -z "${SKIP_DATASET:-}" ]]; then
rm -rf dist && uv build --wheel -q
mkdir -p "$BUILD/dataset"
cp dist/*.whl configs/nyc.yaml results/nyc_validation.json data/cache/nyc_taxi_zone_lookup.csv \
   data/cache/nyc_*.parquet "$BUILD/dataset/"
cat > "$BUILD/dataset/dataset-metadata.json" <<JSON
{"title": "dispatch-eval inputs", "id": "$USER/$DATASET", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status "$USER/$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$BUILD/dataset" -m "update $(git rev-parse --short HEAD)" -q
else
  kaggle datasets create -p "$BUILD/dataset" -q
fi
echo "waiting for dataset to be ready..."
until kaggle datasets status "$USER/$DATASET" 2>/dev/null | grep -q ready; do sleep 10; done
fi

# 2. One kernel per regime.
ALL_REGIMES=$(uv run python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/nyc.yaml'))['study']['regimes']))")
for REGIME in ${REGIMES:-$ALL_REGIMES}; do
  KDIR="$BUILD/kernel-$REGIME"
  mkdir -p "$KDIR"
  sed -e "s/__REGIME__/$REGIME/" -e "s/__N_BOOTSTRAP__/$N_BOOTSTRAP/" kaggle/kernel.py > "$KDIR/kernel.py"
  cat > "$KDIR/kernel-metadata.json" <<JSON
{
  "id": "$USER/fleet-dispatch-study-$REGIME",
  "title": "fleet dispatch study $REGIME",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["$USER/$DATASET"]
}
JSON
  kaggle kernels push -p "$KDIR"
done
echo "pushed; check with: kaggle kernels status $USER/fleet-dispatch-study-<regime>"
