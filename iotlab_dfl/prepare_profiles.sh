#!/bin/sh
set -eu

RAW_CACHE=${1:-data/HHAR/.cache/hhar_efa9d1d8a336cd1d.npz}

for profile in 28 56; do
  case "${profile}" in
    28) minimum=1500 ;;
    56) minimum=750 ;;
  esac
  python -m iotlab_dfl.prepare_hhar \
    --output-dir "iotlab_dfl/bundle_${profile}" \
    --num-clients "${profile}" \
    --split-seed 0 \
    --partition-seed 2026 \
    --model-seed 0 \
    --feature-set rich48 \
    --raw-cache "${RAW_CACHE}" \
    --hidden-dims 64,32 \
    --partition block-dirichlet \
    --dirichlet-alpha 0.3 \
    --block-size 256 \
    --min-samples "${minimum}" \
    --min-labels 2 \
    --min-label-samples 64 \
    --max-sample-factor 2.5
done
