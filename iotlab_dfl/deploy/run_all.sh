#!/bin/sh
set -eu

PROFILE=${1:?usage: run_all.sh 28|56}
case "${PROFILE}" in
  28|56) ;;
  *) echo "invalid profile: ${PROFILE}" >&2; exit 2 ;;
esac
PACKAGE_ROOT=${HOME}/shared/iotlab_dfl
BUNDLE_ROOT=${PACKAGE_ROOT}/bundle_${PROFILE}
RESULTS_ROOT=${PACKAGE_ROOT}/results_${PROFILE}_r300
NODE_RESULTS=/home/root/shared/iotlab_dfl/results_${PROFILE}_r300
export PYTHONPATH=${HOME}/shared

python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["rounds"] == 300; assert len(c["nodes"]) == int(sys.argv[2])' "${BUNDLE_ROOT}/iotlab_config.json" "${PROFILE}"
mkdir -p "${RESULTS_ROOT}"
for seed in 0 3 7; do
  for algorithm in bladefl sparse_dpsgd adpsgd mdfeel; do
    run_id=${algorithm}_${PROFILE}clients_seed${seed}_r300
    if [ -f "${RESULTS_ROOT}/${run_id}/summary.json" ]; then
      echo "[skip] completed ${run_id}"
      continue
    fi
    if [ -e "${RESULTS_ROOT}/${run_id}" ]; then
      echo "[error] incomplete run exists: ${RESULTS_ROOT}/${run_id}" >&2
      exit 1
    fi
    python3 -m iotlab_dfl.controller \
      --algorithm "${algorithm}" \
      --seed "${seed}" \
      --rounds 300 \
      --bundle-dir "${BUNDLE_ROOT}" \
      --config "${BUNDLE_ROOT}/iotlab_config.json" \
      --output-dir "${RESULTS_ROOT}" \
      --node-output-dir "${NODE_RESULTS}" \
      --run-id "${run_id}"
  done
done
