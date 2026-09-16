#!/bin/sh
set -eu

PROFILE=${1:?usage: run_agent.sh 28|56}
case "${PROFILE}" in
  28|56) ;;
  *) echo "invalid profile: ${PROFILE}" >&2; exit 2 ;;
esac

SHARED_ROOT=/home/root/shared
PACKAGE_ROOT=${SHARED_ROOT}/iotlab_dfl
export PYTHONPATH=${SHARED_ROOT}

exec python -m iotlab_dfl.agent \
  --bundle-dir "${PACKAGE_ROOT}/bundle_${PROFILE}" \
  --config "${PACKAGE_ROOT}/bundle_${PROFILE}/iotlab_config.json" \
  --host 0.0.0.0 \
  --port 29600
