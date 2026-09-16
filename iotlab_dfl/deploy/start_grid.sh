#!/bin/sh
set -eu

PROFILE=${1:?usage: start_grid.sh 28|56}
case "${PROFILE}" in
  28|56) ;;
  *) echo "invalid profile: ${PROFILE}" >&2; exit 2 ;;
esac
PACKAGE_ROOT=${HOME}/shared/iotlab_dfl
RESULTS_ROOT=${PACKAGE_ROOT}/results_${PROFILE}_r300
mkdir -p "${RESULTS_ROOT}"
if pgrep -f "[s]h .*deploy/run_all.sh ${PROFILE}" >/dev/null; then
  echo "profile ${PROFILE} grid is already running" >&2
  exit 1
fi
nohup sh "${PACKAGE_ROOT}/deploy/run_all.sh" "${PROFILE}" \
  >"${RESULTS_ROOT}/full_grid.log" 2>&1 </dev/null &
echo $! >"${RESULTS_ROOT}/full_grid.pid"
echo "[started] profile=${PROFILE} pid=$! log=${RESULTS_ROOT}/full_grid.log"
