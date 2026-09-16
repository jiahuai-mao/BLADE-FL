#!/bin/sh
set -eu

PROFILE=${1:?usage: status.sh 28|56}
PACKAGE_ROOT=${HOME}/shared/iotlab_dfl
RESULTS_ROOT=${PACKAGE_ROOT}/results_${PROFILE}_r300
printf 'profile=%s completed=' "${PROFILE}"
find "${RESULTS_ROOT}" -name summary.json -type f 2>/dev/null | wc -l
pgrep -af "[s]h .*deploy/run_all.sh ${PROFILE}|[p]ython3 -m iotlab_dfl.controller" || true
tail -n 20 "${RESULTS_ROOT}/full_grid.log" 2>/dev/null || true
