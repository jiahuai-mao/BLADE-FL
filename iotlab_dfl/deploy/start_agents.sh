#!/bin/sh
set -eu

PROFILE=${1:?usage: start_agents.sh 28|56}
case "${PROFILE}" in
  28|56) ;;
  *) echo "invalid profile: ${PROFILE}" >&2; exit 2 ;;
esac
PACKAGE_ROOT=${HOME}/shared/iotlab_dfl
CONFIG=${PACKAGE_ROOT}/bundle_${PROFILE}/iotlab_config.json

HOSTS=$(mktemp)
trap 'rm -f "${HOSTS}"' EXIT HUP INT TERM
python3 -c 'import json,sys; print("\n".join("node-a8-%d.grenoble.iot-lab.info" % n["node_id"] for n in json.load(open(sys.argv[1]))["nodes"]))' "${CONFIG}" >"${HOSTS}"
export PROFILE
xargs -r -n 1 -P 8 sh -c '
  hostname=$1
  echo "[start] ${hostname} profile=${PROFILE}"
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "root@${hostname}" \
    "pkill -f '\''[p]ython.*-m iotlab_dfl.agent'\'' || true; sleep 1; setsid -f sh -c '\''exec sh /home/root/shared/iotlab_dfl/deploy/run_agent.sh ${PROFILE} >/tmp/iotlab_dfl_agent.log 2>&1 </dev/null'\''"
' sh <"${HOSTS}"

sleep 3
sh "$(dirname "$0")/check_agents.sh" "${PROFILE}"
