#!/bin/sh
set -eu

PROFILE=${1:?usage: check_agents.sh 28|56}
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
  if status=$(ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "root@${hostname}" \
    "python -c '\''import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex((\"127.0.0.1\",29600)) != 0)'\'' && pgrep -af '\''[p]ython.*-m iotlab_dfl.agent.*bundle_${PROFILE}'\''" 2>&1); then
    echo "[ready] ${hostname}: ${status}"
  else
    echo "[failed] ${hostname}: ${status}" >&2
    exit 1
  fi
' sh <"${HOSTS}"
