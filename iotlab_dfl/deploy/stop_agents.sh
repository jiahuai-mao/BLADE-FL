#!/bin/sh
set -eu

PROFILE=${1:?usage: stop_agents.sh 28|56}
CONFIG=${HOME}/shared/iotlab_dfl/bundle_${PROFILE}/iotlab_config.json
python3 -c 'import json,sys; print("\n".join("node-a8-%d.grenoble.iot-lab.info" % n["node_id"] for n in json.load(open(sys.argv[1]))["nodes"]))' "${CONFIG}" |
while IFS= read -r hostname; do
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "root@${hostname}" 'pkill -f "[p]ython.*-m iotlab_dfl.agent" || true'
done
