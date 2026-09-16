#!/bin/sh
set -eu

echo "[frontend]"
python3 -c 'import sys, numpy; print("python", sys.version.split()[0], "numpy", numpy.__version__)'

echo "[A8 nodes]"
iotlab-ssh run-cmd \
  'python -c "import socket,sys,numpy; print(socket.gethostname(), sys.version.split()[0], numpy.__version__)"'
