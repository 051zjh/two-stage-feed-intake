#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/mnt/B/charles/Software/conda_envs/imf_yip/bin/python}"
"$PYTHON" src/reproduce_all.py
