#!/usr/bin/env bash
# launch.sh CONDITION SESSION A100 [papermill arguments]
set -euo pipefail
COND=${1:?condition}; SESSION=${2:?session}; GPU=${3:?explicit GPU}; shift 3
exec bash "$(dirname "$0")/launch_seq.sh" "$SESSION" "$GPU" "$COND" "$@"
