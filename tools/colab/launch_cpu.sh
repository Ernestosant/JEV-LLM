#!/usr/bin/env bash
# launch_cpu.sh SESSION COLLECTED_FINALS CONFIG_RELATIVE GOLD_JSONL [papermill args]
set -euo pipefail
S=${1:?session}; export JEV_FINALS_DIR=${2:?final directory}
export JEV_CONFIG_FILE=${3:?config path relative to bundle root}
export JEV_GOLD_FILE=${4:?gold JSONL}; shift 4
exec bash "$(dirname "$0")/launch_seq.sh" "$S" CPU ANALYSIS "$@"
