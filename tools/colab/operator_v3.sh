#!/usr/bin/env bash
# prepare is offline. start/status SESSION CONDITION OUTPUT [run_v3.py options]
set -euo pipefail
ACTION=${1:?prepare, start or status}; shift
SCRIPT="$(dirname "$0")/run_v3.py"
PY=/root/.local/share/uv/tools/google-colab-cli/bin/python
case "$ACTION" in
  prepare) exec "$PY" -B "$SCRIPT" --plan "$@" ;;
  start|status)
    SESSION=${1:?session}; CONDITION=${2:?condition}; OUT=${3:?output}; shift 3
    exec "$PY" -B "$SCRIPT" "--operator-$ACTION" --session "$SESSION" \
      --condition "$CONDITION" --output "$OUT" "$@" ;;
  *) echo 'Expected prepare, start or status' >&2; exit 2 ;;
esac
