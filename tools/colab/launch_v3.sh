#!/usr/bin/env bash
# CONDITION SESSION A100|CPU OUTPUT [run_v3.py authorization/evidence options]
set -euo pipefail
CONDITION=${1:?condition}; SESSION=${2:?session}; GPU=${3:?A100 or CPU}; OUT=${4:?output}; shift 4
if [ "$CONDITION" = ANALYSIS ]; then
  [ "$GPU" = CPU ] || { echo 'Analysis requires CPU' >&2; exit 2; }
else
  [ "$GPU" = A100 ] || { echo 'v3 requires explicit A100' >&2; exit 2; }
fi
PY=/root/.local/share/uv/tools/google-colab-cli/bin/python
exec "$PY" -B "$(dirname "$0")/run_v3.py" --operator-start --session "$SESSION" \
  --condition "$CONDITION" --output "$OUT" "$@"
