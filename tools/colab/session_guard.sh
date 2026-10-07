#!/usr/bin/env bash
# start|release SESSION CONFIG_PATH OUTPUT_DIRECTORY (WSL; CLI Python environment).
set -euo pipefail
ACTION=$1; SESSION=$2; CONFIG=$3; OUT=$4
PY=${COLAB_PYTHON:-"$HOME/.local/share/uv/tools/google-colab-cli/bin/python"}
SCRIPT="$(dirname "$0")/session_guard.py"
test -x "$PY" || { echo "Set COLAB_PYTHON to the CLI environment's Python" >&2; exit 1; }
mkdir -p "$OUT"
if [ "$ACTION" = start ]; then
  nohup "$PY" "$SCRIPT" watch "$SESSION" --config "$CONFIG" --output "$OUT" --interval 20 \
    >> "$OUT/session_guard.log" 2>&1 </dev/null &
  PID=$!
  sleep 2
  kill -0 "$PID" 2>/dev/null || { echo "Guard failed at startup; inspect $OUT/session_guard.log" >&2; exit 1; }
  echo "Guard requested for $SESSION; pid=$PID; evidence=$OUT/session_lifecycle.jsonl"
elif [ "$ACTION" = release ]; then
  "$PY" "$SCRIPT" release "$SESSION" --config "$CONFIG" --output "$OUT"
else
  echo "Expected start or release" >&2; exit 1
fi
