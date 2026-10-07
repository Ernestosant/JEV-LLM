#!/usr/bin/env bash
# Libera la VM y comprueba su ausencia en el servidor, no solo el registro local.
# Descarga antes con pull.sh: /content se borra. El guard deja de enviar pings al cancelar.
#   tools/colab/stop.sh <SESSION|all>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG=${COLAB_SESSION_CONFIG:-"$HOME/.config/colab-cli/sessions.json"}
if [ "$1" = "all" ]; then
  for s in $(colab sessions | sed -n 's/^\[\(jev-[A-Za-z0-9_-]*\)\].*/\1/p'); do
    bash "$0" "$s"
  done
else
  bash "$ROOT/tools/colab/session_guard.sh" release "$1" "$CONFIG" "$ROOT/results/lifecycle/$1"
fi
