#!/usr/bin/env bash
# Descarga los zips de resultados (checkpoint/final) y logs de papermill de una sesión.
#   tools/colab/pull.sh <SESSION> [DEST]      (DEST por defecto: results/colab/<SESSION>)
set -euo pipefail
source "$(dirname "$0")/_cexec.sh"
S=$1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST=${2:-"$ROOT/results/colab/$S"}
mkdir -p "$DEST"
TMP=$(mktemp /tmp/jev_pull_XXXX.py)
cat > "$TMP" <<'EOF'
import pathlib
root = pathlib.Path("/content/jev_llm")
for z in sorted((root / "results").glob("*.zip")) + sorted(root.glob("papermill_*.log")) \
        + sorted((root / "notebooks").glob("*.out.*.ipynb")):
    print("FILE::" + str(z))
EOF
mapfile -t FILES < <(cexec "$S" "$TMP" 120 | grep -a '^FILE::' | sed 's/^FILE:://' | tr -d '\r')
rm -f "$TMP"
for f in "${FILES[@]}"; do
  echo "[pull] $f"
  for n in 1 2 3; do
    colab download -s "$S" "$f" "$DEST/$(basename "$f")" >/dev/null 2>&1 && break
    echo "[pull] reintento $n: $f"; sleep 15
  done
done
echo "[pull] ${#FILES[@]} archivos en $DEST"
