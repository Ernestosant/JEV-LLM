#!/usr/bin/env bash
# Estado de una corrida en segundo plano: proceso vivo, progress.json, últimas líneas del log, GPU.
#   tools/colab/status.sh <SESSION> <COND> [N_LINES]
set -euo pipefail
source "$(dirname "$0")/_cexec.sh"
S=$1; COND=$2; N=${3:-25}
TMP=$(mktemp /tmp/jev_status_XXXX.py)
cat > "$TMP" <<EOF
import json, os, pathlib, subprocess
root = pathlib.Path("/content/jev_llm")
cur = root / "papermill_$COND.current"
if not cur.exists():
    print("no hay corrida lanzada para $COND"); raise SystemExit
c = json.loads(cur.read_text())
try:
    alive = open(f"/proc/{c['pid']}/stat").read().split(")")[-1].split()[0] not in ("Z", "X")
except OSError:
    alive = False
print(f"== $COND pid={c['pid']} alive={alive}")
pr = root / "results" / "progress_$COND.json"
if "$COND" == "LATENCY":
    prs = sorted((root / "results" / "LATENCY").glob("*/progress.json"))
    pr = prs[-1] if prs else pr
print("progress:", pr.read_text() if pr.exists() else "(aún sin progress)")
lines = pathlib.Path(c["log"]).read_text(errors="replace").splitlines()
print("\n".join(lines[-$N:]))
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
for z in sorted((root / "results").glob("*.zip")):
    print("ZIP::", z, z.stat().st_size)
EOF
cexec "$S" "$TMP" 120
rm -f "$TMP"
