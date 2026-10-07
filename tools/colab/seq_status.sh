#!/usr/bin/env bash
# Estado de la secuencia lanzada con launch_seq.sh.   tools/colab/seq_status.sh <SESSION>
set -euo pipefail
source "$(dirname "$0")/_cexec.sh"
S=$1
TMP=$(mktemp /tmp/jev_seqst_XXXX.py)
cat > "$TMP" <<'EOF'
import json, os, pathlib, subprocess
root = pathlib.Path("/content/jev_llm")
cur = root / "seq_current.json"
if not cur.exists():
    print("sin secuencia"); raise SystemExit
c = json.loads(cur.read_text())
def _alive(pid):
    try:
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] not in ("Z", "X")
    except OSError:
        return False
print("sequence pid", c["pid"], "alive", _alive(c["pid"]), "items", c["items"])
st = root / f"seq_{c['ts']}.status.json"
print(st.read_text() if st.exists() else "(ninguna condición terminada aún)")
for p in sorted(root.glob("results/progress_*.json")):
    print(p.name, p.read_text())
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
EOF
cexec "$S" "$TMP" 120
rm -f "$TMP"
