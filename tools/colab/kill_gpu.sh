#!/usr/bin/env bash
# Detiene papermill/secuencias en curso y mata todo proceso que ocupe la GPU de la sesión.
#   tools/colab/kill_gpu.sh <SESSION>
set -euo pipefail
source "$(dirname "$0")/_cexec.sh"
S=$1
TMP=$(mktemp /tmp/jev_kill_XXXX.py)
cat > "$TMP" <<'EOF'
import subprocess, time, os, signal
subprocess.run("pkill -f 'seq_2' ; pkill -f papermill ; true", shell=True)
time.sleep(3)
out = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout
for pid in [int(x) for x in out.split() if x.strip().isdigit()]:
    try:
        os.kill(pid, signal.SIGKILL); print("killed", pid)
    except Exception as e:
        print("could not kill", pid, e)
time.sleep(5)
print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout)
EOF
cexec "$S" "$TMP" 60
rm -f "$TMP"
