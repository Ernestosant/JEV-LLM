#!/usr/bin/env bash
# cexec SESSION FILE TIMEOUT : colab exec with retries (the runtime proxy/websocket may time out
# while the VM is busy compiling/loading models). Prints only the kernel output.
cexec() {
  local S=$1 F=$2 T=${3:-120} n
  for n in 1 2 3 4 5; do
    if out=$(timeout $((T + 90)) colab exec -s "$S" --timeout "$T" -f "$F" 2>&1); then
      if ! grep -qa "Connection was lost\|Read timed out\|ReadTimeout\|Traceback (most recent call last) ──" <<<"$out"; then
        grep -av 'new version\|colab update\|silence the\|enable_update_check' <<<"$out"
        return 0
      fi
    fi
    if grep -qa "appears to be lost" <<<"$out"; then
      echo "[cexec] la sesión $S fue marcada como perdida por el CLI" >&2
      return 2
    fi
    echo "[cexec] intento $n falló (conexión con la VM); reintento en 20 s" >&2
    sleep 20
  done
  echo "$out" | tail -5 >&2
  return 1
}
