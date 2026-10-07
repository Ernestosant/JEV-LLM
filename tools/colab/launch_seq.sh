#!/usr/bin/env bash
# launch_seq.sh SESSION A100 "CONDITION ..." [papermill arguments]
# Also sourced by operator.sh: one wrapper/API safety boundary, existing lifecycle helpers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/colab/_cexec.sh"

prepare_colab() {
  local session=$1 out=$2
  [[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] && [ "$session" != all ] || return 1
  export COLAB_PYTHON=${COLAB_PYTHON:-/root/.local/share/uv/tools/google-colab-cli/bin/python}
  test -x "$COLAB_PYTHON" || { echo "Missing CLI venv Python: $COLAB_PYTHON" >&2; return 1; }
  mkdir -p "$out/operator_bin"
  out=$(cd "$out" && pwd)
  export COLAB_SESSION_CONFIG=${COLAB_SESSION_CONFIG:-"$out/sessions.json"}
  [[ "$COLAB_SESSION_CONFIG" = /* ]] || { echo "Config must be absolute" >&2; return 1; }
  [ "$COLAB_SESSION_CONFIG" != "$HOME/.config/colab-cli/sessions.json" ] || return 1
  mkdir -p "$(dirname "$COLAB_SESSION_CONFIG")"
  export COLAB_REAL_CLI=${COLAB_REAL_CLI:-"$(dirname "$COLAB_PYTHON")/colab"}
  test -x "$COLAB_REAL_CLI" || return 1
  # pull/status/_cexec invoke colab themselves; PATH isolation bounds EVERY invocation.
  cat > "$out/operator_bin/colab" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_SESSION_CONFIG:?}" "${COLAB_REAL_CLI:?}"
for arg in "$@"; do
  case "$arg" in --config|--config=*) echo 'Wrapper config cannot be overridden' >&2; exit 2;; esac
done
seconds=${COLAB_CALL_TIMEOUT:-120}
[[ "$seconds" =~ ^[1-9][0-9]*$ ]] && [ "$seconds" -le 120 ] || { echo 'CLI timeout must be 1-120 seconds' >&2; exit 2; }
exec timeout --kill-after=10 "$seconds" "$COLAB_REAL_CLI" --config "$COLAB_SESSION_CONFIG" "$@"
SH
  chmod 700 "$out/operator_bin/colab"
  export PATH="$out/operator_bin:$PATH"
  # Both spellings are accepted at the launch boundary; persist one set for the operator.
  [ -z "${CONFIG_FILE:-}" ] || export JEV_CONFIG_FILE=${JEV_CONFIG_FILE:-$CONFIG_FILE}
  [ -z "${DATA_SUBDIR:-}" ] || export JEV_DATA_SUBDIR=${JEV_DATA_SUBDIR:-$DATA_SUBDIR}
  [ -z "${SPLIT:-}" ] || export JEV_SPLIT=${JEV_SPLIT:-$SPLIT}
  [ -z "${RUN_TAG:-}" ] || export JEV_RUN_TAG=${JEV_RUN_TAG:-$RUN_TAG}
}

canonical_conditions() {
  local default_config=config/experiment_v2.json
  [ "${JEV_SERIES:-v1}" != v2-reload ] || default_config=config/experiment_v2_reload.json
  python3 - "$ROOT/${JEV_CONFIG_FILE:-${CONFIG_FILE:-$default_config}}" "$1" <<'PY'
import json, sys
aliases = json.load(open(sys.argv[1])).get('condition_aliases', {})
print(' '.join(aliases.get(c, c) for c in sys.argv[2].split()))
PY
}

colab_identity() {
  # No secrets printed; network errors are failures, not empty assignment lists.
  timeout --kill-after=5 60 "$COLAB_PYTHON" - "$COLAB_SESSION_CONFIG" "$1" "${2:-}" <<'PY'
import json, logging, os, sys
logging.disable(logging.CRITICAL)
from colab_cli.common import state
state.config_path = sys.argv[1]
state.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
client = state.client
request = client.session.request
def bounded(method, url, **kwargs):
    kwargs.setdefault("timeout", (5, 15))
    return request(method, url, **kwargs)
client.session.request = bounded
try:
    assignments = client.list_assignments()
    sessions = state.store.list()
    if set(sessions) - {sys.argv[2]}:
        raise ValueError("Isolated config contains another session")
    session = sessions.get(sys.argv[2])
    assignment = next((a for a in assignments if session and a.endpoint == session.endpoint), None)
    endpoint = sys.argv[3] or (session.endpoint if session else None)
    print(json.dumps({"assignment_count": len(assignments), "local_exists": session is not None,
                      "endpoint": endpoint, "present": any(a.endpoint == endpoint for a in assignments),
                      "accelerator": assignment.accelerator.value if assignment else None}))
except Exception as error:
    print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
    sys.exit(1)
PY
}

launch_sequence() {
  local S=$1 GPU=$2 CONDS=$3; shift 3
  local SERIES=${JEV_SERIES:-v1} NB_DIR="$ROOT/notebooks" DIST="$ROOT/dist"
  [[ "$SERIES" != v2 && "$SERIES" != v2-reload ]] || { NB_DIR="$ROOT/notebooks/$SERIES"; DIST="$ROOT/dist/$SERIES"; }
  local PREFIX=jev_llm
  [[ "$SERIES" != v2 && "$SERIES" != v2-reload ]] || PREFIX=jev_llm_v2
  local BUNDLE="$DIST/${PREFIX}_bundle.zip"
  [[ "$SERIES" != v2 && "$SERIES" != v2-reload ]] || CONDS=$(canonical_conditions "$CONDS")
  declare -A NB=([G_SINGLE]=01_G_SINGLE [G4_SINGLE]=01_G4_SINGLE [B13]=02_B13
    [B13_GREEDY]=03_B13_GREEDY [J64]=04_J64 [J64_G4]=04_J64_G4
    [JSTEP]=05_JSTEP [JSTEP_G4]=05_JSTEP_G4 [JFINAL]=06_JFINAL
    [JFINAL_G4]=06_JFINAL_G4 [LATENCY]=08_estudio_latencia [ANALYSIS]=07_analisis)
  if [ "$CONDS" = ANALYSIS ]; then
    [ "$GPU" = CPU ] || { echo "Analysis requires CPU, no --gpu" >&2; return 1; }
    BUNDLE="$DIST/${PREFIX}_analysis_bundle.zip"
    : "${JEV_FINALS_DIR:?collected final ZIP directory}" "${JEV_GOLD_FILE:?gold JSONL}"
    test -f "$JEV_GOLD_FILE"
  else
    [ "$GPU" = A100 ] || { echo "All GPU launches require explicit A100" >&2; return 1; }
  fi
  test -f "$BUNDLE" || { echo "Missing $BUNDLE" >&2; return 1; }
  local c nb LIST=() seen=" "
  for c in $CONDS; do
    nb=${NB[$c]:?unknown condition}
    [[ "$seen" != *" $c "* ]] || return 1; seen+="$c "
    test -f "$NB_DIR/$nb.ipynb" || { echo "Missing $NB_DIR/$nb.ipynb" >&2; return 1; }
    LIST+=("$c:$nb")
  done
  [ ${#LIST[@]} -gt 0 ]
  local DEFAULT_CONFIG=config/experiment.json DEFAULT_DATA=data DEFAULT_TAG=main
  [[ "$SERIES" != v2 && "$SERIES" != v2-reload ]] || { DEFAULT_CONFIG=config/experiment_v2.json; DEFAULT_DATA=data/v2; DEFAULT_TAG=$SERIES; }
  [ "$SERIES" != v2-reload ] || DEFAULT_CONFIG=config/experiment_v2_reload.json
  local CONFIG_FILE=${JEV_CONFIG_FILE:-${CONFIG_FILE:-$DEFAULT_CONFIG}}
  local DATA_SUBDIR=${JEV_DATA_SUBDIR:-${DATA_SUBDIR:-$DEFAULT_DATA}}
  local SPLIT=${JEV_SPLIT:-${SPLIT:-test}} RUN_TAG=${JEV_RUN_TAG:-${RUN_TAG:-$DEFAULT_TAG}}
  if [[ "$SERIES" = v2 || "$SERIES" = v2-reload ]]; then
    set -- -p CONFIG_FILE "$CONFIG_FILE" -p DATA_SUBDIR "$DATA_SUBDIR" -p SPLIT "$SPLIT" -p RUN_TAG "$RUN_TAG" "$@"
  elif [ -n "${JEV_RUN_TAG:-}" ] || [ -n "${JEV_SPLIT:-}" ]; then
    set -- -p SPLIT "$SPLIT" -p RUN_TAG "$RUN_TAG" "$@"
  fi
  [[ "$CONFIG_FILE" =~ ^[A-Za-z0-9_./-]+$ && "$CONFIG_FILE" != /* && "$CONFIG_FILE" != *..* ]] || return 1
  [[ "$DATA_SUBDIR" =~ ^[A-Za-z0-9_./-]+$ && "$DATA_SUBDIR" != /* && "$DATA_SUBDIR" != *..* ]] || return 1
  [[ "$SPLIT" =~ ^(dev|pilot|test)$ && "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
  test -f "$ROOT/$CONFIG_FILE"
  # Reject stale/tampered bundles and accidental gold before any allocation.
  python3 - "$BUNDLE" "$ROOT" "$CONDS" "$CONFIG_FILE" "$DATA_SUBDIR" <<'PY'
import hashlib, json, pathlib, sys, zipfile
bundle, root, conditions, config, data = sys.argv[1:]
root = pathlib.Path(root)
with zipfile.ZipFile(bundle) as z:
    if z.testzip() is not None:
        raise SystemExit('Bundle ZIP integrity failure')
    manifest = json.loads(z.read('BUNDLE_MANIFEST.json'))
    if config not in manifest['files']:
        raise SystemExit('Selected config is not in the frozen bundle')
    if conditions != 'ANALYSIS' and (manifest.get('contains_gold') or any('_gold.' in n for n in z.namelist())):
        raise SystemExit('Gold in inference bundle')
    for name, digest in manifest['files'].items():
        if hashlib.sha256(z.read(name)).hexdigest() != digest or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Bundle/file hash mismatch: ' + name)
PY
  if [ "$CONDS" = ANALYSIS ]; then
    test -f "$ROOT/$CONFIG_FILE"
    local finals=("$JEV_FINALS_DIR"/*_final.zip)
    test -f "${finals[0]}" || { echo "No collected finals" >&2; return 1; }
    python3 - "$ROOT" "$CONFIG_FILE" "$DATA_SUBDIR" "$SPLIT" "$RUN_TAG" "$JEV_FINALS_DIR" <<'PY'
import json, os, pathlib, sys, zipfile
root, config, data, split, tag, directory = sys.argv[1:]
root = pathlib.Path(root)
sys.path.insert(0, str(root / 'tools'))
from verify_run import verify_archive, jsonl
seen = set()
for path in pathlib.Path(directory).glob('*_final.zip'):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('/manifest.json')]
        if len(names) != 1:
            raise SystemExit('Missing unique collected manifest')
        condition = json.loads(z.read(names[0]))['condition']
    if condition in seen:
        raise SystemExit('Duplicate collected condition: ' + condition)
    seen.add(condition)
    source = root / data / (('dev' if condition == 'LATENCY' else split) + '_inputs.jsonl')
    count = int(os.environ.get('JEV_ANALYSIS_EXPECTED_COUNT', len(jsonl(source.read_text()))))
    verify_archive(path, condition, inputs=source, config=root / config, expected_count=count,
                   split='dev' if condition == 'LATENCY' else split, run_tag=tag,
                   latency_rows=int(os.environ.get('JEV_LATENCY_ROWS', 192)),
                   latency_dev_count=int(os.environ.get('JEV_LATENCY_DEV_COUNT', 20)))
PY
  fi
  if [ "${JEV_PREFLIGHT_ONLY:-0}" = 1 ]; then
    echo "Offline preflight passed: bundle=$BUNDLE config=$CONFIG_FILE data=$DATA_SUBDIR split=$SPLIT tag=$RUN_TAG"
    return 0
  fi
  prepare_colab "$S" "${JEV_OPERATOR_OUT:-$ROOT/results/lifecycle/$S}"
  local CREATED=0 identity exists count accel
  cleanup_launch_failure() {
    local rc=$?
    trap - ERR
    if [ "$CREATED" = 1 ]; then
      timeout --kill-after=10 180 bash "$ROOT/tools/colab/stop.sh" "$S" || echo "Release unconfirmed: $S" >&2
    fi
    exit "$rc"
  }
  trap cleanup_launch_failure ERR
  # Serialize all cooperating launchers, including independently isolated configs.
  exec 8>/tmp/jev-colab-allocation.lock
  flock -w 120 8
  identity=$(colab_identity "$S")
  read -r exists count accel < <(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d["local_exists"]), d["assignment_count"], d["accelerator"])' "$identity")
  if [ "$exists" = 1 ]; then
    echo "Existing session: use operator monitor with explicit ownership; no remote work modified" >&2
    return 1
  else
    [ "$count" -lt 2 ] || { echo "Two assignment cap reached; unknown sessions untouched" >&2; return 1; }
    # Mark the allocation attempt before new: an uncertain CLI return must still attempt release.
    CREATED=1
    if [ "$GPU" = CPU ]; then colab new -s "$S"; else colab new -s "$S" --gpu A100; fi
  fi
  flock -u 8; exec 8>&-
  timeout --kill-after=5 30 bash "$ROOT/tools/colab/session_guard.sh" start "$S" "$COLAB_SESSION_CONFIG" "$ROOT/results/lifecycle/$S"
  colab upload -s "$S" "$BUNDLE" "/content/$(basename "$BUNDLE")"
  for c in "${LIST[@]}"; do nb=${c#*:}; colab upload -s "$S" "$NB_DIR/$nb.ipynb" "/content/$nb.ipynb"; done
  if [ "$CONDS" = ANALYSIS ]; then
    # Stage beside the bundle; extraction precedes moving these into the notebook's paths.
    colab upload -s "$S" "$ROOT/$CONFIG_FILE" /content/jev_analysis_config.json
    colab upload -s "$S" "$JEV_GOLD_FILE" /content/jev_analysis_gold.jsonl
    for c in "${finals[@]}"; do colab upload -s "$S" "$c" "/content/$(basename "$c")"; done
  fi
  local TMP ARGS_PY LIST_PY
  ARGS_PY=$(python3 -c 'import sys; print(repr(sys.argv[1:]))' "$@")
  LIST_PY=$(python3 -c 'import sys; print(repr(sys.argv[1:]))' "${LIST[@]}")
  TMP=$(mktemp /tmp/jev_seq_XXXX.py)
  cat > "$TMP" <<PY
import json, os, pathlib, subprocess, sys, time, zipfile
root = pathlib.Path('/content/jev_llm'); root.mkdir(exist_ok=True)
items, args = $LIST_PY, $ARGS_PY
# A prior run, even completed, is not permission to regenerate quality results.
if (root / 'launch_reserved.json').exists() or (root / 'seq_current.json').exists() or list(root.glob('papermill_*.current')):
    raise RuntimeError('Existing work: monitor it; never restart or rerun')
(root / 'launch_reserved.json').write_text(json.dumps({'items': items, 'args': args}))
if '$SERIES' in ('v2', 'v2-reload'):
    alias = pathlib.Path('/content/jev_llm_v2')
    if alias.exists() and alias.resolve() != root.resolve():
        raise RuntimeError('Existing v2 workspace: refusing replacement')
    if not alias.exists():
        alias.symlink_to(root, target_is_directory=True)
# The v2 notebook uses its series-specific path; the alias keeps existing pull/status helpers.
with zipfile.ZipFile('/content/$(basename "$BUNDLE")') as z:
    z.extractall(root)
(root / 'notebooks').mkdir(exist_ok=True)
for item in items:
    nb = item.split(':')[1]
    os.replace('/content/' + nb + '.ipynb', root / 'notebooks' / (nb + '.ipynb'))
if items[0].startswith('ANALYSIS:'):
    incoming = root / 'incoming'; incoming.mkdir(exist_ok=True)
    for path in pathlib.Path('/content').glob('*_final.zip'):
        os.replace(path, incoming / path.name)
    config_path = root / '$CONFIG_FILE'; config_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace('/content/jev_analysis_config.json', config_path)
    gold_path = root / '$DATA_SUBDIR' / '${SPLIT}_gold.jsonl'
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace('/content/jev_analysis_gold.jsonl', gold_path)
try:
    import papermill
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'papermill'], check=True, timeout=180)
ts = time.strftime('%Y%m%dT%H%M%S')
driver = root / ('seq_' + ts + '.py')
driver.write_text('''import json, pathlib, subprocess, sys, time
root = pathlib.Path('/content/jev_llm')
items, args, ts = ''' + repr(items) + ', ' + repr(args) + ', ' + repr(ts) + '''
status = {}
for item in items:
    c, nb = item.split(':')
    log = root / f'papermill_{c}.{ts}.log'
    out = root / 'notebooks' / f'{nb}.out.{ts}.ipynb'
    cmd = [sys.executable, '-m', 'papermill', str(root / 'notebooks' / (nb + '.ipynb')), str(out),
           '--log-output', '--request-save-on-cell-execute', '--autosave-cell-every', '60', *args]
    t0 = time.time()
    with log.open('w') as stream:
        p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, cwd='/content')
        (root / f'papermill_{c}.current').write_text(json.dumps({'pid': p.pid, 'log': str(log), 'out': str(out), 'cmd': cmd}))
        rc = p.wait()
    status[c] = {'rc': rc, 'seconds': round(time.time() - t0, 1)}
    (root / f'seq_{ts}.status.json').write_text(json.dumps(status))
    if rc:
        break
    # Only this fresh, isolated VM is used; do not run another notebook after failure.
    if c != 'ANALYSIS':
        probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        for pid in probe.stdout.split():
            if pid.isdigit():
                subprocess.run(['kill', '-9', pid], timeout=10)
        time.sleep(10)
(root / f'seq_{ts}.status.json').write_text(json.dumps({**status, '_done': True}))
''')
with (root / ('seq_' + ts + '.log')).open('w') as stream:
    p = subprocess.Popen([sys.executable, str(driver)], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, cwd='/content')
(root / 'seq_current.json').write_text(json.dumps({'pid': p.pid, 'ts': ts, 'items': items}))
print('LAUNCHED sequence', p.pid)
PY
  # _cexec's retries cannot rerun: the remote current marker refuses a second submission.
  local launched
  launched=$(cexec "$S" "$TMP" 300)
  printf '%s\n' "$launched"
  grep -q '^LAUNCHED sequence ' <<<"$launched"
  rm -f "$TMP"
  trap - ERR
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  launch_sequence "$@"
fi
