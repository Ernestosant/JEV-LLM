#!/usr/bin/env bash
# prepare SESSION OUT: create isolated wrapper/config environment without contacting Colab.
# start launch|monitor|cpu SESSION "CONDITIONS" OUT EXPECTED_COUNT INPUTS [papermill args]
# monitor requires COLAB_SESSION_CONFIG + JEV_OWN_SESSION=SESSION (explicit ownership).
set -euo pipefail
source "$(dirname "$0")/launch_seq.sh"
ACTION=${1:?prepare or start}; shift
if [ "$ACTION" = prepare ]; then
  prepare_colab "${1:?session}" "${2:?output directory}"
  printf 'export COLAB_SESSION_CONFIG=%q\nexport COLAB_PYTHON=%q\nexport COLAB_REAL_CLI=%q\nexport PATH=%q\n' \
    "$COLAB_SESSION_CONFIG" "$COLAB_PYTHON" "$COLAB_REAL_CLI" "$PATH"
  exit 0
fi
[ "$ACTION" = start ] || [ "$ACTION" = _worker ] || { echo "Expected prepare or start" >&2; exit 2; }
MODE=${1:?launch/monitor/cpu}; S=${2:?session}; CONDS=${3:?conditions}
OUT=${4:?output}; EXPECTED=${5:?count}; INPUTS=${6:?inputs}; shift 6
[[ "$MODE" = launch || "$MODE" = monitor || "$MODE" = cpu ]] || exit 2
[[ "$EXPECTED" =~ ^[1-9][0-9]*$ ]] || exit 2
[ "$MODE" != monitor ] || [ "${JEV_OWN_SESSION:-}" = "$S" ] || { echo "Monitor requires explicit JEV_OWN_SESSION=$S" >&2; exit 2; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
INPUTS=$(realpath "$INPUTS")
test -f "$INPUTS"
export JEV_OPERATOR_OUT="$OUT"
prepare_colab "$S" "$OUT"
[[ "${JEV_SERIES:-v1}" != v2 && "${JEV_SERIES:-v1}" != v2-reload ]] || CONDS=$(canonical_conditions "$CONDS")
if [ "$ACTION" = start ]; then
  # One durable operator per session, not just per output folder.
  exec 9>"/tmp/jev-operator-$S.lock"
  flock -n 9 || { echo "An operator already owns $S; do not relaunch" >&2; exit 1; }
  python3 - "$OUT" "$S" "$MODE" "$CONDS" <<'PY'
import json, os, pathlib, sys, time
out = pathlib.Path(sys.argv[1])
record = json.loads((out / 'execution_handover.json').read_text()) if (out / 'execution_handover.json').exists() else {}
if record and (record['session'] != sys.argv[2] or record['conditions'] != sys.argv[4].split()):
    raise SystemExit('Output directory belongs to different work')
duration = float(os.environ.get('JEV_DEADLINE_SECONDS', 14400))
if not 0 < duration <= 14400:
    raise SystemExit('Deadline must be positive and at most four hours')
record.update({"status": "starting", "session": sys.argv[2], "mode": sys.argv[3],
          "conditions": sys.argv[4].split(), "verified": False, "released": False,
          "cli_config": os.environ["COLAB_SESSION_CONFIG"], "start_epoch": record.get('start_epoch', time.time()),
          "cli_wrapper": str(out / "operator_bin/colab"), "output": str(out)})
record['deadline_epoch'] = min(record.get('deadline_epoch', float('inf')), record['start_epoch'] + duration)
for name in ('execution_handover.json', 'status.json'):
    (out / name).write_text(json.dumps(record, indent=2) + '\n')
PY
  # The worker inherits fd 9, so the lock remains held across parent exit.
  nohup setsid bash "$ROOT/tools/colab/operator.sh" _worker "$MODE" "$S" "$CONDS" "$OUT" "$EXPECTED" "$INPUTS" "$@" \
    >> "$OUT/operator.log" 2>&1 </dev/null &
  worker=$!
  # Keep the WSL client alive until the detached worker has actually initialized.
  python3 - "$OUT/status.json" "$worker" <<'PY'
import json, pathlib, sys, time
status = pathlib.Path(sys.argv[1])
for _ in range(100):
    if json.loads(status.read_text()).get('operator_pid'):
        break
    if not pathlib.Path('/proc/' + sys.argv[2]).exists():
        raise SystemExit('Detached operator exited during startup; inspect operator.log')
    time.sleep(0.1)
else:
    raise SystemExit('Detached operator did not acknowledge startup; inspect operator.log')
PY
  echo "Operator pid=$worker; handover=$OUT/execution_handover.json; status=$OUT/status.json"
  exit 0
fi

export JEV_ROOT="$ROOT"
exec python3 - "$MODE" "$S" "$CONDS" "$OUT" "$EXPECTED" "$INPUTS" "$@" <<'PY'
import datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time, threading, zipfile
from concurrent.futures import ThreadPoolExecutor

mode, session, cond_text, output, count, inputs, *pm_args = sys.argv[1:]
root, out = pathlib.Path(os.environ['JEV_ROOT']), pathlib.Path(output)
conditions = cond_text.split()
count = int(count)
state = json.loads((out / 'execution_handover.json').read_text())
start = state['start_epoch']
deadline = state['deadline_epoch']
work_deadline = deadline - min(360, (deadline - start) / 4)
art = out / 'artifacts'; art.mkdir(exist_ok=True)
mirror = out / 'checkpoints'; mirror.mkdir(exist_ok=True)
staging = out / 'checkpoint_staging'; staging.mkdir(exist_ok=True)
guard = root / 'results/lifecycle' / session
owner = None
finished = False
verified = False
released = False
before = {}
series = os.environ.get('JEV_SERIES', 'v1')
v2_series = series in ('v2', 'v2-reload')
state.update(operator_pid=os.getpid(), deadline_epoch=deadline, work_deadline_epoch=work_deadline, inputs=inputs, expected_count=count,
             checkpoints=str(mirror), artifacts=str(art),
             series=series,
             config_file=os.environ.get('JEV_CONFIG_FILE', 'config/experiment_v2_reload.json' if series == 'v2-reload'
                                        else 'config/experiment_v2.json' if v2_series else 'config/experiment.json'),
             data_subdir=os.environ.get('JEV_DATA_SUBDIR', 'data/v2' if v2_series else 'data'),
             split='dev' if conditions == ['LATENCY'] else os.environ.get('JEV_SPLIT', 'test'),
             run_tag=os.environ.get('JEV_RUN_TAG', series if v2_series else 'main'),
             papermill_args=pm_args, guard_directory=str(guard),
             colab_python=os.environ.get('COLAB_PYTHON', '/root/.local/share/uv/tools/google-colab-cli/bin/python'),
             poll_seconds=25, controls_minutes=[2, 5, 10, 30], guard_heartbeat_max_age_seconds=65,
             guard_recovery_window_seconds=180,
             remaining_obligations='Monitor, preserve checkpoints, verify final artifacts, release only recorded endpoint; no retries/restarts.')
pool = ThreadPoolExecutor(max_workers=2)
jobs = []
children = set()
children_lock = threading.Lock()

def write(status, **fields):
    state.update(status=status, verified=verified, released=released, **fields)
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for name in ('execution_handover.json', 'status.json'):
        target = out / name
        tmp = target.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(target)
    print(state['updated_utc'], status, flush=True)

def call(command, timeout, name, *, cleanup=False):
    remaining = work_deadline - time.time()
    if not cleanup and remaining <= 0:
        raise TimeoutError('Four-hour deadline expired')
    budget = timeout if cleanup else min(timeout, max(1, remaining))
    # Separate groups allow killing a stuck local CLI without touching the independent guard.
    with (out / (name + '.log')).open('a') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with children_lock:
            children.add(p.pid)
        try:
            rc = p.wait(timeout=budget)
        except BaseException:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
        finally:
            with children_lock:
                children.discard(p.pid)
    return rc

def finish_background():
    for job in jobs:
        job.cancel()
    with children_lock:
        for pid in children:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    pool.shutdown(wait=True, cancel_futures=True)

def mirror_checkpoints():
    # Never overwrite the last intact local checkpoint with a timed-out/partial download.
    call(['bash', str(root / 'tools/colab/pull.sh'), session, str(staging)], 180, 'checkpoint_pull')
    for path in staging.glob('*.zip'):
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = mirror / path.name
            path.replace(target)
            with (out / 'checkpoint_index.jsonl').open('a') as stream:
                stream.write(json.dumps({'path': str(target), 'sha256': digest, 'epoch': time.time()}) + '\n')
        except (OSError, zipfile.BadZipFile):
            continue

def identity(endpoint='', cleanup=False):
    command = ['bash', '-c', 'source "$1"; colab_identity "$2" "$3"', 'bash',
               str(root / 'tools/colab/launch_seq.sh'), session, endpoint]
    rc = call(command, 65, 'identity', cleanup=cleanup)
    if rc:
        raise RuntimeError('Backend identity uncertain; see identity.log')
    lines = (out / 'identity.log').read_text().splitlines()
    return json.loads(lines[-1])

def interrupted(signum, frame):
    raise InterruptedError(f'Operator interrupted by signal {signum}')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

# Read-only remote observation. Execution transport is the existing _cexec helper.
probe = out / 'health_remote.py'
probe.write_text('''import json, pathlib
root = pathlib.Path('/content/jev_llm')
def alive(pid):
    try:
        return pathlib.Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0] not in ('Z', 'X')
    except OSError:
        return False
seq = root / 'seq_current.json'
result = {'known': False, 'alive': None, 'conditions': {}}
if seq.exists():
    cur = json.loads(seq.read_text())
    status_path = root / f"seq_{cur['ts']}.status.json"
    result.update(known=True, alive=alive(cur['pid']), sequence_status=json.loads(status_path.read_text()) if status_path.exists() else {})
for c in ''' + repr(conditions) + ''':
    path = root / f'papermill_{c}.current'
    if path.exists():
        cur = json.loads(path.read_text())
        result['conditions'][c] = {'alive': alive(cur['pid']), 'log': cur['log'], 'out': cur['out']}
if not result['known'] and result['conditions']:
    result.update(known=True, alive=any(c['alive'] for c in result['conditions'].values()))
print('HEALTH_JSON::' + json.dumps(result))
''')
def health():
    log = out / 'health.log'
    offset = log.stat().st_size if log.exists() else 0
    budget = 20 if guard_recovery_started is None else max(1, min(20, guard_recovery_started + 180 - time.time()))
    try:
        rc = call(['bash', '-c', 'source "$1"; cexec "$2" "$3" 10', 'bash',
                   str(root / 'tools/colab/_cexec.sh'), session, str(probe)], budget, 'health')
    except subprocess.TimeoutExpired:
        return None  # Transport uncertainty is not proof of a stopped or healthy notebook.
    if rc:
        return None
    with log.open() as stream:
        stream.seek(offset)
        lines = stream.read().splitlines()
    return next((json.loads(line.split('::', 1)[1]) for line in reversed(lines) if line.startswith('HEALTH_JSON::')), None)

def guard_healthy():
    try:
        if not guard_running():
            return False
        recorded = json.loads((guard / 'session_guard_identity.json').read_text())
        if recorded != {'session': session, 'endpoint': owner}:
            return False
        records = [json.loads(line) for line in (guard / 'session_lifecycle.jsonl').read_text().splitlines() if line]
        # A transient health_unknown cannot erase recent positive evidence. A restart or
        # confirmed absence does erase it: never credit an earlier guard's connection.
        for record in reversed(records):
            if record['event'] in ('guard_started', 'guard_stopped', 'assignment_absent'):
                return False
            if (record['event'] == 'heartbeat' and record.get('assignment_present') is True
                    and record.get('socket_connected') is True):
                at = datetime.datetime.fromisoformat(record['utc']).timestamp()
                return 0 <= time.time() - at < 65
        return False
    except (OSError, ValueError, KeyError, TypeError):
        return False

def guard_running():
    try:
        pid = int((guard / 'session_guard.pid').read_text())
        return pathlib.Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0] not in ('Z', 'X')
    except (OSError, ValueError):
        return False

guard_recovery_started = state.get('guard_recovery_started_epoch')
guard_restart_attempt = state.get('guard_restart_attempt_epoch')

def maintain_guard():
    global guard_recovery_started, guard_restart_attempt
    if guard_healthy():
        was_degraded = guard_recovery_started is not None
        guard_recovery_started = guard_restart_attempt = None
        state.update(guard_healthy=True, guard_degraded=False, guard_recovery_started_epoch=None,
                     guard_restart_attempt_epoch=None)
        if was_degraded:
            write('running')
        return True
    now = time.time()
    if guard_recovery_started is None:
        guard_recovery_started = now
    write('guard_degraded', guard_healthy=False, guard_degraded=True,
          guard_recovery_started_epoch=guard_recovery_started,
          guard_recovery_elapsed_seconds=round(now - guard_recovery_started, 1))
    if now - guard_recovery_started >= 180:
        raise RuntimeError('Independent guard sustained failure for 180 seconds')
    # Live guards reconnect themselves. Restart only a dead guard whose existing
    # recorded identity AND local session record still identify the owned endpoint.
    if not guard_running() and (guard_restart_attempt is None or now - guard_restart_attempt >= 20):
        try:
            recorded = json.loads((guard / 'session_guard_identity.json').read_text())
            local = json.loads(pathlib.Path(os.environ['COLAB_SESSION_CONFIG']).read_text()).get(session)
            same_owner = recorded == {'session': session, 'endpoint': owner} and local and local['endpoint'] == owner
        except (OSError, ValueError, KeyError, TypeError):
            same_owner = False
        if same_owner:
            guard_restart_attempt = now
            write('guard_degraded', guard_restart_attempt_epoch=now)
            try:
                state['guard_restart_rc'] = call(['bash', str(root / 'tools/colab/session_guard.sh'), 'start', session,
                                                os.environ['COLAB_SESSION_CONFIG'], str(guard)],
                                               min(30, 180 - (now - guard_recovery_started)), 'guard')
            except Exception as error:
                state['guard_restart_error_type'] = type(error).__name__
    return False

def wait_for_guard():
    while time.time() < work_deadline:
        if maintain_guard():
            return
        time.sleep(max(0, min(2, work_deadline - time.time())))
    raise TimeoutError('Four-hour deadline expired waiting for a fresh guard heartbeat')

controls = set()
download = None
try:
    write('checking_ownership')
    before = identity()
    if mode == 'monitor':
        if not before['local_exists'] or not before['present']:
            raise RuntimeError('Provided owned session missing or assignment cap exceeded; no replacement')
        owner = before['endpoint']
        (out / 'ownership.json').write_text(json.dumps({'session': session, 'endpoint': owner}))
        if before['assignment_count'] > 2 or before['accelerator'] != ('NONE' if conditions == ['ANALYSIS'] else 'A100'):
            raise RuntimeError('Assignment cap or explicit A100/CPU requirement violated')
    else:
        if before['local_exists']:
            raise RuntimeError('Existing session: use monitor with explicit ownership; no rerun')
        split = state['split']
        tag = state['run_tag']
        arguments = ['-p', 'N_PROBLEMS', str(count), '-p', 'SEEDS', '17', '-p', 'SPLIT', split, '-p', 'RUN_TAG', tag, *pm_args]
        if mode == 'cpu':
            if conditions != ['ANALYSIS']:
                raise ValueError('CPU mode requires ANALYSIS only')
            arguments = ['-p', 'RUN_TAG', tag, *pm_args]
            os.environ['JEV_ANALYSIS_EXPECTED_COUNT'] = str(count)
        command = ['bash', str(root / 'tools/colab/launch_seq.sh'), session,
                   'CPU' if mode == 'cpu' else 'A100', cond_text, *arguments]
        write('launching')
        rc = call(command, 900, 'launch')
        after = identity()
        owner = after['endpoint']
        if owner:
            (out / 'ownership.json').write_text(json.dumps({'session': session, 'endpoint': owner}))
        if rc or not owner or not after['present']:
            raise RuntimeError('Launch failed or uncertain; never retry/rerun')
    write('running', owned_endpoint=owner)
    last_pull = 0
    while time.time() < work_deadline:
        tick = time.time()
        elapsed = int(tick - start)
        maintain_guard()
        observed = health()
        healthy = maintain_guard()
        write('running' if healthy else 'guard_degraded', elapsed_seconds=elapsed, health=observed)
        with (out / 'monitoring.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': state['updated_utc'], 'elapsed_seconds': elapsed, 'health': observed}) + '\n')
        for minute in (2, 5, 10, 30):
            if elapsed >= minute * 60 and minute not in controls:
                controls.add(minute)
                with (out / 'controls.jsonl').open('a') as stream:
                    stream.write(json.dumps({'minute': minute, 'elapsed_seconds': elapsed, 'health': observed}) + '\n')
                for condition in conditions:
                    if condition != 'ANALYSIS':
                        jobs.append(pool.submit(call, ['bash', str(root / 'tools/colab/status.sh'), session, condition],
                                                45, f'control_{minute}_{condition}'))
        if tick - last_pull >= 120 and (download is None or download.done()):
            # Existing pull downloads checkpoints AND final artifacts; keep them throughout the run.
            download = pool.submit(mirror_checkpoints)
            jobs.append(download)
            last_pull = tick
        if observed and observed['known'] and observed['alive'] is False:
            seq_status = observed.get('sequence_status')
            if seq_status is not None and (not seq_status.get('_done') or any(seq_status.get(c, {}).get('rc') != 0 for c in conditions)):
                raise RuntimeError('Sequence incomplete or nonzero papermill return code')
            finished = True
            break
        recovery_remaining = 180 - (time.time() - guard_recovery_started) if guard_recovery_started is not None else 25
        time.sleep(max(0, min(work_deadline - time.time(), recovery_remaining, 25 - (time.time() - tick))))
    if not finished:
        raise TimeoutError('Four-hour deadline; preserve checkpoints and release')
    finish_background()
    wait_for_guard()
    write('verifying')
    if call(['bash', str(root / 'tools/colab/pull.sh'), session, str(art)], 240, 'final_pull'):
        raise RuntimeError('Final download failed')
    wait_for_guard()
    if conditions == ['ANALYSIS']:
        # pull.sh covers result ledgers; analysis ZIP is outside results in the notebook contract.
        tag = state['run_tag']
        name = 'analysis_' + tag + '.zip'
        if call(['colab', 'download', '-s', session, '/content/jev_llm/' + name, str(art / name)], 120, 'analysis_pull'):
            raise RuntimeError('Analysis archive download failed')
        sys.path.insert(0, str(root / 'tools'))
        from verify_run import check_notebook
        notebooks = list(art.glob('07_*.out.*.ipynb'))
        if len(notebooks) != 1:
            raise RuntimeError('Missing unique completed analysis notebook')
        check_notebook(notebooks[0])
        import zipfile
        with zipfile.ZipFile(art / name) as archive:
            if archive.testzip() is not None or 'report.md' not in archive.namelist() or 'analysis.json' not in archive.namelist():
                raise RuntimeError('Incomplete analysis archive')
        verified = True
    else:
        config_file = state['config_file']
        command = ['python3', str(root / 'tools/verify_run.py'), str(art), '--conditions', *conditions,
                   '--inputs', inputs, '--config', str(root / config_file), '--expected-count', str(count),
                   '--split', state['split'],
                   '--run-tag', state['run_tag'],
                   '--latency-rows', os.environ.get('JEV_LATENCY_ROWS', '192'), '--output', str(out / 'verification.json')]
        command += ['--latency-dev-count', os.environ.get('JEV_LATENCY_DEV_COUNT', '20')]
        verified = call(command, 120, 'verification') == 0
        if not verified:
            raise RuntimeError('Final integrity verification failed; no quality rerun')
    write('verified_pending_release')
except BaseException as error:
    verified = False
    write('failed', error=f'{type(error).__name__}: {error}')
finally:
    finish_background()
    # A launch timeout may have persisted its OWN new assignment before the CLI returned.
    if owner is None and mode != 'monitor' and not before.get('local_exists', True):
        try:
            owner = identity(cleanup=True)['endpoint']
        except Exception:
            pass
    if owner:
        write('preserving_and_releasing', owned_endpoint=owner)
        try:
            call(['bash', str(root / 'tools/colab/pull.sh'), session, str(art)], 90, 'cleanup_pull', cleanup=True)
        except Exception as error:
            state['checkpoint_preservation_error'] = f'{type(error).__name__}: {error}'
        try:
            # Never release a different endpoint if somebody replaced the session record.
            config_path = pathlib.Path(os.environ['COLAB_SESSION_CONFIG'])
            local_record = json.loads(config_path.read_text()).get(session) if config_path.exists() else None
            if local_record and local_record['endpoint'] != owner:
                raise RuntimeError('Session identity changed; replacement left untouched')
            guard_record = guard / 'session_guard_identity.json'
            if guard_record.exists() and json.loads(guard_record.read_text())['endpoint'] != owner:
                raise RuntimeError('Guard identity changed; replacement left untouched')
            rc = call(['bash', str(root / 'tools/colab/stop.sh'), session], 180, 'stop', cleanup=True)
            after = identity(owner, cleanup=True)
            released = rc == 0 and after['present'] is False
        except Exception as error:
            state['release_error'] = f'{type(error).__name__}: {error}'
    write('completed' if verified and released else 'failed', completed_execution=finished,
          remaining_obligations='' if released else 'Release unconfirmed. Inspect stop/identity logs and recorded endpoint; never stop all or unowned sessions.')
sys.exit(0 if verified and released else 1)
PY
