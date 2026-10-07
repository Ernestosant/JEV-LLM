"""Run with the Colab CLI's Python environment, never the notebook kernel.

watch: WebSocket pings every 15 seconds and assignment check every 20 seconds.
release: unassign ONLY the recorded session and verify removal on the server.
Transport read-timeouts are logged as uncertain, never proof of VM health.
"""

import argparse
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def event(directory, kind, **fields):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with (directory / "session_lifecycle.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
    print(json.dumps(record), flush=True)


def release(client, store, session, endpoint, directory):
    for attempt in range(1, 4):
        try:
            if endpoint in {assignment.endpoint for assignment in client.list_assignments()}:
                client.unassign(endpoint)
            absent = endpoint not in {assignment.endpoint for assignment in client.list_assignments()}
            event(directory, "release_check", attempt=attempt, assignment_absent=absent)
            if absent:
                store.remove(session)
                return True
        except Exception as error:
            # Exception bodies/URLs can contain authentication material; log only the type.
            event(directory, "release_error", attempt=attempt, error_type=type(error).__name__)
        time.sleep(5)
    return False


def connect_existing_kernel(session, proxy):
    import jupyter_kernel_client as kernels

    cls = getattr(kernels, "ColabKernelClient", None)
    token_key = "proxy_token"
    if cls is None:
        cls = getattr(kernels, "KernelClient", None) or kernels.JupyterKernelClient
        token_key = "token"
    options = {"subprotocol": kernels.JupyterSubprotocol.DEFAULT,
               "extra_params": {"colab-runtime-proxy-token": proxy.token},
               "ping_interval": 15, "reconnect_interval": 0}
    if session.session_id:
        options["session"] = session.session_id
    connection = cls(server_url=proxy.url, **{token_key: proxy.token},
                     kernel_id=session.kernel_id, client_kwargs=options,
                     headers={"X-Colab-Client-Agent": "colab-cli",
                              "X-Colab-Runtime-Proxy-Token": proxy.token})
    connection._own_kernel = False
    if not connection._manager.has_kernel:
        raise RuntimeError("Refusing to create a replacement kernel")
    connection.start(timeout=10)
    return connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("watch", "release"))
    parser.add_argument("session")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--once", action="store_true", help="One live heartbeat, then exit (verification)")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("interval must be >= 10 seconds")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    event(directory, "guard_initializing", action=args.action, session=args.session, pid=os.getpid())
    # The CLI client logs request headers and response bodies at DEBUG. Do not enable it.
    logging.disable(logging.CRITICAL)
    from colab_cli.common import state

    state.config_path = args.config
    state.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
    client = state.client
    original_request = client.session.request

    def bounded_request(method, url, **kwargs):
        kwargs.setdefault("timeout", (5, 15))
        return original_request(method, url, **kwargs)

    client.session.request = bounded_request
    identity_path = directory / "session_guard_identity.json"
    stop_path = directory / "session_guard.stop"
    session = state.store.get(args.session)
    if session:
        identity = {"session": args.session, "endpoint": session.endpoint}
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            old = json.loads(identity_path.read_text())
            if old["session"] != args.session or old["endpoint"] in {
                assignment.endpoint for assignment in client.list_assignments()
            }:
                raise SystemExit("Previous guard assignment still present; refusing ambiguous ownership")
            event(directory, "guard_identity_rollover", session=args.session)
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
    elif identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity["session"] != args.session:
            raise SystemExit("Guard identity belongs to another session")
    else:
        if args.action == "release" and not client.list_assignments():
            event(directory, "no_assignments_confirmed", session=args.session)
            return
        raise SystemExit("No local session or recorded endpoint; refusing unknown assignment")
    endpoint = identity["endpoint"]
    if args.action == "release":
        stop_path.touch()
        raise SystemExit(0 if release(client, state.store, args.session, endpoint, directory) else 1)

    import fcntl

    lock = (directory / "session_guard.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An existing guard owns this session", flush=True)
        return
    # Only attach to the known assignment; this guard never provisions resources or executes
    # dummy notebook code. Notebook progress is monitored separately by the experiment operator.
    stop_path.unlink(missing_ok=True)
    (directory / "session_guard.pid").write_text(str(os.getpid()), encoding="utf-8")
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    event(directory, "guard_started", session=args.session, pid=os.getpid(), interval_s=args.interval)
    connection = None
    attached_at = 0
    try:
        while running and not stop_path.exists():
            started = time.monotonic()
            try:
                assignments = client.list_assignments()
                assignment = next((a for a in assignments if a.endpoint == endpoint), None)
                if assignment is None:
                    event(directory, "assignment_absent", session=args.session)
                    break
                session = state.store.get(args.session)
                connected = bool(connection and connection._manager.client.connection_ready.is_set())
                # Reattach with refreshed proxy credentials before the one-hour proxy expiry.
                # Only an existing CLI kernel is allowed: no dummy code or replacement kernels.
                if session and session.kernel_id and (not connected or started - attached_at >= 3000):
                    if connection:
                        connection.stop(shutdown_kernel=False)
                    connection = connect_existing_kernel(session, assignment.runtime_proxy_info)
                    attached_at = time.monotonic()
                    connected = connection._manager.client.connection_ready.is_set()
                event(directory, "heartbeat", assignment_present=True,
                      socket_connected=connected, ping_interval_s=15,
                      waiting_existing_kernel=not bool(session and session.kernel_id),
                      duration_s=round(time.monotonic() - started, 3))
            except Exception as error:
                event(directory, "health_unknown", error_type=type(error).__name__)
            if args.once:
                break
            # No CPU spin; wait interruptibly so cancellation is independent of monitor cadence.
            until = started + args.interval
            while running and not stop_path.exists() and time.monotonic() < until:
                time.sleep(min(1, until - time.monotonic()))
    finally:
        if connection:
            connection.stop(shutdown_kernel=False)
        event(directory, "guard_stopped", session=args.session)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"event": "guard_error", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
