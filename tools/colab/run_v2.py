"""Foreground, resumable v2 coordinator. --plan is offline; only --run submits work.

The independent operators retain their four-hour deadlines and cleanup on parent
death. Submission intent is durable BEFORE spawning: ambiguous starts never retry.
Unknown assignments are capacity, never cleanup targets. Requires a live WSL host.
"""

import argparse
import ast
import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from verify_run import check_notebook, verify_run

CONFIG = "config/experiment_v2.json"
RELOAD_CONFIG = "config/experiment_v2_reload.json"
DIAGNOSTIC = "config/diagnostic_08b_a100.json"
TAG = "v2_4b_confirmatory"
NOTEBOOKS = dict(G_SINGLE="01_G_SINGLE", B13="02_B13", B13_GREEDY="03_B13_GREEDY",
                 J64="04_J64", JSTEP="05_JSTEP", JFINAL="06_JFINAL",
                 LATENCY="08_estudio_latencia", ANALYSIS="07_analisis")

# Same CLI State/auth defaults as launch_seq.sh; stdout is an allowlisted schema.
BACKEND_CODE = '''import json, logging, os, sys
logging.disable(logging.CRITICAL)
try:
    from colab_cli.common import state
    state.config_path = sys.argv[1]
    state.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
    client = state.client
    request = client.session.request
    def bounded(method, url, **kwargs):
        kwargs["timeout"] = (5, 15)
        return request(method, url, **kwargs)
    client.session.request = bounded
    print(json.dumps([{"endpoint": a.endpoint, "accelerator": a.accelerator.value}
                      for a in client.list_assignments()]))
except Exception:
    sys.exit(1)
'''


class CoordinatorError(ValueError):
    """Local safety messages, distinct from untrusted subprocess/API exceptions."""


def require(value, message):
    if not value:
        raise CoordinatorError(message)


def diagnosis(error):
    message = str(error) if isinstance(error, CoordinatorError) else {
        "JSONDecodeError": "Invalid JSON in local evidence",
        "FileNotFoundError": "Required local evidence/file missing",
        "TimeoutExpired": "Bounded subprocess deadline expired; no retry",
        "InterruptedError": "Coordinator interrupted; requesting owned operator cleanup",
        "KeyboardInterrupt": "Coordinator interrupted; requesting owned operator cleanup",
        "BadZipFile": "Invalid local ZIP evidence",
    }.get(type(error).__name__, "Untrusted exception details suppressed; inspect scoped local logs")
    message = re.sub(r"https?://\S+|\beyJ[A-Za-z0-9_.-]+|(?i:Bearer)\s+\S+", "[redacted]", message)
    message = re.sub(r"(?i)(token|password|secret|authorization|credential|api[_-]?key)\s*[=:]\s*\S+",
                     r"\1=[redacted]", message)
    message = re.sub(r"[\x00-\x1f\x7f]", " ", message)[:500]
    return dict(error_type=type(error).__name__, error_message=message)


def digest(path):
    path = Path(path)
    require(path.is_file(), "Required local file missing: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), "Required local JSON evidence missing: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(value):
    # Persisted operators ran under WSL; offline --plan also works from Windows.
    match = re.fullmatch(r"/mnt/([a-z])/(.*)", str(value)) if os.name == "nt" else None
    return Path(match[1].upper() + ":/" + match[2]) if match else Path(value)


def source_version(text):
    values = [node.value.value for node in ast.parse(text).body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Constant) and any(isinstance(n, ast.Name) and n.id == "__version__" for n in node.targets)]
    require(len(values) == 1, "Missing unique frozen generator version")
    return values[0]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def plan(amend_latency_reload=False):
    def job(key, conditions, count, split, tag, **extra):
        return dict(key=key, session="jev-v2-" + key.replace("/", "-") + "-20260930",
                    conditions=conditions.split(), count=count, split=split, tag=tag,
                    config=CONFIG, latency_rows=192, latency_dev_count=20, args=[], **extra)

    hybrid = job("smoke/hybrid", "JFINAL J64 JSTEP", 1, "dev", "smoke")
    single = job("smoke/single_lat", "G_SINGLE LATENCY", 1, "dev", "smoke")
    single["session"] = "jev-v2-smoke-single-lat-20260930"
    single.update(latency_rows=36, latency_dev_count=2,
                  args=["-p", "LAT_N_DEV", "2", "-p", "LAT_SYNTH_PER_BAND", "1"])
    diagnostic = job("smoke/diagnostic_08b", "G_SINGLE", 1, "dev", "diagnostic_08b")
    diagnostic.update(config=DIAGNOSTIC, args=["-p", "KV_CACHE_GB_G", "2.0"])
    batches = [[hybrid], [single, diagnostic],
               [job("pilot/" + c, c, 40, "pilot", "pilot") for c in ("JFINAL", "B13")],
               [job("pilot/G_SINGLE", "G_SINGLE", 40, "pilot", "pilot")]]
    for pair in (("B13", "B13_GREEDY"), ("G_SINGLE", "J64"), ("JSTEP", "JFINAL")):
        batches.append([job("confirmatory/" + c, c, 100, "test", TAG) for c in pair])
    batches += [[job("latency", "LATENCY", 20, "dev", TAG)],
                [job("analysis", "ANALYSIS", 100, "test", TAG)]]
    if amend_latency_reload:
        smoke = job("smoke/latency_reload", "LATENCY", 1, "dev", "smoke")
        smoke.update(latency_rows=36, latency_dev_count=2,
                     args=["-p", "LAT_N_DEV", "2", "-p", "LAT_SYNTH_PER_BAND", "1"])
        batches = [[smoke], *batches[2:]]
        for batch in batches:
            for item in batch:
                if not item["key"].startswith("smoke/"):
                    head, slash, tail = item["key"].partition("/")
                    item["key"] = head + "_reload" + slash + tail
                item.update(config=RELOAD_CONFIG, series="v2-reload", code_version="0.2.1",
                            session="jev-v2-reload-" + item["key"].replace("/", "-") + "-20261001")
    return batches


def frozen_files(root, series="v2"):
    """Check both bundles, all member/local hashes and sealed data before any API."""
    root = Path(root).resolve()
    require(series in ("v2", "v2-reload"), "Unsupported frozen series")
    config = RELOAD_CONFIG if series == "v2-reload" else CONFIG
    required = {config, "data/v2/schedule.json",
                *["data/v2/" + s + "_inputs.jsonl" for s in ("dev", "pilot", "test")],
                *["prompts/" + n + ".txt" for n in ("generador", "criterio_paso", "criterio_final")]}
    if series == "v2":
        required.add(DIAGNOSTIC)
    else:
        amended = read_json(root / config)
        require(amended.get("code_version") == "0.2.1" and amended.get("runtime_defaults", {}).get("LATENCY_SWAP_MODE") == "reload",
                "Reload config requires code0.2.1 and phase unload/reload")
        unchanged = {k: v for k, v in amended.items() if k not in ("code_version", "amendment")}
        unchanged["runtime_defaults"] = {k: v for k, v in unchanged["runtime_defaults"].items() if k != "LATENCY_SWAP_MODE"}
        require(unchanged == read_json(root / CONFIG), "Unapproved quality/model/infrastructure config change")
    hashes = {}
    prefix = "jev_llm_v2"
    for gold, name in ((False, prefix + "_bundle.zip"), (True, prefix + "_analysis_bundle.zip")):
        path = root / "dist" / series / name
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)) and archive.testzip() is None, "Invalid bundle ZIP")
            manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
            require(manifest.get("series") == series and manifest.get("contains_gold") is gold,
                    "Wrong bundle series/gold policy")
            files = manifest["files"]
            require(required <= files.keys() and any(n.startswith("src/jevlab/") for n in files),
                    "Missing frozen config/data/code")
            require(set(names) == set(files) | {"BUNDLE_MANIFEST.json"}, "Unsealed bundle members")
            if not gold:
                require(not any("_gold." in n or "review" in n for n in names), "Gold/review in inference bundle")
            for n, expected in files.items():
                p = PurePosixPath(n)
                require(not p.is_absolute() and ".." not in p.parts and "\\" not in n,
                        "Unsafe frozen path")
                local = root / n
                require(local.resolve().is_relative_to(root), "Frozen file outside root")
                require(hashlib.sha256(archive.read(n)).hexdigest() == expected and digest(local) == expected,
                        "Bundle/file hash mismatch: " + n)
                require(n not in hashes or hashes[n] == expected, "Conflicting frozen bundles: " + n)
                hashes[n] = expected
            if series == "v2-reload":
                require("src/jevlab/__init__.py" in files, "Missing frozen generator version")
                require(manifest.get("code_version") == "0.2.1" and source_version(
                        archive.read("src/jevlab/__init__.py").decode()) == "0.2.1", "Reload requires code0.2.1")
        hashes[path.relative_to(root).as_posix()] = digest(path)
    seals = root / "data/v2/SHA256SUMS"
    seen = set()
    for line in seals.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        p = PurePosixPath(name)
        require(not p.is_absolute() and ".." not in p.parts and "\\" not in name and name not in seen,
                "Unsafe/duplicate data seal")
        seen.add(name)
        path = seals.parent / name
        require(path.resolve().is_relative_to(seals.parent.resolve()), "Data seal outside data root")
        require(digest(path) == expected, "Frozen data seal mismatch: " + name)
        hashes[path.relative_to(root).as_posix()] = expected
    require({s + suffix for s in ("dev", "pilot", "test") for suffix in ("_inputs.jsonl", "_gold.jsonl")} <= seen,
            "Missing input/gold seals")
    hashes["data/v2/SHA256SUMS"] = digest(seals)
    for name in ("operator.sh", "launch_seq.sh", "session_guard.sh", "session_guard.py",
                 "stop.sh", "pull.sh", "status.sh", "_cexec.sh"):
        hashes["tools/colab/" + name] = digest(root / "tools/colab" / name)
    for name in ("verify_run.py", "pilot_decision.py"):
        hashes["tools/" + name] = digest(root / "tools" / name)
    hashes["tools/colab/run_v2.py"] = digest(root / "tools/colab/run_v2.py")
    for name in NOTEBOOKS.values():
        path = root / "notebooks" / series / (name + ".ipynb")
        nb = read_json(path)
        parameters = "\n".join("".join(c["source"]) for c in nb["cells"]
                               if "parameters" in c.get("metadata", {}).get("tags", []))
        # These notebooks are v2-only; do not accept a v1/root/config default silently.
        defaults = {}
        for node in ast.parse(parameters).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                defaults.update({n.id: node.value.value for n in node.targets if isinstance(n, ast.Name)})
        require(all(defaults.get(k) == v for k, v in
                    {"ROOT": "/content/jev_llm_v2", "CONFIG_FILE": config, "DATA_SUBDIR": "data/v2"}.items()),
                "Notebook root/config/data sentinel mismatch: " + name)
        hashes[path.relative_to(root).as_posix()] = digest(path)
    return hashes


class Coordinator:
    def __init__(self, root=ROOT, deadline_hours=12, amend_latency_reload=False):
        require(math.isfinite(deadline_hours) and 0 < deadline_hours <= 12, "Deadline must be positive and <=12 hours")
        self.root = Path(root).resolve()
        self.out = self.root / "results/v2"
        self.amended = amend_latency_reload
        self.suffix = "_reload" if self.amended else ""
        self.config = RELOAD_CONFIG if self.amended else CONFIG
        self.path = self.out / ("coordinator" + self.suffix + "_state.json")
        self.events = self.out / ("coordinator" + self.suffix + "_events.jsonl")
        self.summary = self.out / ("summary_reload.md" if self.amended else "coordinator_report.md")
        self.pilot_key = "pilot" + self.suffix
        self.confirm_key = "confirmatory" + self.suffix
        self.latency_key = "latency" + self.suffix
        self.analysis_key = "analysis" + self.suffix
        self.hours = deadline_hours
        self.batches = plan(self.amended)
        self.jobs = {j["key"]: j for batch in self.batches for j in batch}
        self.state = {}
        self.lock = None

    def save(self, event, **fields):
        self.state.update(fields, updated_epoch=time.time())
        atomic_json(self.path, self.state)
        record = {"epoch": time.time(), "event": event, "phase": self.state.get("phase"), **fields}
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def frozen(self):
        return frozen_files(self.root, "v2-reload") if self.amended else frozen_files(self.root)

    def legacy(self, assignments=None):
        """Read-only amendment prerequisites; the failed combo is never promoted."""
        master = self.out / "coordinator_state.json"
        previous = read_json(master)
        old_jobs = {j["key"]: j for batch in plan() for j in batch}
        accepted = {"smoke/hybrid", "smoke/single_lat", "smoke/diagnostic_08b"}
        require(previous.get("version") == 1 and previous.get("outcome") == "failed"
                and previous.get("plan") == plan() and previous.get("max_total_assignments") == 2,
                "Amendment requires the original stopped v2 master, not another run")
        for field in ("submissions", "verified_jobs", "ownership"):
            require(set(previous.get(field, {})) <= accepted, "Old non-smoke work exists; amendment cannot retry it")
        require(set(previous.get("observed_jobs", [])) <= accepted and "pilot_go" not in previous,
                "Pilot already attempted; no reload amendment retry")
        for key in old_jobs.keys() - accepted:
            directory = self.out / key
            require(not directory.exists() or not any(directory.iterdir()), "Old non-smoke work exists: " + key)
        for path in (self.out / "smoke").glob("*/status.json"):
            key = path.parent.relative_to(self.out).as_posix()
            require(key in accepted or key in self.jobs, "Unapproved prior smoke exists: " + key)
        for job in self.jobs.values():
            status = self.status(job)
            require(not status or status.get("status") != "failed", "Existing failed reload work; no retry: " + job["key"])

        anchor = previous["frozen_files"]
        bundles = {}
        for gold, name in ((False, "jev_llm_v2_bundle.zip"), (True, "jev_llm_v2_analysis_bundle.zip")):
            path = self.root / "dist/v2" / name
            relative = path.relative_to(self.root).as_posix()
            require(digest(path) == anchor.get(relative), "Old immutable bundle SHA changed: " + relative)
            bundles[relative] = digest(path)
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
                files = manifest["files"]
                require(len(names) == len(set(names)) and archive.testzip() is None
                        and set(names) == set(files) | {"BUNDLE_MANIFEST.json"}
                        and manifest.get("series") == "v2" and manifest.get("contains_gold") is gold,
                        "Invalid immutable old bundle manifest")
                for name, expected in files.items():
                    require(anchor.get(name) == expected and hashlib.sha256(archive.read(name)).hexdigest() == expected,
                            "Old sealed member changed: " + name)
                require(source_version(archive.read("src/jevlab/__init__.py").decode()) == "0.2.0", "Old bundle must contain code0.2.0")
        # Current amended source cannot equal old source. Only immutable model/config,
        # prompt and data files must still match their original sealed local bytes.
        immutable = {}
        for name, expected in anchor.items():
            if name.startswith(("config/", "data/v2/", "prompts/")):
                require(digest(self.root / name) == expected, "Old immutable config/data/prompt changed: " + name)
                immutable[name] = expected

        paths = [master, self.out / "coordinator_events.jsonl", self.out / "coordinator_report.md"]
        for key in sorted(accepted):
            paths += sorted(p for p in (self.out / key).rglob("*") if p.is_file())
        require(all(p.resolve().is_relative_to(self.root) for p in paths), "Legacy evidence outside root")
        preserved = {p.relative_to(self.root).as_posix(): digest(p) for p in paths}
        evidence, endpoints = {}, {}
        for key in ("smoke/hybrid", "smoke/diagnostic_08b", "smoke/single_lat"):
            job = old_jobs[key]
            status = self.status(job)
            require(status is not None, "Missing legacy operator status: " + key)
            require((self.out / key / "ownership.json").is_file(), "Missing legacy ownership record: " + key)
            endpoint = self.endpoint(job, status)
            require(endpoint and previous["ownership"].get(key) == endpoint and status.get("released") is True,
                    "Legacy release/ownership proof missing: " + key)
            require(read_json(self.out / key / "execution_handover.json") == status, "Legacy handover/status disagree")
            partial = key == "smoke/single_lat"
            if partial:
                require(status.get("status") == "failed" and status.get("verified") is False
                        and status.get("completed_execution") is False, "Known combo must remain truthfully failed")
                health = status.get("health", {})
                sequence = health.get("sequence_status", {})
                require(health.get("known") is True and health.get("alive") is False
                        and set(sequence) == {"G_SINGLE", "LATENCY", "_done"} and sequence["_done"] is True
                        and sequence["G_SINGLE"].get("rc") == 0 and sequence["LATENCY"].get("rc") == 1,
                        "Missing isolated G success / LATENCY failure sequence proof")
                logs = list((self.out / key / "artifacts").glob("papermill_LATENCY.*.log"))
                require(len(logs) == 1 and re.search(r"CUDA (?:Error: )?out of memory|torch\.OutOfMemoryError",
                        logs[0].read_text(encoding="utf-8")), "Missing known LATENCY CUDA OOM proof")
                conditions, stored_name = ["G_SINGLE"], "gsingle_verification.json"
            else:
                require(status.get("status") == "completed" and status.get("verified") is True
                        and status.get("completed_execution") is True, "Legacy smoke not completed/verified: " + key)
                conditions, stored_name = job["conditions"], "verification.json"
            art = self.out / key / "artifacts"
            fresh = verify_run(art, conditions=conditions, inputs=self.root / "data/v2/dev_inputs.jsonl",
                               config=self.root / job["config"], expected_count=1, split="dev", run_tag=job["tag"],
                               seeds=(17,), prompts=self.root / "prompts", code_version="0.2.0")
            require(fresh.get("ok") is True and set(fresh.get("runs", {})) == set(conditions),
                    "Legacy fresh development verification failed: " + key)
            stored = read_json(self.out / key / stored_name)
            require(stored.get("ok") is True and set(stored.get("runs", {})) == set(conditions), "Missing stage-scoped stored evidence: " + key)
            require({p.name for p in art.glob("*_final.zip")} == {Path(r["archive"]).name for r in fresh["runs"].values()},
                    "Unexpected extra legacy finals")
            for condition, current in fresh["runs"].items():
                old = stored["runs"][condition]
                require(current.get("ok") is True and old.get("ok") is True and all(old.get(k) == current[k] for k in
                        ("condition", "archive_sha256", "notebook_sha256", "config_hash", "records")),
                        "Stored legacy evidence does not bind fresh stage: " + key)
                if not partial:
                    require(previous["verified_jobs"][key].get("ok") is True, "Old master did not verify completed smoke")
                    original = previous["verified_jobs"][key]["runs"][condition]
                    require(all(original.get(k) == current[k] for k in
                                ("archive_sha256", "notebook_sha256", "config_hash", "records")), "Old master evidence changed")
                current["archive"] = Path(current["archive"]).relative_to(self.root).as_posix()
            evidence[key] = dict(scope="G_SINGLE passed development stage ONLY; combo remains failed" if partial else "completed development smoke",
                                 operator_status=status["status"], verification=fresh,
                                 stored_verification=(self.out / key / stored_name).relative_to(self.root).as_posix())
            endpoints[key] = endpoint
        require(preserved == {p.relative_to(self.root).as_posix(): digest(p) for p in paths}, "Legacy evidence changed during re-audit")
        if assignments is not None:
            require(not set(endpoints.values()) & {r["endpoint"] for r in assignments}, "Legacy endpoint still present; no amended allocation")
        return dict(evidence=evidence, endpoints=endpoints, old_bundles_sha256=bundles,
                    preserved_files=preserved, immutable_files=immutable)

    def backend(self):
        python = os.environ.get("COLAB_PYTHON", "/root/.local/share/uv/tools/google-colab-cli/bin/python")
        remaining = self.state.get("deadline_epoch", float("inf")) - time.time()
        require(remaining > 0, "Global deadline expired; backend observation unavailable")
        result = subprocess.run([python, "-c", BACKEND_CODE, str(self.out / "coordinator_sessions.json")],
                                capture_output=True, text=True, timeout=min(60, remaining))
        require(result.returncode == 0, "Backend uncertain (details/credentials suppressed)")
        try:
            rows = json.loads(result.stdout)
            require(isinstance(rows, list) and all(isinstance(r, dict) and set(r) == {"endpoint", "accelerator"}
                    and isinstance(r["endpoint"], str) and r["endpoint"] and isinstance(r["accelerator"], str)
                    for r in rows), "Invalid backend schema")
            require(len({r["endpoint"] for r in rows}) == len(rows), "Duplicate backend endpoints")
        except (ValueError, TypeError):
            raise CoordinatorError("Backend uncertain (invalid response suppressed)") from None
        return rows

    def status(self, job):
        path = self.out / job["key"] / "status.json"
        if not path.exists():
            return None
        status = read_json(path)
        require(status.get("session") == job["session"] and status.get("conditions") == job["conditions"]
                and status.get("mode") == ("cpu" if job["conditions"] == ["ANALYSIS"] else "launch")
                and local_path(status["output"]).resolve() == path.parent.resolve()
                and local_path(status["cli_config"]).resolve() == (path.parent / "sessions.json").resolve(),
                "Operator identity mismatch: " + job["key"])
        for field, expected in (("series", job.get("series", "v2")), ("expected_count", job["count"]), ("split", job["split"]),
                                ("run_tag", job["tag"]), ("config_file", job["config"]), ("data_subdir", "data/v2")):
            require(field not in status or status[field] == expected, "Operator plan mismatch: " + field)
        return status

    def endpoint(self, job, status):
        directory = self.out / job["key"]
        ownership = directory / "ownership.json"
        endpoint = status.get("owned_endpoint") if status else None
        if ownership.exists():
            # operator.sh publishes this small record in-place, unlike atomic status.
            # A partial write is uncertainty, not permission to terminate good work.
            until = time.time() + 2
            while True:
                try:
                    record = read_json(ownership)
                    break
                except json.JSONDecodeError:
                    require(time.time() < until, "Ownership publication remains uncertain")
                    time.sleep(0.1)
            require(record.get("session") == job["session"] and record.get("endpoint"), "Invalid ownership record")
            require(endpoint is None or endpoint == record["endpoint"], "Conflicting endpoint ownership")
            endpoint = record["endpoint"]
        if endpoint:
            previous = self.state.setdefault("ownership", {}).get(job["key"])
            require(previous is None or previous == endpoint, "Owned endpoint changed")
            self.state["ownership"][job["key"]] = endpoint
        return endpoint

    def worker(self, job, status):
        """Identify the stdin Python worker, not a reused PID or unrelated operator."""
        if not status or not status.get("operator_pid"):
            return None
        pid = int(status["operator_pid"])
        require(pid > 1, "Invalid operator PID")
        proc = Path("/proc") / str(pid)
        try:
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] in ("Z", "X"):
                return None
            args = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            env = dict(item.split("=", 1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
        except FileNotFoundError:
            return None
        directory = self.out / job["key"]
        require(len(args) >= 8 and Path(args[0]).name.startswith("python") and args[1:5] ==
                ["-", status["mode"], job["session"], " ".join(job["conditions"])]
                and Path(args[5]).resolve() == directory and args[6] == str(job["count"])
                and Path(args[7]).resolve() == self.root / "data/v2" / (job["split"] + "_inputs.jsonl")
                and env.get("JEV_OPERATOR_OUT") == str(directory)
                and env.get("COLAB_SESSION_CONFIG") == str(directory / "sessions.json")
                and env.get("JEV_ROOT") == str(self.root) and env.get("JEV_SERIES") == job.get("series", "v2"),
                "Operator PID identity mismatch: " + job["key"])
        return pid, stat[19]  # Linux starttime, checked again immediately before SIGTERM.

    def no_worker(self, job):
        # operator.sh inherits this flock through exec. Check it even without a PID.
        result = subprocess.run(["flock", "-n", "/tmp/jev-operator-" + job["session"] + ".lock", "true"],
                                capture_output=True, timeout=5)
        require(result.returncode in (0, 1), "Operator lock uncertain")
        return result.returncode == 0

    def verify(self, job, status, assignments):
        endpoint = self.endpoint(job, status)
        require(status.get("status") == "completed" and status.get("verified") is True
                and status.get("released") is True and status.get("completed_execution") is True,
                "Operator failed or release unconfirmed: " + job["key"])
        if endpoint in {r["endpoint"] for r in assignments}:
            # Release can complete between the capacity snapshot and status read.
            assignments = self.backend()
        require(endpoint and endpoint not in {r["endpoint"] for r in assignments}, "Owned release not verified")
        directory = self.out / job["key"]
        art = directory / "artifacts"
        if job["conditions"] == ["ANALYSIS"]:
            notebooks = list(art.glob("07_*.out.*.ipynb"))
            require(len(notebooks) == 1, "Missing unique analysis notebook")
            check_notebook(notebooks[0])
            path = art / ("analysis_" + TAG + ".zip")
            with zipfile.ZipFile(path) as archive:
                require(archive.testzip() is None and {"report.md", "analysis.json"} <= set(archive.namelist()),
                        "Incomplete analysis ZIP")
                meta = json.loads(archive.read("analysis.json"))["meta"]
                run_names = {Path(entry["archive"]).name.removesuffix("_final.zip")
                             for key, info in self.state["verified_jobs"].items() if key.startswith(self.confirm_key + "/")
                             for entry in info["runs"].values()}
                require(meta.get("protocol_version") == "2" and meta.get("run_tag") == TAG
                        and meta.get("n_gold") == 100 and meta.get("n_predictions") == 600
                        and len(run_names) == 6 and set(meta.get("runs", [])) == run_names,
                        "Analysis count/tag/conditions mismatch")
                require("reviewed=false" in archive.read("report.md").decode("utf-8"),
                        "Analysis must truthfully report reviewed=false")
            report = {"ok": True, "archive": path.relative_to(self.root).as_posix(), "archive_sha256": digest(path)}
        else:
            report = verify_run(art, conditions=job["conditions"], inputs=self.root / "data/v2" / (job["split"] + "_inputs.jsonl"),
                                config=self.root / job["config"], expected_count=job["count"], split=job["split"],
                                run_tag=job["tag"], seeds=(17,), latency_rows=job["latency_rows"],
                                latency_dev_count=job["latency_dev_count"], prompts=self.root / "prompts",
                                code_version=job.get("code_version", "0.2.0"))
            require(report["ok"] is True, "Fresh final verification failed: " + job["key"])
            if job.get("series") == "v2-reload":
                for result in report["runs"].values():
                    with zipfile.ZipFile(result["archive"]) as archive:
                        names = [n for n in archive.namelist() if n == "manifest.json" or n.endswith("/manifest.json")]
                        require(len(names) == 1, "Missing reload provenance manifest")
                        cfg = json.loads(archive.read(names[0]))["config"]
                        require(cfg.get("code_version") == "0.2.1" and cfg["params"].get("CONFIG_FILE") == RELOAD_CONFIG,
                                "Reload code/config provenance mismatch")
                        if job["conditions"] == ["LATENCY"]:
                            require(cfg["params"].get("LATENCY_SWAP_MODE") == "reload", "Latency did not execute approved reload mode")
            require({p.name for p in art.glob("*_final.zip")} ==
                    {Path(r["archive"]).name for r in report["runs"].values()}, "Unexpected extra final archives")
            stored = read_json(directory / "verification.json")
            require(stored.get("ok") is True, "Stored verification failed")
            for condition, current in report["runs"].items():
                old = stored["runs"][condition]
                require(all(old.get(k) == current[k] for k in
                            ("archive_sha256", "notebook_sha256", "config_hash", "records")), "Final evidence changed")
                current["archive"] = Path(current["archive"]).relative_to(self.root).as_posix()
        self.state.setdefault("verified_jobs", {})[job["key"]] = report
        self.save("job_verified_and_released", job=job["key"])

    def launch(self, job, status, assignments):
        directory = self.out / job["key"]
        require(job["key"] != "smoke/hybrid", "Existing hybrid must never be started again")
        require(len(assignments) < 2, "No allocation capacity")
        require(not self.worker(job, status) and self.no_worker(job), "Existing running worker; do not relaunch")
        config = directory / "sessions.json"
        require(not config.exists() or read_json(config) == {}, "Existing local assignment; do not relaunch")
        require(not (directory / "ownership.json").exists() and not (status or {}).get("owned_endpoint"),
                "Existing ownership; do not relaunch")
        entry = self.state.setdefault("submissions", {}).get(job["key"])
        require(entry is None, "Ambiguous prior submission; no retries: " + job["key"])
        if status:
            require(job["key"] == "smoke/single_lat" and status.get("status") == "starting"
                    and status.get("mode") == "launch" and not status.get("operator_pid")
                    and status.get("verified") is False and status.get("released") is False,
                    "Only unstarted single_lat may be taken over")
            require(set(status) == {"status", "session", "mode", "conditions", "verified", "released",
                    "cli_config", "start_epoch", "cli_wrapper", "output", "deadline_epoch"},
                    "Not the initial unstarted operator record")
            require({p.name for p in directory.iterdir()} <= {"status.json", "execution_handover.json", "operator_bin", "coordinator_takeover.json"}
                    and not (self.root / "results/lifecycle" / job["session"]).exists(),
                    "Initial work may have started; refusing takeover")
            require(status.get("deadline_epoch", 0) > time.time() + 360, "Initial operator deadline expired")
            originals = {}
            for name in ("status.json", "execution_handover.json"):
                path = directory / name
                originals[name] = {"sha256": digest(path), "record": read_json(path),
                                   "text": path.read_bytes().decode("utf-8")}
                require(originals[name]["record"] == status, "Initial handover/status disagree")
            snapshot = directory / "coordinator_takeover.json"
            if snapshot.exists():
                require(read_json(snapshot).get("originals") == originals, "Takeover initial records changed")
                event = "takeover_proof_revalidated"
            else:
                atomic_json(snapshot, dict(epoch=time.time(), originals=originals,
                            proof="No local assignment, no live recorded PID, per-session flock available; same planned work only"))
                event = "takeover_unstarted_single_lat"
            self.save(event, job=job["key"], initial_records="smoke/single_lat/coordinator_takeover.json")
        else:
            require(not directory.exists() or not any(directory.iterdir()), "Unrecorded work exists; no rerun")
        # Recheck the frozen launch boundary; changes after initial approval stop here.
        require(self.frozen() == self.state["frozen_files"], "Frozen files changed before launch")
        if self.amended:
            require(self.legacy(assignments) == self.state["legacy_evidence"], "Legacy amendment evidence changed before launch")
        for key in self.state.get("observed_jobs", []):
            other = self.status(self.jobs[key])
            require(not other or other.get("status") != "failed", "Observed operator failed; no further allocation")
        mode = "cpu" if job["conditions"] == ["ANALYSIS"] else "launch"
        finals = self.collect() if mode == "cpu" else None
        require(time.time() < self.state["work_deadline_epoch"] and time.time() + 360 < self.state["deadline_epoch"],
                "Insufficient global deadline for safe launch")
        # Hashing/re-auditing can be slow: the earlier scheduling snapshot is not
        # allocation permission. Recheck now and retain not-yet-visible reservations.
        try:
            current = self.backend()
            require(len(current) <= 2, "Global assignment cap exceeded; no allocation")
            present = {row["endpoint"] for row in current}
            if self.amended:
                require(not set(self.state["legacy_evidence"]["endpoints"].values()) & present,
                        "Legacy endpoint still present; amended work blocked")
            reserved = 0
            for key in self.state.get("observed_jobs", []):
                planned = self.jobs[key]
                observed = self.status(planned)
                endpoint = self.endpoint(planned, observed)
                if (key in self.state["submissions"] or self.worker(planned, observed)) and (
                        not observed or observed.get("status") not in ("completed", "failed")) and endpoint not in present:
                    reserved += 1
            if len(current) + reserved >= 2:
                self.save("capacity_blocked_before_submission", blocked_reason="fresh capacity or in-flight reservations filled both slots")
                return False
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.save("backend_blocked_before_submission", blocked_reason="fresh backend/cap uncertain", **diagnosis(error))
            return False
        directory.mkdir(parents=True, exist_ok=True)
        self.state["submissions"][job["key"]] = {"epoch": time.time(), "session": job["session"], "status": "intent"}
        self.save("submission_intent", job=job["key"])
        env = {k: v for k, v in os.environ.items() if not k.startswith("JEV_") and k not in
               {"CONFIG_FILE", "DATA_SUBDIR", "SPLIT", "RUN_TAG", "COLAB_SESSION_CONFIG", "COLAB_REAL_CLI"}}
        env.update(JEV_SERIES=job.get("series", "v2"), JEV_CONFIG_FILE=job["config"], JEV_DATA_SUBDIR="data/v2", JEV_SPLIT=job["split"],
                   JEV_RUN_TAG=job["tag"], JEV_LATENCY_ROWS=str(job["latency_rows"]),
                   JEV_LATENCY_DEV_COUNT=str(job["latency_dev_count"]), JEV_DEADLINE_SECONDS="14400",
                   COLAB_SESSION_CONFIG=str(config), JEV_ANALYSIS_EXPECTED_COUNT="100")
        if mode == "cpu":
            env.update(JEV_FINALS_DIR=str(finals), JEV_GOLD_FILE=str(self.root / "data/v2/test_gold.jsonl"))
        args = ["-p", "ROOT", "/content/jev_llm_v2", *job["args"]]
        if job["key"].startswith("smoke/") and mode != "cpu":
            args += ["-p", "PREFLIGHT_MODE", "full"]
        command = ["bash", str(self.root / "tools/colab/operator.sh"), "start", mode, job["session"],
                   " ".join(job["conditions"]), str(directory), str(job["count"]),
                   str(self.root / "data/v2" / (job["split"] + "_inputs.jsonl")), *args]
        remaining = self.state["deadline_epoch"] - time.time()
        require(remaining > 360 and time.time() < self.state["work_deadline_epoch"], "Global deadline expired before submission")
        env["JEV_DEADLINE_SECONDS"] = str(min(14400, int(remaining)))
        with (directory / "coordinator_start.log").open("a") as log:
            # Timeout never kills the detached operator; reconcile its durable status, never retry.
            try:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(90, max(1, self.state["deadline_epoch"] - time.time())))
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = None
        self.state["submissions"][job["key"]].update(status="submitted", returncode=rc)
        self.save("operator_submitted", job=job["key"], returncode=rc)
        return True

    def collect(self):
        sources = {}
        for key in [self.confirm_key + "/" + c for c in ("G_SINGLE", "B13", "B13_GREEDY", "J64", "JSTEP", "JFINAL")] + [self.latency_key]:
            report = self.state["verified_jobs"][key]
            for entry in report["runs"].values():
                path = self.root / entry["archive"]
                require(path.name not in sources and digest(path) == entry["archive_sha256"], "Duplicate/changed final")
                sources[path.name] = (path, entry["archive_sha256"])
        require(len(sources) == 7, "Analysis requires exactly seven final ZIPs")
        target = self.out / ("collected_finals" + self.suffix)
        target.mkdir(exist_ok=True)
        staging_names = {Path(n).with_suffix(".tmp").name for n in sources}
        require({p.name for p in target.iterdir()} <= set(sources) | staging_names, "Unexpected collected files/checkpoints")
        for name, (source, expected) in sources.items():
            destination = target / name
            if not destination.exists():
                tmp = destination.with_suffix(".tmp")
                shutil.copyfile(source, tmp)
                require(digest(tmp) == expected, "Final changed while collecting")
                tmp.replace(destination)
            require(digest(destination) == expected, "Collected final changed")
        self.save("collected_seven_finals", finals=[target.relative_to(self.root).as_posix() + "/" + n for n in sorted(sources)])
        return target

    def pilot(self):
        path = self.out / self.pilot_key / "decision.json"
        require(self.frozen() == self.state["frozen_files"], "Frozen files changed before pilot decision")
        require(time.time() < self.state["work_deadline_epoch"], "Global work deadline expired before pilot decision")
        result = subprocess.run(["python3", str(self.root / "tools/pilot_decision.py"), "--results", str(path.parent),
                                 "--gold", str(self.root / "data/v2/pilot_gold.jsonl"), "--config", str(self.root / self.config),
                                 "--prompts", str(self.root / "prompts"), "--output", str(path)],
                                capture_output=True, timeout=min(120, self.state["deadline_epoch"] - time.time()))
        require(result.returncode in (0, 1) and path.is_file(), "Pilot decision failed; no further allocation")
        report = read_json(path)
        go = report.get("go") is True and report.get("decision") == "go" and result.returncode == 0
        require(not go or (report.get("errors") == [] and set(report.get("conditions", {})) == {"JFINAL", "B13", "G_SINGLE"}
                and all(e.get("verified") is True and e.get("graded") is True for e in report["conditions"].values())),
                "Invalid GO evidence")
        self.save("pilot_go" if go else "pilot_no_go", pilot_decision=path.relative_to(self.root).as_posix(),
                  pilot_decision_sha256=digest(path), pilot_go=go)
        self.cost()
        return go

    def cost(self):
        jobs = {}
        for key in self.state.get("verified_jobs", {}):
            job = self.jobs[key]
            if job["conditions"] == ["ANALYSIS"]:
                continue
            status = self.status(job)
            end = datetime.datetime.fromisoformat(status["updated_utc"]).timestamp()
            start = status["start_epoch"]
            if (self.out / key / "coordinator_takeover.json").exists():
                # The preserved initial epoch predates a proven never-started worker.
                start = max(start, self.state["submissions"][key]["epoch"])
            seconds = end - start
            require(math.isfinite(seconds) and seconds >= 0, "Invalid operator duration")
            stages = {}
            for c, info in self.state["verified_jobs"][key]["runs"].items():
                with zipfile.ZipFile(self.root / info["archive"]) as archive:
                    names = [n for n in archive.namelist() if n == "manifest.json" or n.endswith("/manifest.json")]
                    require(len(names) == 1, "Missing cost manifest")
                    stages[c] = json.loads(archive.read(names[0])).get("stages", {})
            stage_seconds = [stage["seconds"] for entries in stages.values() for stage in entries.values()]
            require(all(isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and s >= 0
                        for s in stage_seconds), "Invalid recorded stage duration")
            jobs[key] = dict(operator_wall_seconds=seconds, duration_start_epoch=start,
                             preserved_operator_start_epoch=status["start_epoch"], a100_hours=seconds / 3600,
                             stage_seconds_sum=sum(stage_seconds), stages=stages)
        pilot = {k: v for k, v in jobs.items() if k.startswith(self.pilot_key + "/")}
        total = sum(v["a100_hours"] for v in jobs.values())
        # Wall time includes setup/release, not an invented billed/CU measurement.
        report = dict(jobs=jobs, observed_a100_operator_hours_sum=total,
                      pilot_a100_operator_hours_sum=sum(v["a100_hours"] for v in pilot.values()),
                      pilot_recorded_stage_seconds_sum=sum(v["stage_seconds_sum"] for v in pilot.values()),
                      confirmatory_extrapolated_a100_hours=(sum(v["a100_hours"] for v in pilot.values()) * 2.5
                                                          if len(pilot) == 3 else None),
                      extrapolation_scope="100/40 scaling of three pilot arms only; other arms/latency not estimated",
                      compute_units=None, monetary_cost=None, reviewed=False,
                      note="Sum of actual operator wall durations (includes setup/cleanup), not billed hours. CU/rate unavailable."
                           + (" Reload jobs only; historical smokes/failed combo are not included." if self.amended else ""))
        path = self.out / ("cost_estimate" + self.suffix + ".json")
        atomic_json(path, report)
        self.save("cost_reestimated", cost_report=path.relative_to(self.root).as_posix())

    def cleanup(self):
        requested, waiting = [], []
        for key in self.state.get("observed_jobs", []):
            job = self.jobs[key]
            fd = None
            try:
                status = self.status(job)
                if not status or not status.get("operator_pid"):
                    continue
                # A pidfd pins the process across identity validation and signal delivery;
                # checking starttime followed by os.kill alone still permits PID reuse.
                fd = os.pidfd_open(int(status["operator_pid"]))
                identity = self.worker(job, status)
                if not identity:
                    continue
                endpoint = self.endpoint(job, status)
                config = self.out / key / "sessions.json"
                if config.exists():
                    local = read_json(config)
                    require(set(local) <= {job["session"]}, "Unowned local session")
                    if local:
                        require(endpoint and local[job["session"]]["endpoint"] == endpoint, "Cleanup ownership uncertain")
                require(self.worker(job, self.status(job)) == identity, "Operator PID changed before cleanup")
                waiting.append(key)
                requests = self.state.setdefault("cleanup_requests", {})
                if status.get("status") in ("failed", "completed", "preserving_and_releasing", "verified_pending_release"):
                    continue  # Interrupting operator.finally would interrupt release itself.
                if key not in requests:
                    requests[key] = {"pid": identity[0], "starttime": identity[1], "epoch": time.time()}
                    self.save("cleanup_signal_intent", job=key)
                    latest = self.status(job)
                    if latest.get("status") not in ("failed", "completed", "preserving_and_releasing", "verified_pending_release"):
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                        requested.append(key)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.save("cleanup_not_requested_identity_uncertain", job=key, **diagnosis(error))
            finally:
                if fd is not None:
                    os.close(fd)
        self.save("operator_cleanup_requested", cleanup_requested=requested)
        # Operators clean up independently; wait a bounded grace, never stop endpoints ourselves.
        until = min(time.time() + 360, self.state.get("deadline_epoch", float("inf")))
        while waiting and time.time() < until:
            try:
                rows = self.backend()
                endpoints = {r["endpoint"] for r in rows}
                for key in waiting[:]:
                    status = self.status(self.jobs[key])
                    endpoint = self.endpoint(self.jobs[key], status)
                    if status.get("released") is True and endpoint and endpoint not in endpoints:
                        waiting.remove(key)
                if not waiting:
                    break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                pass
            time.sleep(min(5, max(0, until - time.time())))
        self.save("cleanup_finished", cleanup_unconfirmed=waiting)

    def publish(self, outcome):
        observation = None
        try:
            rows = self.backend()
            owned = set(self.state.get("ownership", {}).values())
            observation = dict(global_assignments=len(rows), own_active_assignments=sum(r["endpoint"] in owned for r in rows),
                               external_assignments=sum(r["endpoint"] not in owned for r in rows))
            if outcome in ("no-go", "completed"):
                require(observation["own_active_assignments"] == 0, "Owned assignments still active")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            if outcome in ("no-go", "completed"):
                outcome = "blocked_release_unconfirmed"
        self.save("finished", outcome=outcome, phase="finished", current_jobs=[], assignment_observation=observation)
        summary = ["# V2 Coordinator", "", "Outcome: " + outcome, "", "reviewed=false; no quality retry.",
                   "", "Assignment observation: " + json.dumps(observation), "",
                   "- State: `" + self.path.relative_to(self.root).as_posix() + "`",
                   "- Events: `" + self.events.relative_to(self.root).as_posix() + "`"]
        if self.state.get("error_message"):
            summary.append("- Diagnostic: " + self.state["error_message"])
        if self.amended:
            summary += ["", "Amendment: phase unload/reload on the same A100; legacy artifacts unchanged.",
                        "Old G_SINGLE accepted only as a passed development stage; original combo remains failed."]
        for field in ("pilot_decision", "cost_report"):
            if self.state.get(field):
                summary.append("- " + field + ": `" + self.state[field] + "`")
        for key in self.state.get("observed_jobs", []):
            summary.append("- " + key + ": `results/v2/" + key + "/status.json`")
        if self.analysis_key in self.state.get("verified_jobs", {}):
            summary.append("- Analysis ZIP: `" + self.state["verified_jobs"][self.analysis_key]["archive"] + "`")
        target = self.summary
        tmp = target.with_suffix(".md.tmp")
        tmp.write_text("\n".join(summary) + "\n", encoding="utf-8")
        tmp.replace(target)
        return 0 if outcome == "completed" else 1

    def run(self):
        import fcntl
        hashes = self.frozen()  # No API/process launch until frozen checks pass.
        legacy = self.legacy() if self.amended else None
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / "coordinator.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise CoordinatorError("Another coordinator is active") from None
        old_handlers = {}
        def interrupted(signum, frame):
            raise InterruptedError("Coordinator interrupted")
        try:
            if self.path.exists():
                previous = read_json(self.path)
                require(previous.get("version") == 1 and previous.get("root") == str(self.root)
                        and previous.get("plan") == self.batches and previous.get("frozen_files") == hashes,
                        "Coordinator frozen state/plan mismatch")
                if self.amended:
                    require(previous.get("legacy_evidence") == legacy, "Preserved legacy amendment evidence changed")
                self.state = previous
                if self.state.get("outcome"):
                    rows = self.backend()
                    for key in list(self.state.get("verified_jobs", {})):
                        self.verify(self.jobs[key], self.status(self.jobs[key]), rows)
                    if self.state.get("pilot_decision_sha256"):
                        require(digest(self.out / self.pilot_key / "decision.json") == self.state["pilot_decision_sha256"],
                                "Published pilot decision changed")
                    return self.publish(self.state["outcome"])
                self.state["deadline_epoch"] = min(self.state["deadline_epoch"], self.state["start_epoch"] + self.hours * 3600)
            else:
                now = time.time()
                self.state = dict(version=1, root=str(self.root), plan=self.batches, frozen_files=hashes,
                                  start_epoch=now, deadline_epoch=now + self.hours * 3600, reviewed=False,
                                  max_total_assignments=2, submissions={}, ownership={}, verified_jobs={},
                                  observed_jobs=[], checkpoints={}, outcome=None)
                if self.amended:
                    self.state.update(amendment="latency-phase-reload", legacy_evidence=legacy,
                                      ownership=legacy["endpoints"].copy())
            budget = self.state["deadline_epoch"] - self.state["start_epoch"]
            self.state["work_deadline_epoch"] = self.state["deadline_epoch"] - min(480, budget / 4)
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, interrupted)
            self.save("coordinator_resumed", phase="smoke", current_jobs=[], coordinator_pid=os.getpid())
            blocked_since = self.state.get("blocked_since")
            for index, batch in enumerate(self.batches):
                if batch[0]["key"].startswith(self.confirm_key + "/") and index == (3 if self.amended else 4):
                    if not self.pilot():
                        return self.publish("no-go")
                phase = batch[0]["key"].split("/")[0]
                for job in batch:
                    key = job["key"]
                    if key not in self.state["observed_jobs"]:
                        self.state["observed_jobs"].append(key)
                    self.state["checkpoints"][key] = {
                        "status": "results/v2/" + key + "/status.json",
                        "directory": "results/v2/" + key + "/checkpoints",
                        "index": "results/v2/" + key + "/checkpoint_index.jsonl"}
                self.save("batch_started", phase=phase, batch=index, current_jobs=[j["key"] for j in batch])
                pending = list(batch)
                while pending:
                    require(time.time() < self.state["work_deadline_epoch"], "Global work deadline expired; reserve bounded cleanup")
                    try:
                        rows = self.backend()
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        if blocked_since is None:
                            blocked_since = time.time()
                        self.save("backend_blocked", phase=phase, blocked_reason="backend/cap uncertain", blocked_since=blocked_since,
                                  **diagnosis(error))
                        require(time.time() - blocked_since < 1800, "Backend/capacity blocked for 30 minutes")
                        time.sleep(min(10, max(0, self.state["deadline_epoch"] - time.time())))
                        continue
                    require(len(rows) <= 2, "Confirmed global assignment cap exceeded; stop owned work, unknowns untouched")
                    if self.amended:
                        require(not set(legacy["endpoints"].values()) & {r["endpoint"] for r in rows},
                                "Legacy endpoint still present; amended work blocked")
                    endpoints = {r["endpoint"] for r in rows}
                    # A just-submitted operator may not have allocated yet. Its reserved
                    # slot cannot be lent to its pair while the backend still looks empty.
                    in_flight = 0
                    for job in pending:
                        status = self.status(job)
                        endpoint = self.endpoint(job, status)
                        if (job["key"] in self.state["submissions"] or self.worker(job, status)) and (
                                not status or status.get("status") not in ("completed", "failed")) and endpoint not in endpoints:
                            in_flight += 1
                    slots = max(0, 2 - len(rows) - in_flight)
                    waiting = False
                    active = False
                    # Reconcile the whole pair before allocating either missing member.
                    # A resumed first member must not launch ahead of its failed partner.
                    for job in pending[:]:
                        key = job["key"]
                        status = self.status(job)
                        self.endpoint(job, status)
                        if status and status.get("status") in ("completed", "failed"):
                            self.verify(job, status, rows)
                            pending.remove(job)
                    for job in pending[:]:
                        key = job["key"]
                        status = self.status(job)
                        self.endpoint(job, status)
                        if status and status.get("status") in ("completed", "failed"):
                            self.verify(job, status, rows)
                            pending.remove(job)
                        elif self.worker(job, status):
                            active = True
                            self.save("operator_observed", job=key, operator_status=status["status"])
                        elif key == "smoke/hybrid":
                            raise ValueError("Existing hybrid worker missing; never restart")
                        elif key in self.state["submissions"]:
                            intent = self.state["submissions"][key]
                            require(time.time() - intent["epoch"] < 120, "Submitted operator missing; never retry")
                            active = True
                        elif slots:
                            if self.launch(job, status, rows):
                                slots -= 1  # Reserve even before backend reflects a new assignment.
                                active = True
                            else:
                                slots = 0
                                waiting = True
                        else:
                            waiting = True
                    if waiting and not active:
                        if blocked_since is None:
                            blocked_since = time.time()
                        self.save("capacity_blocked", blocked_reason="two assignments occupied; unknowns untouched", blocked_since=blocked_since)
                        require(time.time() - blocked_since < 1800, "Allocation capacity blocked for 30 minutes")
                    else:
                        blocked_since = None
                        self.state["blocked_since"] = None
                    if pending:
                        time.sleep(min(10, max(0, self.state["deadline_epoch"] - time.time())))
                self.save("batch_barrier_passed", batch=index, current_jobs=[])
            self.cost()
            return self.publish("completed")
        except BaseException as error:
            if not self.state:
                raise
            details = diagnosis(error)
            self.save("coordinator_stopped", phase=self.state.get("phase"), **details)
            print(json.dumps(details), file=sys.stderr, flush=True)
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            self.cleanup()
            return self.publish("interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed")
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            self.lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Check frozen files and print plan without API/execution or writes")
    mode.add_argument("--run", action="store_true", help="Run in foreground; resume existing durable state")
    parser.add_argument("--deadline-hours", type=float, default=12)
    parser.add_argument("--amend-latency-reload", action="store_true", help="Isolated approved phase-reload amendment; never retries old quality stages")
    args = parser.parse_args(argv)
    coordinator = Coordinator(deadline_hours=args.deadline_hours, amend_latency_reload=args.amend_latency_reload)
    if args.plan:
        hashes = frozen_files(ROOT, "v2-reload") if args.amend_latency_reload else frozen_files(ROOT)
        legacy = Coordinator(ROOT, args.deadline_hours, True).legacy() if args.amend_latency_reload else None
        print(json.dumps(dict(batches=coordinator.batches, frozen_files=hashes, reviewed=False,
                              max_total_assignments=2, deadline_hours=args.deadline_hours, legacy_evidence=legacy,
                              state="results/v2/" + coordinator.path.name, events="results/v2/" + coordinator.events.name,
                              summary="results/v2/" + coordinator.summary.name,
                              backend_release_check="required at --run; --plan does not contact backend"), indent=2))
        return 0
    require(os.name == "posix" and Path("/proc/self").exists(), "--run requires Linux/WSL")
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "--run requires Python/Linux pidfd support")
    return coordinator.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps(diagnosis(error)), file=sys.stderr)
        raise SystemExit(1)
