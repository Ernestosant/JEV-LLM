"""Execute generated collection code offline against a sandbox filesystem."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKER = {"series": "v3", "run_id": "fixture", "session": "owned", "endpoint": "fake"}


@pytest.fixture
def m():
    spec = importlib.util.spec_from_file_location("run_v3_snapshot_test", ROOT / "tools/colab/run_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / "remote"
    files = {
        "launch_v3.json": json.dumps(MARKER),
        "job_status_v3.json": json.dumps({"run_id": "fixture", "done": True, "rc": 0}),
        "results/v3/JFINAL/run/predictions.jsonl": "{}\n",
        "results/v3/JFINAL/run/metrics.jsonl": "{}\n",
        "results/v3/JFINAL/run/progress.json": "{}",
        "results/v3/JFINAL/run/manifest.json": "{}",
        "results/v3/JFINAL/run/preflight/full.json": "{}",
        "results/v3/JFINAL/run/preflight/weights.json": "{}",
        "notebooks/03_JFINAL.out.fixture.ipynb": json.dumps({
            "cells": [], "metadata": {"papermill": {"end_time": "2026-10-07T00:00:00Z"}},
            "nbformat": 4, "nbformat_minor": 5}),
        "papermill_v3.log": "complete",
        "driver_v3.log": "complete",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for name in ("JFINAL_run_checkpoint.zip", "JFINAL_run_final.zip", "other_checkpoint.zip"):
        with zipfile.ZipFile(root / "results/v3" / name, "w") as archive:
            archive.writestr("predictions.jsonl", "{}\n")
    return root, files


def generated_collection(m, root, tmp_path):
    source = m.observation_code(MARKER, collect=True)
    source = source.replace("'/content/jev_llm_v3'", repr(str(root)))
    source = source.replace("'/content/jev_v3_collect_'", repr(str(tmp_path / "collect_")))
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=20)
    messages = [json.loads(line.removeprefix("V3_JSON::")) for line in result.stdout.splitlines()
                if line.startswith("V3_JSON::")]
    return result, messages


@pytest.mark.parametrize("status,omit", [
    ({"done": True, "rc": 0}, True),
    ({"done": False, "rc": 0}, False),
    ({"done": True, "rc": 1}, False),
    ({"done": True, "rc": -1}, False),
    ({"done": True}, False),
    ({"done": "true", "rc": 0}, False),
    ({"done": True, "rc": False}, False),
    ({"done": True, "rc": "0"}, False),
    (None, False),
])
def test_generated_final_vs_active_or_failed(m, remote, tmp_path, status, omit):
    root, files = remote
    status_path = root / "job_status_v3.json"
    if status is None:
        status_path.unlink()
    else:
        status_path.write_text(json.dumps({"run_id": "fixture", **status}))
    result, messages = generated_collection(m, root, tmp_path)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(messages[-1]["snapshot"]["path"]) as archive:
        names = set(archive.namelist())
        assert ("results/v3/JFINAL_run_checkpoint.zip" not in names) == omit
        assert "results/v3/JFINAL_run_final.zip" in names
        assert "results/v3/other_checkpoint.zip" in names
        assert set(files) - {"launch_v3.json", "job_status_v3.json"} <= names
        for name in set(files) - {"launch_v3.json", "job_status_v3.json"}:
            assert archive.read(name) == (root / name).read_bytes()
        assert json.loads(archive.read("COLLECTION_MANIFEST.json"))["marker"] == MARKER
    assert (root / "results/v3/JFINAL_run_checkpoint.zip").is_file()


@pytest.mark.parametrize("final_state", ["missing", "truncated", "crc"])
def test_incomplete_final_keeps_checkpoint(m, remote, tmp_path, final_state):
    root, _ = remote
    final = root / "results/v3/JFINAL_run_final.zip"
    if final_state == "missing":
        final.unlink()
    elif final_state == "truncated":
        final.write_bytes(b"partial ZIP")
    else:
        raw = final.read_bytes()
        final.write_bytes(raw.replace(b"{}\n", b"!}\n", 1))
    result, messages = generated_collection(m, root, tmp_path)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(messages[-1]["snapshot"]["path"]) as archive:
        assert "results/v3/JFINAL_run_checkpoint.zip" in archive.namelist()


@pytest.mark.parametrize("identity", ["launch", "missing-launch", "status", "worker"])
def test_generated_marker_checks_remain_strict(m, remote, tmp_path, identity):
    root, _ = remote
    if identity == "missing-launch":
        (root / "launch_v3.json").unlink()
    else:
        name = {"launch": "launch_v3.json", "status": "job_status_v3.json", "worker": "worker_v3.json"}[identity]
        (root / name).write_text(json.dumps({**MARKER, "run_id": "foreign", "done": True, "rc": 0}))
    result, messages = generated_collection(m, root, tmp_path)
    assert result.returncode != 0
    assert messages == [{"blocked": "blocked_ownership"}]
    assert not list(tmp_path.glob("collect_*.zip"))


def test_previous_local_checkpoint_survives_final_collection(m, remote, tmp_path):
    root, _ = remote
    status = root / "job_status_v3.json"
    status.write_text(json.dumps({"run_id": "fixture", "done": False}))
    result, messages = generated_collection(m, root, tmp_path)
    assert result.returncode == 0, result.stderr
    snapshot = Path(messages[-1]["snapshot"]["path"])
    output = tmp_path / "local"
    m.preserve_snapshot(snapshot, output, MARKER, m.digest(snapshot))
    previous = {p: p.read_bytes() for p in (output / "checkpoints").glob("*.zip")}
    assert len(previous) == 2
    status.write_text(json.dumps({"run_id": "fixture", "done": True, "rc": 0}))
    result, messages = generated_collection(m, root, tmp_path)
    assert result.returncode == 0, result.stderr
    m.preserve_snapshot(snapshot, output, MARKER, m.digest(snapshot))
    assert all(p.read_bytes() == content for p, content in previous.items())
    assert (output / "artifacts/JFINAL_run_final.zip").is_file()
    assert (output / "artifacts/03_JFINAL.out.fixture.ipynb").read_bytes() == (
        root / "notebooks/03_JFINAL.out.fixture.ipynb").read_bytes()
    assert m.MAX_COLLECTION_BYTES == 512 * 1024 * 1024
