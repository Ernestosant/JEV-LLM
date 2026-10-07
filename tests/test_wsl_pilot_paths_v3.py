"""Offline historical pilot paths: real ZIPs and notebooks, unchanged receipts."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from jevlab.common import config_hash, read_json, read_jsonl, sha256_file
from jevlab.v3 import runner
from jevlab.v3.operational import DOMAINS, build_calendar
from test_colab_v3 import bound_consent, m, write_json
from test_operational_verify_v3 import calendar_evidence, transition
from test_verify_v3 import (
    bind_approval, complete_fixture, evidence, full_evidence, seal_public_fixture,
    snapshot, verify, write_archive,
)


def bind_policy(policy, config, report, data_dir):
    policy = deepcopy(policy)
    items = read_jsonl(data_dir / "latency_inputs.jsonl")
    order = read_json(data_dir / "schedule.json")["order"]
    metadata = {r["item_id"]: r for r in read_json(data_dir / "latency_plan.json")["items"]}
    calendar = build_calendar(items, order, metadata)
    policy.update(config_sha256=sha256_file(config), pilot_report_sha256=sha256_file(report),
                  approved_utc="2025-12-31T00:00:00Z", calendar_sha256=config_hash(calendar),
                  latency_source_plan_sha256=sha256_file(data_dir / "latency_plan.json"),
                  latency_schedule_sha256=sha256_file(data_dir / "schedule.json"),
                  cohort_a=list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "A")),
                  cohort_c=list(dict.fromkeys(r["item_id"] for r in calendar if r["cohort"] == "C")))
    return policy


@pytest.fixture
def runtime_paths(bound_consent, transition, m):
    s = bound_consent
    old, new, approval, _, _, _ = transition
    report = read_json(s.report)
    for arm, path in s.archives.items():
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            progress = archive.read("progress.json")
        manifest["config"]["code_sha256"] = old
        manifest["config_hash"] = config_hash(manifest["config"])
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("progress.json", progress)
        historical = s.root / "historical_pilots" / arm / path.name
        historical.parent.mkdir(parents=True, exist_ok=True)
        path.rename(historical)
        s.archives[arm] = path = historical
        report["conditions"][arm]["verification"].update(
            archive=str(path), archive_sha256=sha256_file(path), config_hash=manifest["config_hash"])
    write_json(s.report, report)
    policy = bind_policy(approval["operational_amendment"], s.root / m.CONFIG,
                         s.report, s.root / "data/v3")
    budget = read_json(s.approval)
    budget.update(pilot_report_sha256=sha256_file(s.report), operational_amendment=policy)
    write_json(s.approval, budget)
    s.context = SimpleNamespace(root=s.root, data_dir=s.root / "data/v3", cond="JFINAL",
        config={**s.provenance, "code_sha256": new}, manifest={"created_utc": "2026-01-02T00:00:00+00:00"},
        p=dict(SPLIT="test", APPROVAL_FILE=str(s.approval), PILOT_REPORT_FILE=str(s.report),
               CONFIRMATORY_AUTHORIZED=True))
    return s


@pytest.fixture
def auditor_paths(evidence, transition):
    full = full_evidence(evidence)
    root, inputs, config, prompts, _ = full
    old, new, approval, _, _, _ = transition
    report_path, sentinel = root / "pilot_go.json", root / "budget_approval.json"
    source_path = root / "latency_plan.json"
    source = read_json(source_path)
    for row in source["items"]:
        row["domain"] = sorted(DOMAINS)[int(row["domain"].removeprefix("domain"))]
    write_json(source_path, source)
    write_json(root / "schedule.json", {"order": [r["id"] for r in read_jsonl(inputs)]})
    seal_public_fixture(root, (inputs, root / "pilot_inputs.jsonl", root / "dataset_manifest.json",
        root / "REVIEW_CONTRACT.md", root / "latency_inputs.jsonl", source_path, root / "schedule.json"))
    report = read_json(report_path)
    archives = {}
    for arm, entry in report["conditions"].items():
        pilot_evidence = (root, root / "pilot_inputs.jsonl", *full[2:])
        data = snapshot(pilot_evidence, arm)
        data["manifest"]["config"]["code_sha256"] = old
        data["manifest"]["config"]["data_sha256"]["schedule.json"] = sha256_file(root / "schedule.json")
        complete_fixture(data, pilot_evidence)
        data["manifest"]["run_name"] = f"{arm}_mock_{data['manifest']['config_hash'][:8]}"
        path = write_archive(root / "historical_pilots" / arm, data)
        archives[arm] = path
        entry["verification"].update(archive=str(path), archive_sha256=sha256_file(path),
            notebook_sha256=sha256_file(next(path.parent.glob("*.ipynb"))),
            config_hash=data["manifest"]["config_hash"])
    write_json(report_path, report)
    policy = bind_policy(approval["operational_amendment"], config, report_path, root)
    budget = read_json(sentinel)
    budget.update(pilot_report_sha256=sha256_file(report_path), operational_amendment=policy,
                  new_verifier_sha256=sha256_file(Path(verify.__file__)))
    write_json(sentinel, budget)
    execution = snapshot(full, "B13_GREEDY", "test")
    execution["manifest"]["config"]["code_sha256"] = new
    execution["manifest"]["config"]["data_sha256"]["schedule.json"] = sha256_file(root / "schedule.json")
    complete_fixture(execution, full)
    bind_approval(execution, sentinel)
    execution["manifest"]["authorization"]["operational_amendment"] = deepcopy(policy)
    return SimpleNamespace(root=root, report=report_path, approval=sentinel, archives=archives,
                           execution=execution, inputs=inputs, config=config, prompts=prompts)


def check(s, gate):
    if gate == "runtime":
        return runner.check_budget_approval(s.context)
    return verify.check_approval(s.approval, s.report, s.execution["manifest"],
        s.config.read_bytes(), "B13_GREEDY", inputs=s.inputs, config=s.config, prompts=s.prompts)


def record_paths(s, mode, gate):
    report = read_json(s.report)
    for arm, path in s.archives.items():
        recorded = path.as_posix()
        if mode not in {"native", "posix-native"}:
            assert path.drive and len(path.drive) == 2
            recorded = "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])
            if mode == "uppercase":
                recorded = "/mnt/" + path.drive[0].upper() + "/" + "/".join(path.parts[1:])
            elif mode == "prefix":
                recorded = "/historical" + recorded
        report["conditions"][arm]["verification"]["archive"] = recorded
        # The only available copy is at the actual historical location.
        assert not (s.root / "incoming" / path.name).exists()
        assert not (s.root / "results/v3" / path.name).exists()
        assert not (s.report.parent / path.name).exists()
    s.report.write_bytes((json.dumps(report, indent=3) + "\n\n").encode())
    budget = read_json(s.approval)
    digest = sha256_file(s.report)
    budget["pilot_report_sha256"] = digest
    budget["operational_amendment"]["pilot_report_sha256"] = digest
    write_json(s.approval, budget)
    if gate == "auditor":
        bind_approval(s.execution, s.approval)
        s.execution["manifest"]["authorization"]["operational_amendment"] = deepcopy(budget["operational_amendment"])


@pytest.mark.parametrize("gate", ["runtime", "auditor"])
@pytest.mark.parametrize("mode", ["native", "wsl", "uppercase", "prefix", "posix-native", "posix-wsl"])
@pytest.mark.parametrize("changed_sha", [False, True])
def test_historical_paths_preserve_bound_evidence(request, monkeypatch, gate, mode, changed_sha):
    if mode not in {"native", "posix-native"} and os.name != "nt":
        pytest.skip("WSL-to-native-drive regression requires native Windows temporary files")
    s = request.getfixturevalue(gate + "_paths")
    record_paths(s, mode, gate)
    if mode.startswith("posix-"):
        # Isolate platform selection without replacing Path or either real resolver.
        if gate == "runtime":
            monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix"))
        else:
            monkeypatch.setattr(verify, "sys", SimpleNamespace(platform="linux", path=verify.sys.path))
    if changed_sha:
        path = next(iter(s.archives.values()))
        path.write_bytes(path.read_bytes() + b"synthetic changed transfer")
        assert sha256_file(path) != read_json(s.report)["conditions"][next(iter(s.archives))]["verification"]["archive_sha256"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in s.root.rglob("*") if p.is_file()}
    report_sha = sha256_file(s.report)
    try:
        if changed_sha or mode in {"uppercase", "prefix", "posix-wsl"}:
            error = ("missing/changed pilot archive" if gate == "runtime" else
                     "Changed public pilot final" if changed_sha and mode in {"native", "wsl", "posix-native"} else
                     "Missing public pilot final")
            with pytest.raises(ValueError, match=error):
                check(s, gate)
        else:
            result = check(s, gate)
            assert result["pilot_report_sha256"] == report_sha
            assert result["approval_sha256"] == sha256_file(s.approval)
            assert result["operational_amendment"] == read_json(s.approval)["operational_amendment"]
            if gate == "runtime":
                assert result["pilot_archive_sha256"] == {arm: sha256_file(p) for arm, p in s.archives.items()}
    finally:
        assert sha256_file(s.report) == report_sha
        assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in s.root.rglob("*") if p.is_file()}
