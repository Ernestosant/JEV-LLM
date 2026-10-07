"""Offline v3 safety tests. Backend/transport fixtures never authenticate or rent VMs."""

from contextlib import nullcontext
from collections import Counter
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def m():
    spec = importlib.util.spec_from_file_location("run_v3_test", ROOT / "tools/colab/run_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def forbidden(*args, **kwargs):
    pytest.fail("Offline test attempted a real tool/API/auth/allocation call")


def test_plan_is_offline_and_exact(m, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(m, "config_gate", lambda root: {})
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.subprocess, "run", forbidden)
    assert m.main(["--plan", "--phase", "test", "--root", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] and not report["allocation_permitted"]
    assert {r["status"] for r in report["readiness"]} == {"blocked_dataset_review"}
    assert len(report["jobs"]) == 6
    for item in report["jobs"]:
        assert item["params"]["ROOT"] == "/content/jev_llm_v3"
        assert item["params"]["N_PROBLEMS"] == 500
        assert item["expected_rows"] == (1500 if item["condition"] in m.SAMPLED else 500)
        assert item["params"]["SEEDS"] == ("17,29,43" if item["condition"] in m.SAMPLED else "17")
        assert item["sharding"] is False
    lat = m.plan("latency")[0]
    assert lat["expected_rows"] == 1800
    assert lat["params"]["SPLIT"] == "test"
    assert lat["params"]["LAT_N_ITEMS"] == 100 and lat["params"]["LAT_REPETITIONS"] == 3
    smoke = m.plan("smoke")
    assert all(j["params"]["SPLIT"] == "dev" and j["params"]["SEEDS"] == "17" for j in smoke)
    assert smoke[-1]["expected_rows"] == 4 and smoke[-1]["params"]["LAT_N_ITEMS"] == 2
    assert smoke[-1]["params"]["LAT_REPETITIONS"] == 1
    assert len(m.plan()) == 13 and not any(j["phase"] in {"test", "latency", "analysis"} for j in m.plan())
    assert sum(j["expected_rows"] for j in m.plan("test")) == 6000
    assert m.EXPERIMENT_ID == "jev-llm-v3-20261003" and m.MAX_GPU_SECONDS == 25 * 3600


@pytest.mark.parametrize("phase", ["prepared", "smoke", "pilot", "test", "latency"])
def test_unsealed_run_zero_backend_calls(m, tmp_path, monkeypatch, phase):
    monkeypatch.setattr(m, "flock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(m, "config_gate", lambda root: {})
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    c = m.Coordinator(tmp_path, phase, authorize_initial=True, authorize_confirmatory=phase in {"test", "latency"})
    assert c.run() == 1
    assert c.state["outcome"] == "blocked_dataset_review" and not c.state["submissions"]


@pytest.mark.parametrize("phase", ["test", "latency", "analysis"])
def test_missing_funding_fails_before_backend_or_host(m, tmp_path, monkeypatch, phase):
    monkeypatch.setattr(m, "flock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(m, "config_gate", lambda root: {})
    monkeypatch.setattr(m, "dataset_gate", lambda root: {"sealed": True})
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    c = m.Coordinator(tmp_path, phase, authorize_confirmatory=True)
    assert c.run() == 1
    assert c.state["outcome"] == "blocked_budget_approval" and not c.state["submissions"]


def test_review_failure_even_explicit_dev_bundle_blocks_gpu(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "config_gate", lambda root: {})
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    with pytest.raises(m.Blocked, match="seal") as failure:
        m.preflight(tmp_path, m.job("JFINAL", "smoke"), authorize_initial=True, dev_bundle=True)
    assert failure.value.state == "blocked_dataset_review"


def pilot_fixture(m):
    entries = {}
    for c in m.CONDITIONS:
        n = 150 if c in m.SAMPLED else 50
        entries[c] = {"verified": True, "denominator": n, "verification": {"ok": True, "condition": c, "records": n},
                      "rates": {"denominator": n, "timeout_infra_rate": 0}}
    return dict(protocol_version="3", code_version="0.3.0", go=True, decision="go", errors=[],
                budget_approval=False, conditions=entries,
                aggregate=dict(complete=True, denominator=600, observed_cases=600, timeout_infra_rate=0))


@pytest.fixture
def bound_consent(m, tmp_path):
    """Synthetic ABI fixture only: no real review, funding, run or project publication."""
    from jevlab.common import config_hash, load_prompt, sha256_text
    root = tmp_path / "synthetic-consent-project"
    write_json(root / m.CONFIG, json.loads((ROOT / m.CONFIG).read_bytes()))
    for name in ("src/jevlab/__init__.py", "src/jevlab/v3/__init__.py"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    shutil.copytree(ROOT / "prompts", root / "prompts")
    data = root / "data/v3"
    data.mkdir(parents=True)
    inputs = {}
    for split, count in (("test", 500), ("pilot", 50), ("dev", 20)):
        inputs[split] = [{"id": f"synthetic-{split}-{i}", "problem": f"Unit fixture {i} only; not a real admitted problem.", "language": "en"}
                         for i in range(count)]
        (data / (split + "_inputs.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in inputs[split]))
    latency = inputs["test"][:100]
    (data / "latency_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in latency))
    write_json(data / "schedule.json", {"order": [r["id"] for r in inputs["test"]], "pilot_order": [r["id"] for r in inputs["pilot"]]})
    domains = ["arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"]
    write_json(data / "latency_plan.json", {"items": [{"item_id": r["id"], "problem": r["problem"], "domain": domains[i // 20],
        "difficulty": "easy" if i % 3 else "medium",
        "sampling_tier": "easy" if i % 20 < 6 else "medium" if i % 20 < 14 else "hard"} for i, r in enumerate(latency)],
        "quota_axis": "source_sampling_tier", "intrinsic_difficulty_counts": {"easy": 66, "medium": 34, "hard": 0}})
    input_hashes = {p.name: m.digest(p) for p in data.iterdir()}
    write_json(data / "dataset_manifest.json", {"dataset_version": "v3", "sealed": True, "synthetic_fixture_only": True,
        "actual_n": {"test": 500, "pilot": 50, "dev": 20, "latency": 100}, "input_sha256": input_hashes,
        "review": {"agent_reviewed_all": True, "human_reviewed": False, "policy_id": "synthetic-fixture-only", "policy_sha256": "a" * 64}})
    (data / "SHA256SUMS").write_text("".join(f"{m.digest(p)}  {p.name}\n" for p in data.iterdir()))
    cfg = json.loads((root / m.CONFIG).read_bytes())
    provenance = dict(models=cfg["models"], profile=cfg["implementation_profile"], code_version="0.3.0", protocol_version="3",
        code_sha256={p.relative_to(root / "src/jevlab").as_posix(): m.digest(p) for p in (root / "src/jevlab").rglob("*.py")},
        experiment_config_sha256=m.digest(root / m.CONFIG), data_sha256={p.name: m.digest(p) for p in data.iterdir()},
        prompt_sha256={n: sha256_text(load_prompt(root / "prompts" / (n + ".txt"))) for n in ("generador", "criterio_paso", "criterio_final")})
    entries, archives = {}, {}
    for condition in m.CONDITIONS:
        count = 150 if condition in m.SAMPLED else 50
        runtime_config = {**provenance, "params": dict(CONDITION=condition, SPLIT="pilot", RUN_TAG="pilot_v3",
            SEEDS="17,29,43" if condition in m.SAMPLED else "17")}
        chash = config_hash(runtime_config)
        path = root / (condition + "_synthetic_pilot_final.zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(zipfile.ZipInfo("manifest.json"), json.dumps(dict(condition=condition, config=runtime_config, config_hash=chash,
                coverage_complete=True, finished_utc="2020-01-01T00:00:00+00:00", expected_coverage=dict(n_problems=50, n_predictions=count),
                stages={"snapshots": {"ok": True, "seconds": 30}, "main_loop": {"ok": True, "seconds": 10}})))
            archive.writestr(zipfile.ZipInfo("progress.json"), "{}")
        archives[condition] = path
        entries[condition] = dict(verified=True, denominator=count, rates=dict(denominator=count, timeout_infra_rate=0),
            verification=dict(ok=True, condition=condition, records=count, config_hash=chash, archive=str(path), archive_sha256=m.digest(path)),
            estimate={"estimated_test_walltime_s": 100}, latency_estimate={"estimated_test_walltime_s": 100})
    report_path = root / "supplied-pilot-fixture.json"
    write_json(report_path, {**pilot_fixture(m), "conditions": entries})
    approval_path = root / "supplied-consent-fixture.json"
    write_json(approval_path, dict(approved=True, approved_by="synthetic-unit-fixture-not-real-user-funding", approved_utc="2020-01-01T00:00:00+00:00",
        scope="test_and_latency", max_gpu_hours=10, config_sha256=m.digest(root / m.CONFIG), pilot_report_sha256=m.digest(report_path)))
    authorization = m.runtime_approval_gate(root, m.job("JFINAL", "test"), approval_path, report_path)
    return SimpleNamespace(root=root, approval=approval_path, report=report_path, archives=archives, provenance=provenance,
        evidence=dict(authorization=authorization, pilot_archives={c: str(p) for c, p in archives.items()}))


def test_approval_pins_config_pilot_scope_and_user_flag(m, tmp_path):
    write_json(tmp_path / m.CONFIG, {"fixture": "not real approval"})
    pilot = tmp_path / "fixture_pilot.json"
    write_json(pilot, pilot_fixture(m))
    approval = tmp_path / "fixture_approval.json"
    fixture = dict(approved=True, scope="test_and_latency", config_sha256=m.digest(tmp_path / m.CONFIG),
                   pilot_report_sha256=m.digest(pilot), approved_by="test-fixture-only", approved_utc="2020-01-01T00:00:00+00:00", max_gpu_hours=1)
    write_json(approval, fixture)
    approved, report = m.approval_gate(tmp_path, "test", approval, pilot, True, fresh=False)
    assert approved == fixture and report["go"]
    with pytest.raises(m.Blocked) as failure:
        m.approval_gate(tmp_path, "test", approval, pilot, False, fresh=False)
    assert failure.value.state == "blocked_budget_approval"
    for key, value in [("config_sha256", "b" * 64), ("pilot_report_sha256", "c" * 64),
                       ("scope", "pilot"), ("approved", "true"), ("approved_by", "")]:
        write_json(approval, {**fixture, key: value})
        with pytest.raises(m.Blocked):
            m.approval_gate(tmp_path, "test", approval, pilot, True, fresh=False)
    assert not list(tmp_path.glob("*real_approval*"))


@pytest.mark.parametrize("change", ["no-go", "version", "coverage", "quality-rule", "rates"])
def test_unknown_or_incomplete_pilot_fails_closed(m, tmp_path, change):
    p = pilot_fixture(m)
    if change == "no-go":
        p["go"] = False
    elif change == "version":
        p["code_version"] = "0.2.1"
    elif change == "coverage":
        p["conditions"]["JFINAL"]["verification"]["records"] = 50
    elif change == "quality-rule":
        p["budget_approval"] = True
    else:
        p["conditions"]["B13"]["rates"]["timeout_infra_rate"] = .05
    write_json(tmp_path / "pilot.json", p)
    with pytest.raises(m.Blocked) as failure:
        m.pilot_gate(tmp_path, tmp_path / "pilot.json", fresh=False)
    assert failure.value.state == "blocked_pilot_go"


def pilot_archive(path, seconds):
    with zipfile.ZipFile(path, "w") as z:
        stages = {name: {"seconds": seconds if name == "engines" else 10, "ok": True} for name in
                  ("environment", "snapshots", "preflight_pre_engines", "engines", "preflight_post_engines", "warmup")}
        z.writestr("manifest.json", json.dumps({"stages": {**stages, "main_loop": {"seconds": 100, "ok": True}}}))
        z.writestr("progress.json", "{}")


def test_eta_includes_setup_margin_and_rejects_unapproved_sharding(m, tmp_path):
    p = pilot_fixture(m)
    for c, entry in p["conditions"].items():
        archive = tmp_path / (c + "_pilot_final.zip")
        pilot_archive(archive, 60)
        entry["verification"].update(archive=str(archive), archive_sha256=m.digest(archive))
        entry["estimate"] = {"estimated_test_walltime_s": 100}
        entry["latency_estimate"] = {"estimated_test_walltime_s": 100}
    jobs = m.plan("test")
    estimates = m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": 10}, p)
    assert estimates[jobs[0]["key"]] == 1.5 * 210 + 900 + m.CLEANUP_SECONDS
    with pytest.raises(m.Blocked) as failure:
        m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": .1}, p)
    assert failure.value.state == "blocked_budget_or_eta"
    p["conditions"]["JFINAL"]["estimate"]["estimated_test_walltime_s"] = 14000
    with pytest.raises(m.Blocked, match="sharding unapproved"):
        m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": 100}, p)
    # Actual phase transitions, not 60 repeated downloads/native references.
    for c in ("JFINAL", "B13_GREEDY"):
        archive = Path(p["conditions"][c]["verification"]["archive"])
        pilot_archive(archive, 400)
        p["conditions"][c]["verification"]["archive_sha256"] = m.digest(archive)
    with pytest.raises(m.Blocked) as failure:
        m.estimate_jobs(tmp_path, m.plan("latency"), {"max_gpu_hours": 100}, p)
    assert failure.value.state == "blocked_budget_or_eta"


def initial_fixture(m, root, cfg, name="initial-unit-fixture.json"):
    value = dict(approved=True, approved_by="synthetic-test-only", approved_utc="1970-01-01T00:00:00+00:00",
        authorization_source="Synthetic unit fixture, not user consent", experiment_id=m.EXPERIMENT_ID,
        scope="all_gpu_phases_conditional_technical_go", max_gpu_hours=25, max_total_assignments=2,
        gpu="A100", technical_go_required=True, config_sha256=m.digest(root / m.CONFIG),
        implementation_profile=cfg["implementation_profile"], models=cfg["models"])
    path = root / name
    write_json(path, value)
    return path


@pytest.fixture
def operator(m, tmp_path, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(m.time, "time", lambda: clock["now"])
    monkeypatch.setattr(m.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(m.time, "sleep", lambda s: clock.update(now=clock["now"] + s))
    monkeypatch.setattr(m, "flock", lambda *a, **k: nullcontext(9))
    monkeypatch.setattr(m, "host_gate", lambda *a, **k: {"host_lease_id": "synthetic-test-host", "accept_allocations": True})
    item = m.job("JFINAL", "pilot")
    out = tmp_path / "results/v3/pilot/JFINAL"
    out.mkdir(parents=True)
    cfg = json.loads((ROOT / m.CONFIG).read_bytes())
    write_json(tmp_path / m.CONFIG, cfg)
    monkeypatch.setattr(m, "config_gate", lambda root: cfg)
    initial = initial_fixture(m, tmp_path, cfg)
    state = dict(job=item, root=str(tmp_path), output=str(out), series="v3", cli_config=str(out / "sessions.json"),
                 host_identity={"host_lease_id": "synthetic-test-host"},
                 run_id="fixture-run-id", start_epoch=1000.0, deadline_epoch=15400.0, evidence={"frozen_files": {}},
                 options={"initial_budget": str(initial)}, allocation_attempted=False)

    class FakeBackend:
        python = "fixture-cli-python"
        config = out / "sessions.json"
        def __init__(self):
            self.rows = [{"endpoint": "unknown-t4", "accelerator": "T4"}]
            self.endpoint = None
            self.calls = []
            self.uncertain = False
            self.timeout_new = False
            self.release_confirmed = True
            self.releases = []
        def identity(self, timeout=60):
            self.calls.append(("identity", timeout))
            if self.uncertain:
                raise m.Blocked("blocked_backend", "fixture backend uncertain")
            return {"local_endpoint": self.endpoint, "assignments": list(self.rows)}
        def call(self, args, timeout=120, capture=False):
            assert 0 < timeout <= 120
            self.calls.append((list(args), timeout))
            if args[0] == "new":
                self.endpoint = "owned-a100"
                self.rows.append({"endpoint": self.endpoint, "accelerator": "A100"})
                if self.timeout_new:
                    raise m.subprocess.TimeoutExpired("fixture-new", timeout)
            return 0, ""
    backend = FakeBackend()
    def helper_call(command, **kwargs):
        if len(command) > 2 and command[2] == m.RELEASE_CODE:
            endpoint = command[5]
            backend.releases.append(endpoint)
            if backend.release_confirmed:
                backend.rows = [r for r in backend.rows if r["endpoint"] != endpoint]
                if backend.endpoint == endpoint:
                    backend.endpoint = None
        return 0, ""
    monkeypatch.setattr(m, "bounded_call", helper_call)
    op = m.Operator(tmp_path, out, state, backend)
    return SimpleNamespace(m=m, op=op, backend=backend, out=out, clock=clock)


def test_allocation_counts_unknowns_and_records_only_own_endpoint(operator):
    s = operator
    s.op.allocate()
    news = [c for c in s.backend.calls if isinstance(c[0], list) and c[0][0] == "new"]
    assert len(news) == 1 and news[0][0][-2:] == ["--gpu", "A100"]
    assert s.op.owner == "owned-a100"
    assert s.m.read_json(s.out / "ownership.json")["output"] == str(s.out)
    assert s.backend.rows[0]["endpoint"] == "unknown-t4"
    assert s.m.read_json(s.out / "status.json")["allocation_attempted"] is True


@pytest.mark.parametrize("failure", ["capacity", "uncertain", "existing"])
def test_capacity_unknown_existing_and_uncertain_never_allocate(operator, failure):
    s = operator
    if failure == "capacity":
        s.backend.rows.append({"endpoint": "unknown-other", "accelerator": "A100"})
    elif failure == "uncertain":
        s.backend.uncertain = True
    else:
        s.backend.endpoint = "old-unowned"
    with pytest.raises(s.m.Blocked):
        s.op.allocate()
    assert not any(isinstance(c[0], list) and c[0][0] == "new" for c in s.backend.calls)


def test_release_requires_matching_owner_and_server_absence(operator, monkeypatch):
    s = operator
    s.op.allocate()
    assert s.op.release()
    assert s.backend.releases == ["owned-a100"]
    assert s.backend.rows == [{"endpoint": "unknown-t4", "accelerator": "T4"}]


def test_replacement_endpoint_never_released(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.backend.endpoint = "replacement"
    monkeypatch.setattr(s.op, "guard_call", forbidden)
    with pytest.raises(s.m.Blocked, match="replaced"):
        s.op.release()
    assert any(r["endpoint"] == "owned-a100" for r in s.backend.rows)


def test_release_return_without_server_absence_is_failure(operator, monkeypatch):
    s = operator
    s.op.allocate()
    write_json(s.op.guard / "session_guard_identity.json", {"session": s.op.item["session"], "endpoint": s.op.owner})
    s.backend.release_confirmed = False
    with pytest.raises(s.m.Blocked, match="unconfirmed"):
        s.op.release()


def test_wrong_accelerator_is_released_without_touching_unknowns(operator, monkeypatch):
    s = operator
    original = s.backend.call
    def wrong_gpu(args, **kwargs):
        result = original(args, **kwargs)
        if args[0] == "new":
            s.backend.rows[-1]["accelerator"] = "L4"
        return result
    monkeypatch.setattr(s.backend, "call", wrong_gpu)
    with pytest.raises(s.m.Blocked, match="accelerator"):
        s.op.allocate()
    assert s.op.owner == "owned-a100"
    assert s.op.release()
    assert s.backend.releases == ["owned-a100"]
    assert s.backend.rows == [{"endpoint": "unknown-t4", "accelerator": "T4"}]


def test_persistence_failure_cannot_skip_owned_release(operator, monkeypatch):
    s = operator
    s.op.allocate()
    monkeypatch.setattr(s.op, "allocate", lambda: None)
    def failed_write(*args, **kwargs):
        raise OSError("fixture disk full")
    monkeypatch.setattr(s.op, "save", failed_write)
    assert s.op.run() == 1
    assert s.op.state["persistence_failed"] is True and s.op.state["released"] is True
    assert s.backend.releases == ["owned-a100"]
    assert s.backend.rows == [{"endpoint": "unknown-t4", "accelerator": "T4"}]


def test_accumulated_budget_across_all_phases_and_whitespace_consent(operator, monkeypatch):
    s = operator
    approval = s.out.parent / "fixture_approval.json"
    write_json(approval, {"max_gpu_hours": 25, "experiment_id": s.m.EXPERIMENT_ID, "fixture_only": True})
    s.op.state["options"]["approval"] = str(approval)
    report = s.op.root / "synthetic-pilot-report.json"
    write_json(report, pilot_fixture(s.m))
    s.op.state["options"]["pilot_report"] = str(report)
    monkeypatch.setattr(s.m, "remaining_confirmatory", lambda *a: s.m.plan("test") + s.m.plan("latency"))
    monkeypatch.setattr(s.m, "estimate_jobs", lambda *a: {"synthetic": 1800})
    s.op.item = s.m.job("JFINAL", "smoke")
    s.op.state["job"] = s.op.item
    s.op.state["evidence"]["eta_seconds"] = 1800
    s.op.reserve_budget()
    assert s.op.deadline == 15400 and s.op.work_deadline == 15400 - s.m.CLEANUP_SECONDS
    s.op.state["allocation_attempted"] = True
    s.op.state["before_endpoints"] = ["unknown-t4"]
    s.backend.endpoint = "owned-a100"
    s.backend.rows.append({"endpoint": "owned-a100", "accelerator": "A100"})
    s.op.record_owner(s.backend.identity())
    s.op.observe_budget(allocating=True)
    s.clock["now"] += 1500
    assert s.op.release()
    s.op.finish_budget(True)
    ledger = s.m.read_json(s.op.state["budget_ledger"])
    assert ledger["jobs"][s.op.state["run_id"]]["actual_seconds"] == 1500
    assert Path(s.op.state["budget_ledger"]) == s.op.root / s.m.BUDGET_LEDGER
    approval.write_text(json.dumps(s.m.read_json(approval), indent=4))
    initial = Path(s.op.state["options"]["initial_budget"])
    initial.write_text(json.dumps(s.m.read_json(initial), indent=4))
    for phase, condition in [("pilot", "JFINAL"), ("test", "JFINAL"), ("latency", "LATENCY")]:
        s.op.item = s.m.job(condition, phase)
        s.op.state.update(job=s.op.item, run_id="next-" + phase, allocation_attempted=False)
        s.op.deadline = s.clock["now"] + s.m.JOB_SECONDS
        s.op.work_deadline = s.op.deadline - s.m.CLEANUP_SECONDS
        s.op.monotonic_deadline = s.op.deadline
        s.op.monotonic_work_deadline = s.op.work_deadline
        s.op.reserve_budget()
        s.op.finish_budget(True)
    ledger = s.m.read_json(s.op.state["budget_ledger"])
    assert len(ledger["jobs"]) == 4 and s.m.budget_spent(ledger) == 1500
    assert {e["key"].split("/")[0] for e in ledger["jobs"].values()} == {"smoke", "pilot", "test", "latency"}
    assert not any(isinstance(c[0], list) and c[0][0] == "new" for c in s.backend.calls)


def test_lost_new_ack_reconciles_only_scoped_owner_no_quality_retry(operator, monkeypatch):
    s = operator
    s.backend.timeout_new = True
    released = []
    monkeypatch.setattr(s.op, "release", lambda: released.append(s.op.owner) or True)
    monkeypatch.setattr(s.op, "submit", forbidden)
    assert s.op.run() == 1
    assert released == ["owned-a100"]
    assert sum(isinstance(c[0], list) and c[0][0] == "new" for c in s.backend.calls) == 1
    assert s.op.state["verified"] is False and s.op.state["released"] is True
    assert s.backend.rows[0]["endpoint"] == "unknown-t4"


def test_independent_guard_failure_bounded_180_no_remote_restart(operator, monkeypatch):
    s = operator
    monkeypatch.setattr(s.op, "submit", lambda: None)
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: {"run_id": s.op.state["run_id"], "known": True, "alive": True})
    monkeypatch.setattr(s.op, "guard_healthy", lambda: False)
    monkeypatch.setattr(s.op, "recover_guard", lambda: None)
    monkeypatch.setattr(s.op, "release", lambda: True)
    monkeypatch.setattr(s.op, "collect", lambda **k: None)
    assert s.op.run() == 1
    assert 1180 <= s.clock["now"] <= 1185
    assert s.op.state["completed_execution"] is False
    assert not any(isinstance(c[0], list) and c[0][0] in {"restart-kernel", "stop"} for c in s.backend.calls)


def test_poll_controls_incremental_collection_and_four_hour_boundary(operator, monkeypatch):
    s = operator
    original_read, original_write, original_ledger = s.m.read_json, s.m.atomic_json, s.m.budget_ledger
    serialized, checks = {}, []
    # Model mutable JSON storage, not validators, in the 550-tick fake-clock loop.
    def write(path, value):
        path = Path(path)
        serialized[path] = json.dumps(value, allow_nan=False)
        if not path.exists():
            original_write(path, value)
    def read(path):
        path = Path(path)
        return json.loads(serialized[path]) if path in serialized else original_read(path)
    def ledger(root):
        checks.append(s.clock["now"])
        return original_ledger(root)
    monkeypatch.setattr(s.m, "atomic_json", write)
    monkeypatch.setattr(s.m, "read_json", read)
    monkeypatch.setattr(s.m, "budget_ledger", ledger)
    monkeypatch.setattr(s.op, "submit", lambda: None)
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: {"run_id": s.op.state["run_id"], "known": True, "alive": True})
    monkeypatch.setattr(s.op, "guard_healthy", lambda: True)
    pulls = []
    monkeypatch.setattr(s.op, "collect", lambda **k: pulls.append(s.clock["now"]))
    assert s.op.run() == 1
    for path, value in serialized.items():
        original_write(path, json.loads(value))
    controls = [json.loads(l)["minute"] for l in (s.out / "controls.jsonl").read_text().splitlines()]
    assert controls == [2, 5, 10, 30]
    assert len(pulls) > 4
    assert s.clock["now"] <= s.op.deadline
    assert s.op.state["completed_execution"] is False
    assert len(checks) > 500
    entry = s.m.read_json(s.op.state["budget_ledger"])["jobs"][s.op.state["run_id"]]
    assert entry["release_verified"] is True and entry["reserved_seconds"] == s.op.item["deadline_seconds"] == 14400


def test_completion_needs_real_done_rc_and_verification(operator, monkeypatch):
    s = operator
    monkeypatch.setattr(s.op, "submit", lambda: None)
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: {"run_id": s.op.state["run_id"], "known": True,
                                                       "alive": False, "execution": {"done": True, "rc": 0}})
    monkeypatch.setattr(s.op, "guard_healthy", lambda: True)
    monkeypatch.setattr(s.op, "collect", lambda **k: None)
    monkeypatch.setattr(s.op, "verify", lambda: None)
    assert s.op.run() == 0 and s.op.state["verified"] is True and s.op.state["released"] is True


def make_snapshot(path, marker, entries):
    entries = dict(entries)
    entries.setdefault("job_status_v3.json", json.dumps(dict(run_id=marker["run_id"], done=True, rc=0)).encode())
    with zipfile.ZipFile(path, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
        z.writestr("COLLECTION_MANIFEST.json", json.dumps({"marker": marker,
                    "files": {n: hashlib.sha256(v).hexdigest() for n, v in entries.items()}}))


def nested_zip(content=b"fixture"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("v3/JFINAL/run/predictions.jsonl", content)
    return buffer.getvalue()


def test_collector_recurses_actual_v3_path_and_keeps_raw_outputs_checkpoints(m, tmp_path):
    marker = {"run_id": "fixture-only"}
    snapshot = tmp_path / "snapshot.zip"
    entries = {"results/v3/JFINAL/run/metrics.jsonl": b"{}\n",
               "results/v3/JFINAL_run_checkpoint.zip": nested_zip(),
               "results/v3/JFINAL_run_final.zip": nested_zip(),
               "notebooks/03_JFINAL.out.fixture.ipynb": b"{}",
               "papermill_v3.log": b"fixture"}
    make_snapshot(snapshot, marker, entries)
    output = tmp_path / "owned-output"
    m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert (output / "remote/results/v3/JFINAL/run/metrics.jsonl").read_bytes() == b"{}\n"
    assert (output / "artifacts/03_JFINAL.out.fixture.ipynb").exists()
    assert (output / "artifacts/JFINAL_run_final.zip").exists()
    assert len(list((output / "checkpoints").glob("*.zip"))) == 1
    entries["results/v3/JFINAL_run_checkpoint.zip"] = nested_zip(b"second intact fixture")
    make_snapshot(snapshot, marker, entries)
    m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert len(list((output / "checkpoints").glob("*.zip"))) == 2
    entries["results/v3/JFINAL_run_checkpoint.zip"] = b"partial transfer fixture"
    make_snapshot(snapshot, marker, entries)
    with pytest.raises(zipfile.BadZipFile):
        m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert len(list((output / "checkpoints").glob("*.zip"))) == 2


@pytest.mark.parametrize("name", ["../escape", "/content/escape", "results/v2/false_final.zip", "data/test_gold.jsonl"])
def test_collector_rejects_legacy_unsafe_or_private_paths(m, tmp_path, name):
    snapshot = tmp_path / "snapshot.zip"
    make_snapshot(snapshot, {"run_id": "fixture"}, {name: b"{}"})
    with pytest.raises(m.Blocked):
        m.preserve_snapshot(snapshot, tmp_path / "out", {"run_id": "fixture"}, m.digest(snapshot))
    assert not (tmp_path / "out").exists()


def test_generated_remote_sources_are_valid_and_do_not_reuse_legacy_helpers(m, tmp_path):
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture-only")
    item = m.job("JFINAL", "pilot")
    evidence = {"bundle": str(bundle), "notebook": str(tmp_path / "03_JFINAL.ipynb")}
    Path(evidence["notebook"]).write_bytes(b"{}")
    evidence["frozen_files"] = {evidence["notebook"]: m.digest(evidence["notebook"])}
    marker = {"run_id": "fixture"}
    for source in (m.bootstrap_code(item, evidence, marker), m.observation_code(marker), m.observation_code(marker, True)):
        compile(source, "generated_remote.py", "exec")
        assert "pull.sh" not in source and "status.sh" not in source and "restart-kernel" not in source
    source = m.observation_code(marker, True)
    assert "results/v3" in source and "rglob('*')" in source and "*.out.*.ipynb" in source
    bootstrap = m.bootstrap_code(item, evidence, marker)
    assert "launch_v3.json" in bootstrap and "Existing workspace" in bootstrap
    assert "alias.symlink_to(root" in bootstrap


@pytest.mark.parametrize("condition,phase,seeds", [
    ("JFINAL", "smoke", "17"),
    ("JFINAL", "pilot", "17,29,43"),
    ("G_GREEDY", "pilot", "17"),
])
def test_generated_remote_bootstrap_preserves_papermill_argument_types(m, tmp_path, condition, phase, seeds):
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"fixture-only")
    notebook = tmp_path / "03_fixture.ipynb"
    notebook.write_bytes(b"{}")
    evidence = dict(bundle=str(bundle), notebook=str(notebook), frozen_files={str(notebook): m.digest(notebook)})
    source = m.bootstrap_code(m.job(condition, phase), evidence, {"run_id": "fixture"})
    payload = ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                                    and any(isinstance(t, ast.Name) and t.id == "P" for t in n.targets)))
    args = payload["args"]
    assert len(args) % 3 == 0
    triples = list(zip(args[::3], args[1::3], args[2::3]))
    parameters = {key: (flag, value) for flag, key, value in triples}
    assert len(parameters) == len(triples)
    assert parameters["SEEDS"] == ("-r", seeds)
    assert parameters["SPLIT"] == ("-r", "dev" if phase == "smoke" else "pilot")
    assert parameters["RUN_TAG"] == ("-r", "smoke" if phase == "smoke" else "pilot_v3")
    assert parameters["APPROVAL_FILE"] == ("-r", "")
    assert parameters["N_PROBLEMS"] == ("-p", "1" if phase == "smoke" else "50")
    assert parameters["HASH_WEIGHTS"] == ("-p", "True")


def test_status_rejects_wrong_root_output_or_namespace(m, tmp_path):
    out = tmp_path / "results/v3/pilot/JFINAL"
    item = m.job("JFINAL", "pilot")
    state = dict(series="v3", root=str(tmp_path.resolve()), output=str(out.resolve()), job=item,
                 cli_config=str(out / "sessions.json"))
    write_json(out / "status.json", state)
    assert m.status_gate(tmp_path, out, item) == state
    for field, value in [("root", "/legacy"), ("output", "/elsewhere"), ("series", "v2"), ("cli_config", "/shared/sessions.json")]:
        write_json(out / "status.json", {**state, field: value})
        with pytest.raises(m.Blocked):
            m.status_gate(tmp_path, out, item)


def test_verifier_cli_is_v3_only_exact_physical_coverage(operator, bound_consent, monkeypatch):
    s = operator
    commands = []
    def call(command, **kwargs):
        commands.append(command)
        c = s.op.item["condition"]
        write_json(s.out / "verification.json", {"ok": True, "runs": {c: {"records": s.op.item["expected_rows"]}}})
        return 0, ""
    monkeypatch.setattr(s.m, "bounded_call", call)
    for phase, c in [("pilot", "JFINAL"), ("test", "G_GREEDY"), ("test", "JFINAL"), ("latency", "LATENCY")]:
        s.op.item = s.m.job(c, phase)
        s.op.state["options"] = {"approval": str(bound_consent.approval)} if phase != "pilot" else {}
        s.op.verify()
        cmd = commands[-1]
        assert cmd[1].endswith("verify_run_v3.py") and not cmd[1].endswith("verify_run.py")
        assert cmd[cmd.index("--expected-count") + 1] == ("50" if phase == "pilot" else "500")
        assert cmd[cmd.index("--split") + 1] == ("pilot" if phase == "pilot" else "test")
        if phase != "pilot":
            assert cmd[cmd.index("--approval") + 1] == str(bound_consent.approval)


def test_analysis_bootstrap_forwards_existing_sentinel_and_raw_notebooks(m, tmp_path, bound_consent):
    bundle = tmp_path / "analysis_bundle.zip"
    bundle.write_bytes(b"fixture-only")
    item = m.job("ANALYSIS", "analysis")
    notebook = tmp_path / "07_ANALYSIS.ipynb"
    notebook.write_bytes(b"{}")
    source = m.bootstrap_code(item, {"bundle": str(bundle), "notebook": str(notebook), "frozen_files": {str(notebook): m.digest(notebook)}, **bound_consent.evidence},
                              {"run_id": "fixture"}, "jev_v3_budget_approval.json", ["03_JFINAL.out.fixture.ipynb"])
    assert "APPROVAL_FILE" in source and "/content/jev_llm_v3/budget_approval.json" in source
    assert "03_JFINAL.out.fixture.ipynb" in source and "analysis_incoming" in source
    assert item["gpu"] == "CPU"


@pytest.mark.parametrize("condition,phase", [("JFINAL", "test"), ("LATENCY", "latency")])
def test_runtime_gate_accepts_the_operator_schema_without_api_calls(m, bound_consent, monkeypatch, condition, phase):
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    approved, report = m.approval_gate(bound_consent.root, phase, bound_consent.approval, bound_consent.report, True, fresh=False)
    archives = m.pilot_inputs(bound_consent.root, bound_consent.report)
    result = m.runtime_approval_gate(bound_consent.root, m.job(condition, phase), bound_consent.approval, bound_consent.report)
    assert result["scope"] == ("latency" if phase == "latency" else "test")
    assert result["max_gpu_hours"] == approved["max_gpu_hours"]
    assert result["pilot_report_sha256"] == m.digest(bound_consent.report)
    assert result["approval_sha256"] == m.digest(bound_consent.approval)
    assert result["pilot_archive_sha256"] == {c: m.digest(p) for c, p in archives.items()}
    assert set(report["conditions"]) == set(m.CONDITIONS)


@pytest.mark.parametrize("damage", ["missing-zip", "changed-zip", "data-binding", "coverage", "seeds"])
def test_actual_runtime_gate_rejects_missing_or_rebound_pilot_evidence(m, bound_consent, monkeypatch, damage):
    from jevlab.common import config_hash
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    path = bound_consent.archives["G_SINGLE"]
    if damage == "missing-zip":
        path.unlink()
    elif damage == "changed-zip":
        path.write_bytes(b"synthetic changed transfer")
    else:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
        if damage == "data-binding":
            manifest["config"]["data_sha256"]["pilot_inputs.jsonl"] = "0" * 64
        elif damage == "coverage":
            manifest["expected_coverage"]["n_problems"] = 49
        else:
            manifest["config"]["params"]["SEEDS"] = "17"
        manifest["config_hash"] = config_hash(manifest["config"])
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest))
        report = m.read_json(bound_consent.report)
        report["conditions"]["G_SINGLE"]["verification"].update(archive_sha256=m.digest(path), config_hash=manifest["config_hash"])
        write_json(bound_consent.report, report)
        approval = m.read_json(bound_consent.approval)
        approval["pilot_report_sha256"] = m.digest(bound_consent.report)
        write_json(bound_consent.approval, approval)
    with pytest.raises(m.Blocked) as failure:
        m.runtime_approval_gate(bound_consent.root, m.job("JFINAL", "test"), bound_consent.approval, bound_consent.report)
    assert failure.value.state == "blocked_budget_approval"


@pytest.mark.parametrize("flag", [True, "true", 1, 0])
def test_arbitrary_planner_confirmation_never_reaches_backend(m, tmp_path, monkeypatch, flag):
    item = m.job("JFINAL", "test")
    item["params"]["CONFIRMATORY_AUTHORIZED"] = flag
    monkeypatch.setattr(m, "config_gate", forbidden)
    monkeypatch.setattr(m, "dataset_gate", forbidden)
    monkeypatch.setattr(m.Backend, "call", forbidden)
    with pytest.raises(m.Blocked) as failure:
        m.preflight(tmp_path, item, authorize_confirmatory=True)
    assert failure.value.state == "blocked_budget_approval"


@pytest.mark.parametrize("phase", ["smoke", "pilot"])
@pytest.mark.parametrize("option", ["approval", "pilot_report", "authorize_confirmatory"])
def test_initial_phase_consent_options_fail_before_backend_or_host(m, tmp_path, monkeypatch, phase, option):
    monkeypatch.setattr(m, "config_gate", forbidden)
    monkeypatch.setattr(m, "dataset_gate", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    kwargs = {option: True if option == "authorize_confirmatory" else tmp_path / "unused-consent.json"}
    with pytest.raises(m.Blocked, match="Initial phases"):
        m.preflight(tmp_path, m.job("JFINAL", phase), authorize_initial=True, **kwargs)


def test_confirmatory_bootstrap_cannot_create_consent_from_a_flag(m, tmp_path):
    bundle = tmp_path / "bundle-fixture.zip"
    bundle.write_bytes(b"synthetic source fixture")
    evidence = {"bundle": str(bundle), "notebook": "03_JFINAL.ipynb"}
    with pytest.raises(m.Blocked, match="bound user approval"):
        m.bootstrap_code(m.job("JFINAL", "test"), evidence, {"run_id": "fixture"})
    with pytest.raises(m.Blocked, match="Runtime-validated"):
        m.bootstrap_code(m.job("JFINAL", "test"), evidence, {"run_id": "fixture"}, "supplied-fixture.json")
    assert list(tmp_path.iterdir()) == [bundle]


@pytest.mark.parametrize("condition,phase", [("JFINAL", "test"), ("LATENCY", "latency"), ("ANALYSIS", "analysis")])
def test_submit_uploads_unchanged_user_consent_and_six_pilots(operator, bound_consent, monkeypatch, condition, phase):
    s = operator
    s.op.root = bound_consent.root
    s.op.item = s.m.job(condition, phase)
    s.op.state.update(root=str(s.op.root), job=s.op.item, before_endpoints=["unknown-t4"])
    s.op.state["options"] = dict(approval=str(bound_consent.approval), pilot_report=str(bound_consent.report), authorize_confirmatory=True)
    bundle = bound_consent.root / "transport-fixture.zip"
    bundle.write_bytes(b"synthetic transport fixture only, not an evaluation bundle")
    notebook = bound_consent.root / (s.m.NOTEBOOKS[condition] + ".ipynb")
    notebook.write_bytes(b"{}")
    s.op.state["evidence"] = dict(bundle=str(bundle), notebook=str(notebook), frozen_files={str(notebook): s.m.digest(notebook)}, **bound_consent.evidence)
    s.backend.endpoint = "owned-a100"
    s.backend.rows.append(dict(endpoint="owned-a100", accelerator="NONE" if phase == "analysis" else "A100"))
    s.op.record_owner(s.backend.identity())
    monkeypatch.setattr(s.op, "guard_call", lambda *a, **k: None)
    staged, payloads = {}, []
    def upload(args, **kwargs):
        assert args[0] == "upload", "Submission must not allocate/restart a VM"
        staged[args[-1]] = Path(args[-2]).read_bytes()
        return 0, ""
    monkeypatch.setattr(s.backend, "call", upload)
    def remote(source, name, **kwargs):
        assert name == "bootstrap"
        payloads.append(ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                                             and any(isinstance(t, ast.Name) and t.id == "P" for t in n.targets))))
        return dict(launched=True, run_id=s.op.state["run_id"])
    monkeypatch.setattr(s.op, "remote", remote)
    if phase == "analysis":
        analysis_zip = bound_consent.root / "JFINAL_confirmatory_fixture_final.zip"
        analysis_zip.write_bytes(b"synthetic matrix transfer fixture")
        raw_nb = bound_consent.root / "03_JFINAL.out.fixture.ipynb"
        raw_nb.write_bytes(b"{}")
        monkeypatch.setattr(s.m, "analysis_inputs", lambda *a: [analysis_zip, raw_nb])
    before_approval, before_report = bound_consent.approval.read_bytes(), bound_consent.report.read_bytes()
    s.op.submit()
    payload = payloads[0]
    assert staged["/content/jev_v3_budget_approval.json"] == before_approval
    assert staged["/content/jev_v3_pilot_go.json"] == before_report
    assert set(payload["pilot_archives"]) == {p.name for p in bound_consent.archives.values()}
    assert all(staged["/content/" + p.name] == p.read_bytes() for p in bound_consent.archives.values())
    parameters = {payload["args"][i + 1]: payload["args"][i + 2] for i in range(0, len(payload["args"]), 3)}
    assert parameters["APPROVAL_FILE"] == "/content/jev_llm_v3/budget_approval.json"
    assert parameters["PILOT_REPORT_FILE"] == "/content/jev_llm_v3/pilot_go.json"
    assert parameters["CONFIRMATORY_AUTHORIZED"] == "True"
    assert not set(payload["pilot_archives"]) & set(payload["finals"])
    assert bound_consent.approval.read_bytes() == before_approval and bound_consent.report.read_bytes() == before_report
    # ABI check in a simulated remote root. The original host paths are made
    # unavailable, proving that incoming basenames satisfy the unchanged report.
    from jevlab.v3.runner import check_budget_approval
    remote_root = bound_consent.root / "simulated-remote"
    (remote_root / "incoming").mkdir(parents=True)
    (remote_root / "budget_approval.json").write_bytes(before_approval)
    (remote_root / "pilot_go.json").write_bytes(before_report)
    for path in bound_consent.archives.values():
        (remote_root / "incoming" / path.name).write_bytes(staged["/content/" + path.name])
        path.rename(path.with_suffix(".host-unavailable"))
    context = SimpleNamespace(root=remote_root, cond="LATENCY" if condition == "LATENCY" else "JFINAL", config=bound_consent.provenance,
        manifest={"created_utc": s.m.utc()}, p=dict(SPLIT="test", APPROVAL_FILE=str(remote_root / "budget_approval.json"),
            PILOT_REPORT_FILE=str(remote_root / "pilot_go.json"), CONFIRMATORY_AUTHORIZED=True))
    checked = check_budget_approval(context)
    assert checked["pilot_archive_sha256"] == payload["authorization"]["pilot_archive_sha256"]
    assert checked["pilot_report_sha256"] == s.m.digest(remote_root / "pilot_go.json")


def test_config_gate_requires_full_official_revision_not_just_prefix(m, bound_consent):
    cfg = m.read_json(bound_consent.root / m.CONFIG)
    cfg["models"]["O"]["revision"] = cfg["models"]["O"]["revision"][:7] + "0" * 33
    write_json(bound_consent.root / m.CONFIG, cfg)
    with pytest.raises(m.Blocked, match="exact official9"):
        m.config_gate(bound_consent.root)


def test_operator_matches_actual_notebook_generator_and_rejects_stale_code(m, tmp_path):
    import build_notebooks_v3 as builder
    builder.build(root=tmp_path)
    builder.build(root=tmp_path, output_dir=tmp_path / "notebooks/v3/postpilot")
    for item in [*m.plan(), *m.plan("test"), *m.plan("latency"), *m.plan("analysis")]:
        assert m.notebook_gate(tmp_path, item).is_file()
    path = tmp_path / "notebooks/v3/postpilot/07_ANALYSIS.ipynb"
    nb = json.loads(path.read_text())
    nb["cells"][-1]["source"] = "print('stale analysis fixture')"
    path.write_text(json.dumps(nb))
    with pytest.raises(m.Blocked, match="stale/changed"):
        m.notebook_gate(tmp_path, m.job("ANALYSIS", "analysis"))


def test_bootstrap_binds_isolated_postpilot_notebook_path(m, bound_consent):
    root = bound_consent.root
    bundle = root / "bundle.zip"
    notebook = root / "notebooks/v3/postpilot/03_JFINAL.ipynb"
    notebook.parent.mkdir(parents=True)
    bundle.write_bytes(b"synthetic transport")
    notebook.write_bytes(b"synthetic notebook")
    evidence = dict(bundle=str(bundle), notebook=str(notebook),
                    frozen_files={"notebooks/v3/postpilot/03_JFINAL.ipynb": m.digest(notebook)},
                    **bound_consent.evidence)
    source = m.bootstrap_code(m.job("JFINAL", "test"), evidence, {"run_id": "synthetic"}, "approval.json")
    payload = ast.literal_eval(next(node.value for node in ast.parse(source).body
                                   if isinstance(node, ast.Assign) and node.targets[0].id == "P"))
    assert payload["notebook_sha256"] == m.digest(notebook)


def test_recursive_mirrors_do_not_duplicate_canonical_pilot_discovery(m, tmp_path, monkeypatch):
    import verify_run_v3 as verifier
    def fresh(directory, *, conditions, **kwargs):
        c = conditions[0]
        archive = next(directory.glob("*_final.zip"))
        notebook = next(directory.glob("*.ipynb"))
        return {"ok": True, "runs": {c: dict(ok=True, condition=c, records=150 if c in m.SAMPLED else 50,
                    archive=str(archive), archive_sha256=m.digest(archive), notebook_sha256=m.digest(notebook), config_hash="fixture")}}
    monkeypatch.setattr(verifier, "verify_run", fresh)
    for item in m.plan("pilot"):
        out = tmp_path / "results/v3" / item["key"]
        state = dict(series="v3", root=str(tmp_path), output=str(out), job=item, cli_config=str(out / "sessions.json"),
                     status="completed", verified=True, released=True, completed_execution=True)
        for name in ("status.json", "execution_handover.json"):
            write_json(out / name, state)
        marker = {"run_id": item["condition"] + "-fixture"}
        snapshot = tmp_path / (item["condition"] + "_snapshot.zip")
        make_snapshot(snapshot, marker, {"results/v3/" + item["condition"] + "_pilot_v3_fixture_final.zip": nested_zip(),
            "notebooks/" + m.NOTEBOOKS[item["condition"]] + ".out.fixture.ipynb": b"{}"})
        m.preserve_snapshot(snapshot, out, marker, m.digest(snapshot))
        write_json(out / "verification.json", fresh(out / "artifacts", conditions=[item["condition"]]))
    assert len(list((tmp_path / "results/v3/pilot").rglob("*_final.zip"))) == 12
    canonical = m.canonical_pilot(tmp_path)
    assert len(list(canonical.rglob("*_final.zip"))) == 6
    assert len(list(canonical.rglob("*.out.*.ipynb"))) == 6
    assert m.canonical_pilot(tmp_path) == canonical


def test_phase_calendar_matches_actual_runtime_fake_case_loop(m, tmp_path, monkeypatch):
    from jevlab.v3 import algorithms, latency_study
    exp = object.__new__(latency_study.LatencyStudy)
    exp.manifest = {"authorization": {}}
    exp.p = dict(SPLIT="test", LAT_BLOCK_SIZE=10, LAT_REPETITIONS=3, TIMEOUT_S=1, CHECKPOINT_EVERY=5)
    exp.seeds, exp.conds = [17, 29, 43], ["JFINAL", "B13_GREEDY"]
    exp.dir, exp.chash, exp.gpu_uuid, exp.ctx, exp.phase_epoch = tmp_path, "synthetic-calendar", "fake-a100", None, 0
    exp.f = {key: str(tmp_path / (key + ".jsonl")) for key in ("predictions", "proposals", "latency_metrics")}
    items = [{"id": str(i), "problem": "Synthetic loop fixture", "language": "en"} for i in range(100)]
    exp.latency_items = lambda: items
    domains = ("arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability")
    exp._frozen_latency_plan = lambda subset: [{"item_id": row["id"], "problem": row["problem"],
        "domain": domains[int(row["id"]) // 20], "difficulty": "easy" if int(row["id"]) % 3 else "medium",
        "sampling_tier": "easy" if int(row["id"]) % 20 < 6 else "medium" if int(row["id"]) % 20 < 14 else "hard"}
        for row in subset.values()]
    exp.dataset_seals = {"latency_plan.json": "0" * 64}
    exp._check_authorization = lambda: None
    exp._validate_identity = lambda *a: None
    exp._completed_keys = lambda: set()
    exp.phase_id, exp.engines, exp.phase_cleanup_confirmed = "synthetic-phase", {}, True
    exp.assert_same_gpu = lambda: None
    exp._provenance = lambda condition: {}
    exp._case_context = lambda: {"context_frame": {"synthetic": True}, "context_frame_id": "synthetic-frame", "execution_id": str(len(executions))}
    exp._proposal_callback = lambda *a: None
    exp.checkpoint = lambda: None
    exp.stage = lambda *a: nullcontext()
    loads, executions, active = [], [], {"phase": None}
    def swap(phase):
        if phase != active["phase"]:
            loads.append(phase)
            active["phase"] = phase
            exp.phase_epoch += 1
    exp._swap = swap
    measured_metadata = []
    def persist(out, key, start, extra, **kwargs):
        executions.append((extra["repetition"], extra["block"], extra["phase"], extra["seed"], int(extra["item_id"])))
        measured_metadata.append(dict(extra))
    exp._persist = persist
    monkeypatch.setattr(algorithms, "run_case", lambda *a, **k: SimpleNamespace(run={"raw_output": "synthetic"}))
    exp.run()
    assert executions == list(m.phase_calendar())
    assert len(executions) == 1800 and Counter(loads) == {"B": 17, "H": 16}
    assert all(row["difficulty"] != "hard" for row in measured_metadata)
    assert sum(row["sampling_tier"] == "hard" for row in measured_metadata) == 30 * 18
    assert len({row["execution_id"] for row in measured_metadata}) == 1800
    small = list(m.phase_calendar(n_items=7, block_size=3, seeds=(17,), repetitions=2))
    assert len(small) == 28 and {r[-1] for r in small} == set(range(7))


def test_latency_eta_prepares_once_and_counts_only_real_reloads(m, tmp_path):
    report = pilot_fixture(m)
    for condition, entry in report["conditions"].items():
        archive = tmp_path / (condition + "_timing_final.zip")
        pilot_archive(archive, 60)
        entry["verification"].update(archive=str(archive), archive_sha256=m.digest(archive))
        entry["latency_estimate"] = {"estimated_test_walltime_s": 100}
    # Environment/preflight 20 once; disjoint snapshots 20 once; 33 lifecycle loads at 80 each.
    expected = 1.5 * (200 + 40 + 33 * 80) + 900 + m.CLEANUP_SECONDS
    assert m.estimate_jobs(tmp_path, m.plan("latency"), {"max_gpu_hours": 25}, report) == {"latency/LATENCY": expected}


def test_eta_reads_each_immutable_artifact_once_but_revalidates_next_call(m, tmp_path, monkeypatch):
    import verify_run_v3 as verifier
    report = pilot_fixture(m)
    for condition, entry in report["conditions"].items():
        archive = tmp_path / (condition + "_timing_final.zip")
        pilot_archive(archive, 60)
        entry["verification"].update(archive=str(archive), archive_sha256=m.digest(archive))
        entry["estimate"] = entry["latency_estimate"] = {"estimated_test_walltime_s": 100}
    reads, original = [], verifier.read_archive
    def read(path):
        reads.append(Path(path))
        return original(path)
    monkeypatch.setattr(verifier, "read_archive", read)
    jobs = m.plan("test") + m.plan("latency")
    estimates = m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": 25}, report)
    assert len(estimates) == 7 and len(reads) == len(set(reads)) == 6
    assert m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": 25}, report) == estimates
    assert len(reads) == 12  # No process-global success cache across allocation gates.
    archive = Path(report["conditions"]["JFINAL"]["verification"]["archive"])
    pilot_archive(archive, 90)
    with pytest.raises(m.Blocked, match="timing archive changed"):
        m.estimate_jobs(tmp_path, jobs, {"max_gpu_hours": 25}, report)


def test_remote_notebook_hash_failure_executes_zero_processes(m, tmp_path, monkeypatch):
    notebook = tmp_path / "03_JFINAL.ipynb"
    notebook.write_bytes(b'{"frozen":"local"}')
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"unused-before-notebook-hash")
    evidence = dict(notebook=str(notebook), bundle=str(bundle), frozen_files={str(notebook): m.digest(notebook)})
    source = m.bootstrap_code(m.job("JFINAL", "pilot"), evidence, {"run_id": "synthetic"})
    content = tmp_path / "content"
    content.mkdir()
    (content / notebook.name).write_bytes(b'{"changed":"upload"}')
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="Uploaded notebook changed"):
        exec(source.replace("/content", content.as_posix()), {})
    assert not (content / "jev_llm_v3").exists()


def spent_fixture(m, seconds):
    return dict(experiment_id=m.EXPERIMENT_ID, max_gpu_seconds=m.MAX_GPU_SECONDS, jobs={"synthetic-prior":
        dict(key="smoke/prior", session="fixture-prior", output="synthetic-prior", reserved_seconds=m.JOB_SECONDS,
             reserved_epoch=0, actual_seconds=seconds, released=True, release_verified=True, release_evidence="synthetic-test-only")})


def test_worst_case_reservation_and_tail_charges_are_bounded(operator):
    s = operator
    ledger_path = s.op.root / s.m.BUDGET_LEDGER
    write_json(ledger_path, spent_fixture(s.m, s.m.MAX_GPU_SECONDS - 1200))
    s.op.allocate()
    ledger = s.m.read_json(ledger_path)
    entry = ledger["jobs"][s.op.state["run_id"]]
    assert entry["reserved_seconds"] == 1200 and s.op.deadline == s.clock["now"] + 1200
    assert s.m.budget_spent(ledger) == s.m.MAX_GPU_SECONDS
    s.clock["now"] += 1100
    assert s.op.release()
    s.op.finish_budget(True)
    ledger = s.m.read_json(ledger_path)
    assert ledger["jobs"][s.op.state["run_id"]]["actual_seconds"] == 1100
    assert s.m.budget_spent(ledger) == s.m.MAX_GPU_SECONDS - 100
    s.op.state.update(run_id="synthetic-no-tail", allocation_attempted=False)
    s.op.deadline = s.clock["now"] + s.m.JOB_SECONDS
    s.op.work_deadline = s.op.deadline - s.m.CLEANUP_SECONDS
    s.op.monotonic_deadline = s.op.deadline
    s.op.monotonic_work_deadline = s.op.work_deadline
    with pytest.raises(s.m.Blocked, match="bounded tail"):
        s.op.reserve_budget()


def test_refund_cannot_trust_release_boolean(operator):
    s = operator
    s.op.allocate()
    s.clock["now"] += 50
    with pytest.raises(s.m.Blocked, match="verified server release"):
        s.op.finish_budget(True)
    ledger = s.m.read_json(s.op.state["budget_ledger"])
    entry = ledger["jobs"][s.op.state["run_id"]]
    assert not entry["released"] and entry["observed_seconds"] == 50
    assert s.m.budget_spent(ledger) == s.op.item["deadline_seconds"] == 14400


def test_two_fake_process_leases_share_fixed_global_cap(operator, monkeypatch):
    s = operator
    rows, backends = [], {}
    class LeaseBackend:
        python = "fake-cli-only"
        def __init__(self, directory, session):
            self.config, self.session, self.endpoint = directory / "sessions.json", session, None
            self.calls = []
        def identity(self, timeout=60):
            return dict(local_endpoint=self.endpoint, assignments=list(rows))
        def call(self, args, **kwargs):
            self.calls.append(args)
            assert args[0] == "new"
            self.endpoint = self.session + "-owned"
            rows.append(dict(endpoint=self.endpoint, accelerator="A100"))
            return 0, ""
    def make(index, phase):
        item = s.m.job("JFINAL", phase, "fake-process-" + str(index))
        out = s.op.root / "results/v3" / ("fake-process-" + str(index))
        out.mkdir()
        state = dict(series="v3", root=str(s.op.root), output=str(out), cli_config=str(out / "sessions.json"),
            host_identity=dict(s.op.state["host_identity"]),
            job=item, run_id="fake-process-" + str(index), start_epoch=s.clock["now"], deadline_epoch=s.clock["now"] + item["deadline_seconds"],
            evidence={"frozen_files": {}}, options=dict(s.op.state["options"]), allocation_attempted=False)
        backend = LeaseBackend(out, item["session"])
        backends[item["session"]] = backend
        return s.m.Operator(s.op.root, out, state, backend)
    monkeypatch.setattr(s.m, "Backend", lambda directory, session: backends[session])
    first, second, third = make(1, "smoke"), make(2, "pilot"), make(3, "pilot")
    first.allocate()
    second.allocate()
    with pytest.raises(s.m.Blocked, match="Two live GPU"):
        third.allocate()
    ledger = s.m.read_json(s.op.root / s.m.BUDGET_LEDGER)
    assert len(rows) == 2 and len(ledger["jobs"]) == 2
    assert all(e["reserved_seconds"] == s.m.LEGACY_JOB_SECONDS == 14400 for e in ledger["jobs"].values())
    assert not third.backend.calls


def test_prior_lease_reconciled_without_renewal_on_approval_changes(operator, monkeypatch):
    s = operator
    s.op.allocate()
    original_deadline = s.op.deadline
    s.clock["now"] += 300
    approval = s.op.root / "reformatted-parent-fixture.json"
    write_json(approval, {"experiment_id": s.m.EXPERIMENT_ID, "max_gpu_hours": 25, "fixture": True})
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    ledger = s.m.budget_ledger(s.op.root)
    s.op.reconcile_budget(ledger)
    entry = ledger["jobs"][s.op.state["run_id"]]
    assert entry["deadline_epoch"] == original_deadline and entry["reserved_seconds"] == s.op.item["deadline_seconds"] == 14400
    assert entry["observed_seconds"] == 300 and not entry["released"]
    # Authoritative absence, not the new approval filename, makes a refund possible.
    s.backend.rows = [r for r in s.backend.rows if r["endpoint"] != s.op.owner]
    s.op.reconcile_budget(ledger)
    assert entry["released"] and entry["release_verified"] and entry["actual_seconds"] == 300
    assert s.m.budget_spent(ledger) == 300


def test_parent_source_tier_config_rebinding_cannot_reset_budget_or_renew_lease(operator):
    import make_bundle_v3 as bundles
    s = operator
    s.op.allocate()
    s.clock["now"] += 200
    assert s.op.release()
    s.op.finish_budget(True)
    path = Path(s.op.state["budget_ledger"])
    previous = s.m.read_json(path)
    old_initial = s.op.state["options"]["initial_budget"]
    cfg = s.m.config_gate(s.op.root)
    cfg["dataset_review_policy"] = {"id": bundles.REVIEW_POLICY_ID, "sha256": bundles.REVIEW_CONTRACT_SHA256}
    cfg["dataset"]["quota_axis"] = "source_sampling_tier"
    write_json(s.op.root / s.m.CONFIG, cfg)
    with pytest.raises(s.m.Blocked, match="Initial consent"):
        s.m.initial_budget_gate(s.op.root, old_initial)
    new_initial = initial_fixture(s.m, s.op.root, cfg, name="synthetic-parent-new-policy.json")
    s.op.state["options"]["initial_budget"] = str(new_initial)
    assert s.m.initial_budget_gate(s.op.root, new_initial)["experiment_id"] == s.m.EXPERIMENT_ID
    with pytest.raises(s.m.Blocked, match="Prior budget reservation"):
        s.op.reserve_budget()
    assert s.m.read_json(path) == previous
    assert s.m.budget_spent(s.m.budget_ledger(s.op.root)) == 200


@pytest.mark.parametrize("artifact", ["results/v3/JFINAL_run_final.zip", "notebooks/03_JFINAL.out.fixture.ipynb", "analysis_fixture.zip"])
def test_collection_conflict_is_checked_before_any_mirror_mutation(m, tmp_path, artifact):
    snapshot, output, marker = tmp_path / "snapshot.zip", tmp_path / "output", {"run_id": "fixture"}
    original = nested_zip() if artifact.endswith(".zip") else b'{"original":true}'
    entries = {"papermill_v3.log": b"original-log", artifact: original}
    make_snapshot(snapshot, marker, entries)
    m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    entries["papermill_v3.log"] = b"changed-log"
    entries[artifact] = nested_zip(b"changed") if artifact.endswith(".zip") else b'{"changed":true}'
    make_snapshot(snapshot, marker, entries)
    with pytest.raises(m.Blocked, match="Immutable"):
        m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert (output / "remote/papermill_v3.log").read_bytes() == b"original-log"
    assert (output / "remote" / artifact).read_bytes() == original
    assert (output / "artifacts" / Path(artifact).name).read_bytes() == original


def test_live_notebook_outputs_are_content_addressed_not_overwritten(m, tmp_path):
    snapshot, output, marker = tmp_path / "snapshot.zip", tmp_path / "output", {"run_id": "fixture"}
    name = "notebooks/03_JFINAL.out.fixture.ipynb"
    for raw in (b"raw-one", b"raw-two"):
        entries = {name: raw, "job_status_v3.json": json.dumps(dict(run_id="fixture", done=False)).encode()}
        make_snapshot(snapshot, marker, entries)
        m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert sorted(p.read_bytes() for p in (output / "raw_notebooks").rglob("*.ipynb")) == [b"raw-one", b"raw-two"]
    assert not (output / "artifacts").exists()


def test_nested_uncompressed_collection_limit_before_mutation(m, tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("large.jsonl", b"x" * 10000)
    marker = {"run_id": "fixture"}
    make_snapshot(snapshot, marker, {"results/v3/JFINAL_run_final.zip": buffer.getvalue()})
    monkeypatch.setattr(m, "MAX_COLLECTION_BYTES", 1000)
    with pytest.raises(m.Blocked, match="Nested.*expansion"):
        m.preserve_snapshot(snapshot, tmp_path / "output", marker, m.digest(snapshot))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["collection", "canonical"])
def test_symlink_destinations_fail_before_mutation(m, tmp_path, kind):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "root"
    target = root / "results/v3/pilot_verified" if kind == "canonical" else root / "output"
    target.parent.mkdir(parents=True)
    try:
        target.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit creating test symlinks")
    with pytest.raises(m.Blocked, match="Symlink"):
        if kind == "canonical":
            m.canonical_pilot(root)
        else:
            snapshot = tmp_path / "snapshot.zip"
            make_snapshot(snapshot, {"run_id": "fixture"}, {"papermill_v3.log": b"fixture"})
            m.preserve_snapshot(snapshot, target, {"run_id": "fixture"}, m.digest(snapshot))
    assert not list(external.iterdir())


def test_worker_own_unallocated_intent_is_remaining_not_an_ambiguous_prior_launch(m, tmp_path):
    item = m.job("JFINAL", "test")
    out = tmp_path / "results/v3" / item["key"]
    state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"),
                 job=item, run_id="synthetic-own-intent", status="submission_intent", startup_ack=False, allocation_attempted=False)
    write_json(out / "status.json", state)
    with pytest.raises(m.Blocked, match="requires reconciliation"):
        m.remaining_confirmatory(tmp_path)
    assert len(m.remaining_confirmatory(tmp_path, current_intent=state)) == 7
    assert item in m.remaining_confirmatory(tmp_path, current_intent=state)
    with pytest.raises(m.Blocked):
        m.remaining_confirmatory(tmp_path, current_intent={**state, "run_id": "not-this-worker"})


def test_resume_observes_same_remote_work_and_lease_without_new_allocation(operator, monkeypatch):
    s = operator
    s.op.allocate()
    deadline = s.op.deadline
    run_id = s.op.state["run_id"]
    s.op.marker = {"run_id": run_id}
    s.op.state.update(remote_marker=s.op.marker, remote_startup_ack=True)
    monkeypatch.setattr(s.op, "allocate", forbidden)
    monkeypatch.setattr(s.op, "submit", forbidden)
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: dict(run_id=run_id, known=True, alive=False, execution=dict(done=True, rc=0)))
    monkeypatch.setattr(s.op, "guard_healthy", lambda: True)
    monkeypatch.setattr(s.op, "collect", lambda **k: None)
    verified = []
    monkeypatch.setattr(s.op, "verify", lambda: verified.append(run_id))
    s.clock["now"] += 200
    assert s.op.run(resume=True) == 0
    assert s.op.deadline == deadline and s.op.state["run_id"] == run_id and verified == [run_id]
    assert sum(isinstance(c[0], list) and c[0][0] == "new" for c in s.backend.calls) == 1
    entry = s.m.read_json(s.op.state["budget_ledger"])["jobs"][run_id]
    assert entry["release_verified"] and entry["actual_seconds"] == 200


def test_legacy_hash_ledger_import_and_explicit_reconciliation_unblock_stale_spend(operator, monkeypatch):
    s = operator
    s.op.allocate()
    s.clock["now"] += 100
    assert s.op.release()
    s.op.finish_budget(True)
    s.op.save("completed", released=True, verified=True, completed_execution=True)
    path = Path(s.op.state["budget_ledger"])
    old_path = path.parent / ("budget_" + "a" * 64 + ".json")
    old_path.write_bytes(path.read_bytes())
    path.unlink()
    s.clock["now"] += s.m.MAX_GPU_SECONDS + 1
    assert s.m.budget_spent(s.m.budget_ledger(s.op.root)) > s.m.MAX_GPU_SECONDS
    monkeypatch.setattr(s.m, "Backend", lambda *a: s.backend)
    monkeypatch.setattr(s.m, "host_gate", lambda *a: {"synthetic_host": True})
    report = s.m.reconcile_budget_only(s.op.root, "synthetic-ack-input")
    assert report["charged_or_reserved_seconds"] == 100 and report["live_leases"] == 0
    assert not report["allocation_permitted"]
    assert old_path.is_file() and path.is_file()
    assert sum(isinstance(c[0], list) and c[0][0] == "new" for c in s.backend.calls) == 1


def test_parent_budget_documents_are_not_misread_as_legacy_ledgers(m, tmp_path):
    write_json(tmp_path / "results/v3/budget_approval.json", {"synthetic_parent_document": True})
    write_json(tmp_path / "results/v3/budget_initial.json", {"synthetic_parent_document": True})
    assert m.budget_ledger(tmp_path)["jobs"] == {}


def test_full_remaining_eta_rechecked_inside_reservation_lock(operator, monkeypatch):
    s = operator
    s.op.item = s.m.job("JFINAL", "test")
    s.op.state["job"] = s.op.item
    s.op.state["evidence"]["eta_seconds"] = 1000
    approval, report = s.op.root / "synthetic-approval.json", s.op.root / "synthetic-report.json"
    write_json(approval, dict(experiment_id=s.m.EXPERIMENT_ID, max_gpu_hours=25))
    write_json(report, pilot_fixture(s.m))
    s.op.state["options"].update(approval=str(approval), pilot_report=str(report))
    write_json(s.op.root / s.m.BUDGET_LEDGER, spent_fixture(s.m, s.m.MAX_GPU_SECONDS - 5000))
    monkeypatch.setattr(s.m, "remaining_confirmatory", lambda *a: s.m.plan("test") + s.m.plan("latency"))
    monkeypatch.setattr(s.m, "estimate_jobs", lambda *a: {"remaining": 8000})
    with pytest.raises(s.m.Blocked, match="no longer fits"):
        s.op.reserve_budget()
    assert s.op.state["run_id"] not in s.m.read_json(s.op.root / s.m.BUDGET_LEDGER)["jobs"]
    assert not s.backend.calls


def test_pilot_technical_go_and_budget_ready_are_separate_and_report_resumes_immutably(m, tmp_path, monkeypatch):
    import pilot_decision_v3 as pilot
    report = {**pilot_fixture(m), "note": "Synthetic technical-only fixture", "estimates": {}}
    monkeypatch.setattr(pilot, "decide", lambda *a, **k: json.loads(json.dumps(report)))
    monkeypatch.setattr(m, "canonical_pilot", lambda root: tmp_path / "synthetic-canonical")
    monkeypatch.setattr(m, "remaining_confirmatory", lambda *a: m.plan("test") + m.plan("latency"))
    monkeypatch.setattr(m, "estimate_jobs", lambda *a: {"remaining": m.MAX_GPU_SECONDS + 1})
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    coordinator = m.Coordinator(tmp_path, "full")
    assert coordinator.decide_pilot()["go"] is True
    assert coordinator.state["technical_go"] and not coordinator.state["budget_ready"]
    path = Path(coordinator.state["pilot_report"])
    pinned = path.read_bytes()
    assert json.loads(pinned)["meta"]["generated_utc"]
    assert coordinator.decide_pilot()["go"] is True
    assert path.read_bytes() == pinned and coordinator.state["pilot_report_sha256"] == m.digest(path)
    assert len(list(path.parent.glob("decision_*.json"))) == 1


@pytest.mark.parametrize("complete", [True, False])
def test_full_coordinator_resume_requires_all_real_status_slots_and_verification(m, tmp_path, monkeypatch, complete):
    # Synthetic orchestration fixture: every published slot is explicit; no blank completion.
    jobs = m.plan("full")
    assert len(jobs) == 21
    for item in jobs:
        out = tmp_path / "results/v3" / item["key"]
        state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"), job=item,
            run_id="synthetic-" + item["condition"] + "-" + item["phase"], deadline_epoch=m.time.time() + m.JOB_SECONDS,
            status="completed", startup_ack=True, verified=True, released=True, completed_execution=True,
            owned_endpoint="already-released-" + item["key"], evidence={"frozen_files": {}}, options={})
        if not complete and item["key"] == "test/JFINAL":
            state.update(status="failed", verified=False, completed_execution=False)
        write_json(out / "status.json", state)
    verification_calls = []
    def verify(operator):
        verification_calls.append(operator.item["key"])
        write_json(operator.out / "verification.json", {"ok": True, "synthetic_fixture_only": True})
    monkeypatch.setattr(m.Operator, "verify", verify)
    monkeypatch.setattr(m, "preflight", lambda root, item, **k: {"synthetic": item["key"]})
    monkeypatch.setattr(m, "host_gate", lambda *a, **k: {"synthetic": True})
    monkeypatch.setattr(m, "start_operator", forbidden)
    monkeypatch.setattr(m, "flock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(m, "Backend", lambda *a: SimpleNamespace(identity=lambda: dict(assignments=[])))
    pilot_path = tmp_path / "synthetic-pinned-pilot.json"
    write_json(pilot_path, pilot_fixture(m))
    def decide(coordinator):
        coordinator.save("pilot_go", technical_go=True, budget_ready=True, pilot_report_sha256=m.digest(pilot_path))
        return pilot_fixture(m)
    monkeypatch.setattr(m.Coordinator, "decide_pilot", decide)
    monkeypatch.setattr(m, "approval_gate", lambda *a, **k: ({"max_gpu_hours": 25}, pilot_fixture(m)))
    monkeypatch.setattr(m, "estimate_jobs", lambda *a: {})
    coordinator = m.Coordinator(tmp_path, "full", pilot_report=str(pilot_path), approval="synthetic-supplied", authorize_confirmatory=True)
    assert coordinator.run() == (0 if complete else 1)
    assert (coordinator.state["outcome"] == "completed") is complete
    if complete:
        assert set(coordinator.state["verified_jobs"]) == {item["key"] for item in jobs}
        assert set(verification_calls) == {item["key"] for item in jobs}


def test_live_notebook_without_terminal_status_is_not_frozen(m, tmp_path):
    snapshot, output, marker = tmp_path / "snapshot.zip", tmp_path / "output", {"run_id": "fixture"}
    name = "notebooks/03_JFINAL.out.fixture.ipynb"
    for raw in (b"live-one", b"live-two"):
        with zipfile.ZipFile(snapshot, "w") as z:
            z.writestr(name, raw)
            z.writestr("COLLECTION_MANIFEST.json", json.dumps(dict(marker=marker, files={name: hashlib.sha256(raw).hexdigest()})))
        m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    assert not (output / "artifacts").exists()
    assert sorted(p.read_bytes() for p in (output / "raw_notebooks").rglob("*.ipynb")) == [b"live-one", b"live-two"]


def test_collection_transaction_retains_complete_snapshot_if_mirror_write_fails(m, tmp_path, monkeypatch):
    snapshot, output, marker = tmp_path / "snapshot.zip", tmp_path / "output", {"run_id": "fixture"}
    make_snapshot(snapshot, marker, {"papermill_v3.log": b"synthetic-complete-transaction"})
    original = Path.write_bytes
    def fail(path, content):
        if path.name == "papermill_v3.log.pending":
            raise OSError("Synthetic mirror disk failure")
        return original(path, content)
    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError):
        m.preserve_snapshot(snapshot, output, marker, m.digest(snapshot))
    commit = m.read_json(output / "collection_commit.json")
    immutable = output / commit["snapshot"]
    assert m.digest(immutable) == commit["sha256"] == m.digest(snapshot)
    monkeypatch.setattr(Path, "write_bytes", original)
    m.preserve_snapshot(immutable, output, marker, commit["sha256"])
    assert (output / "remote/papermill_v3.log").read_bytes() == b"synthetic-complete-transaction"


@pytest.mark.parametrize("damage", [None, "initial-binding", "before-pilot", "over-cap"])
def test_post_pilot_preflight_binds_parent_initial_consent_and_actual_report(m, bound_consent, monkeypatch, damage):
    # ABI-only synthetic root; bypass packaging/review solely in this unit fixture.
    if damage:
        from test_postpilot_operator_v3 import amended
        bound_consent, _, _, _ = amended.__wrapped__(m, bound_consent)
    root = bound_consent.root
    cfg = m.config_gate(root)
    initial = initial_fixture(m, root, cfg)
    report = m.read_json(bound_consent.report)
    report["meta"] = {"generated_utc": "2020-01-01T00:00:00+00:00"}
    write_json(bound_consent.report, report)
    approved = m.read_json(bound_consent.approval)
    approved.update(experiment_id=m.EXPERIMENT_ID, pilot_report_sha256=m.digest(bound_consent.report),
        initial_authorization_sha256=hashlib.sha256(json.dumps(m.read_json(initial), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    if damage:
        approved["operational_amendment"]["pilot_report_sha256"] = m.digest(bound_consent.report)
    if damage == "initial-binding":
        approved["initial_authorization_sha256"] = "0" * 64
    if damage == "before-pilot":
        approved["approved_utc"] = "2019-01-01T00:00:00+00:00"
    if damage == "over-cap":
        approved["max_gpu_hours"] = 26
    write_json(bound_consent.approval, approved)
    item = m.job("JFINAL", "test")
    monkeypatch.setattr(m, "dataset_gate", lambda root: {"synthetic_fixture_only": True})
    monkeypatch.setattr(m, "estimate_jobs", lambda root, jobs, *a: {j["key"]: 1000 for j in jobs})
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    if damage:
        message = {"initial-binding": "bind initial", "before-pilot": "after the actual immutable pilot", "over-cap": "25-hour ceiling"}[damage]
        with pytest.raises(m.Blocked, match=message):
            m.preflight(root, item, initial_budget=initial, approval=bound_consent.approval,
                        pilot_report=bound_consent.report, authorize_confirmatory=True, fresh_pilot=False)
        return
    with pytest.raises(m.Blocked, match="approved operational amendment") as failure:
        m.preflight(root, item, initial_budget=initial, approval=bound_consent.approval,
                    pilot_report=bound_consent.report, authorize_confirmatory=True, fresh_pilot=False)
    assert failure.value.state == "blocked_budget_approval"
    assert not (root / m.BUDGET_LEDGER).exists()  # Preflight creates no consent or allocation ledger.


@pytest.fixture
def fake_scheduler(m, tmp_path, monkeypatch):
    """Detached workers advance only on this module's fake sleep, never inline."""
    clock = SimpleNamespace(now=1000.0, tick=0)
    active, starts, verified, probes, preflights, events = {}, [], [], [], [], []
    foreign = []
    controls = SimpleNamespace(uncertain=False, fail_key=None, preflight_failure=None, peak=0)
    report_path = tmp_path / "synthetic-pilot.json"
    write_json(report_path, pilot_fixture(m))
    pairs = [("G_SINGLE", "G_GREEDY"), ("JFINAL", "B13"), ("B13_GREEDY", "Q9_GREEDY")]

    def assignments():
        return foreign + [dict(endpoint=s["owned_endpoint"], accelerator=s["job"]["gpu"])
                          for s, _ in active.values()]

    class Backend:
        call = staticmethod(forbidden)

        def __init__(self, output, session):
            self.output, self.session = Path(output), session

        def identity(self, **kwargs):
            if controls.uncertain:
                raise m.Blocked("blocked_backend", "Synthetic uncertain assignment inventory")
            rows = assignments()
            if self.output.name == "capacity_probe":
                probes.append(list(rows))
                return dict(local_endpoint=None, assignments=rows)
            state = m.read_json(self.output / "status.json")
            return dict(local_endpoint=state["owned_endpoint"] if state["job"]["key"] in active else None,
                        assignments=rows)

    def persist(item, status="running"):
        out = tmp_path / "results/v3" / item["key"]
        done = status == "completed"
        state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"),
                     job=item, run_id="synthetic-" + item["key"], status=status, startup_ack=True,
                     owned_endpoint="synthetic-endpoint-" + item["key"], budget_reserved_seconds=100,
                     deadline_epoch=clock.now + m.JOB_SECONDS, deadline_monotonic=clock.now + m.JOB_SECONDS,
                     verified=done, released=done, completed_execution=done,
                     records=item["expected_rows"] if done else 0, evidence={"frozen_files": {}}, options={})
        write_json(out / "status.json", state)
        if status == "running":
            active[item["key"]] = (state, clock.tick + 2)
        return state

    def sleep(seconds):
        clock.now += seconds
        clock.tick += 1
        assert clock.tick < 100, "Scheduler did not finish within synthetic ticks"
        for key, (state, due) in list(active.items()):
            if clock.tick < due:
                continue
            failed = key == controls.fail_key
            state.update(status="failed" if failed else "completed", released=True,
                         verified=not failed, completed_execution=not failed,
                         records=state["job"]["expected_rows"] if not failed else 0)
            write_json(Path(state["output"]) / "status.json", state)
            del active[key]
            events.append(("released", key))

    def barrier(phase):
        for item in m.plan(phase):
            state = m.status_gate(tmp_path, tmp_path / "results/v3" / item["key"], item)
            assert state["status"] == "completed" and state["verified"] and state["released"]
            assert state["records"] == item["expected_rows"]
            assert item["key"] in verified and item["key"] not in active

    def analysis_inputs(root, *args, **kwargs):
        barrier("test")
        barrier("latency")
        paths = [tmp_path / "results/v3" / j["key"] / "synthetic_final.zip"
                 for phase in ("test", "latency") for j in m.plan(phase)]
        assert len(paths) == 7 and all(zipfile.is_zipfile(p) for p in paths)
        events.append(("analysis-seven-zips", "analysis/ANALYSIS"))
        return paths

    def preflight(root, item, **kwargs):
        assert Path(root) == tmp_path
        initial_jobs = [j for j in coordinator.jobs if j["phase"] in {"smoke", "pilot"}]
        if len(preflights) < len(initial_jobs):
            saved = m.read_json(coordinator.path)
            if not preflights:
                controls.initial_submissions = json.loads(json.dumps(coordinator.state["submissions"]))
            assert controls.locked
            assert saved["outcome"] == coordinator.state["outcome"] == "validating_local"
            assert saved["validation_job"] == item["key"]
            assert saved["validation_completed"] == preflights
            assert saved["validation_started_utc"]
            if preflights:
                assert saved["validation_job_done_utc"]
            assert saved["submissions"] == controls.initial_submissions and not starts and not probes
            assert not saved.get("budget_ready") and not saved.get("technical_go")
        preflights.append(item["key"])
        if item["key"] == controls.preflight_failure:
            raise m.Blocked("blocked_integrity", "Synthetic stage preflight failure")
        if item["phase"] == "analysis":
            analysis_inputs(root)
        return dict(synthetic_fixture_only=True, key=item["key"], frozen_files={})

    def verify(operator):
        state = m.read_json(operator.out / "status.json")
        assert state["released"] and state["completed_execution"] and state["verified"]
        assert state["records"] == operator.item["expected_rows"]
        assert state["owned_endpoint"] not in {r["endpoint"] for r in assignments()}
        write_json(operator.out / "verification.json", dict(ok=True, synthetic_fixture_only=True,
                   runs={operator.item["condition"]: dict(records=state["records"])}))
        with zipfile.ZipFile(operator.out / "synthetic_final.zip", "w") as archive:
            archive.writestr("manifest.json", json.dumps(dict(synthetic_fixture_only=True, records=state["records"])))
        verified.append(operator.item["key"])
        events.append(("verified", operator.item["key"]))

    def start(root, output, item, options):
        saved = m.read_json(tmp_path / "results/v3" / ("coordinator_" + coordinator.phase + "_state.json"))
        expected = []
        for phase in ("smoke", "pilot", "test", "latency", "analysis"):
            if not any(j["phase"] == phase for j in coordinator.jobs):
                continue
            expected.extend([[phase + "/" + c for c in pair] for pair in pairs]
                            if phase in {"smoke", "pilot", "test"} else [[phase + "/" + ("LATENCY" if phase == "latency" else "ANALYSIS")]])
            if phase == "smoke":
                expected.append(["smoke/LATENCY"])
        assert saved["batch_plan"] == expected
        assert saved["submissions"][item["key"]]["status"] == "intent"
        assert all(j["key"] in preflights for j in coordinator.jobs if j["phase"] == item["phase"])
        if item["phase"] == "pilot" and coordinator.phase in {"full", "prepared"}:
            barrier("smoke")
        if item["phase"] == "test" and coordinator.phase == "full":
            barrier("pilot")
            assert coordinator.state["technical_go"] and ("approval", "full") in events
        if item["phase"] == "latency" and coordinator.phase == "full":
            barrier("test")
        if item["phase"] == "analysis":
            analysis_inputs(root)
        if foreign and starts:
            assert starts[-1] in verified and not active
        assert len(assignments()) < 2
        starts.append(item["key"])
        events.append(("start", item["key"]))
        state = persist(item)
        controls.peak = max(controls.peak, len(active))
        return state

    def decide(c):
        barrier("pilot")
        c.save("pilot_go", technical_go=True, budget_ready=True, pilot_report=str(report_path),
               pilot_report_sha256=m.digest(report_path))
        events.append(("technical-go", "pilot"))
        return pilot_fixture(m)

    def approval(root, phase, *args, **kwargs):
        if phase == "full":
            barrier("pilot")
            assert ("technical-go", "pilot") in events
        events.append(("approval", phase))
        return {"max_gpu_hours": 25, "synthetic_fixture_only": True}, pilot_fixture(m)

    monkeypatch.setattr(m, "time", SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(m, "Backend", Backend)
    monkeypatch.setattr(m.Operator, "run", forbidden)
    monkeypatch.setattr(m.Operator, "verify", verify)
    monkeypatch.setattr(m, "start_operator", start)
    monkeypatch.setattr(m, "preflight", preflight)
    monkeypatch.setattr(m, "config_gate", lambda *a: {"synthetic_fixture_only": True})
    def host_gate(*args, **kwargs):
        saved = m.read_json(coordinator.path)
        assert controls.locked and saved["outcome"] == "validating_local"
        assert saved["validation_completed"] == preflights
        assert saved["validation_job_done_utc"]
        assert not starts and not probes and saved["submissions"] == controls.initial_submissions
        return {"host_lease_id": "synthetic-host"}
    monkeypatch.setattr(m, "host_gate", host_gate)
    monkeypatch.setattr(m, "approval_gate", approval)
    monkeypatch.setattr(m.Coordinator, "decide_pilot", decide)
    monkeypatch.setattr(m, "estimate_jobs", lambda root, jobs, *a: {j["key"]: 100 for j in jobs})
    monkeypatch.setattr(m, "analysis_inputs", analysis_inputs)
    class CoordinatorLock:
        def __enter__(self):
            assert not getattr(controls, "locked", False)
            controls.locked = True
        def __exit__(self, *args):
            controls.locked = False
    monkeypatch.setattr(m, "flock", lambda *a, **k: CoordinatorLock())
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.subprocess, "run", forbidden)
    monkeypatch.setattr(m, "bounded_call", forbidden)
    coordinator = m.Coordinator(tmp_path, "full", pilot_report=str(report_path), approval="synthetic-only",
                                initial_budget="synthetic-only", authorize_initial=True, authorize_confirmatory=True)
    return SimpleNamespace(c=coordinator, controls=controls, foreign=foreign, starts=starts, active=active,
                           verified=verified, probes=probes, preflights=preflights, events=events, persist=persist)


@pytest.mark.parametrize("foreign_count", [0, 1])
def test_fake_scheduler_full_pairs_capacity_and_stage_barriers(m, fake_scheduler, foreign_count):
    s = fake_scheduler
    s.foreign.extend(dict(endpoint="foreign-" + str(i), accelerator="unknown") for i in range(foreign_count))
    assert s.c.run() == 0
    assert s.starts == [j["key"] for j in m.plan("full")]
    assert s.controls.peak == 2 - foreign_count
    assert len(s.verified) == len(set(s.verified)) == 21 and not s.active
    assert all(len(rows) <= 2 for rows in s.probes)
    assert len(s.foreign) == foreign_count
    assert s.c.state["outcome"] == "completed"
    assert ("analysis-seven-zips", "analysis/ANALYSIS") in s.events


@pytest.mark.parametrize("capacity", ["two-unknown", "uncertain"])
def test_fake_scheduler_unknown_capacity_blocks_before_submission_intent(fake_scheduler, capacity):
    s = fake_scheduler
    if capacity == "two-unknown":
        s.foreign.extend([dict(endpoint="unknown-1"), dict(endpoint="unknown-2")])
    else:
        s.controls.uncertain = True
    assert s.c.run() == 1
    assert not s.starts and not s.c.state["submissions"] and not s.active
    assert s.c.state["outcome"] == ("blocked_capacity" if capacity == "two-unknown" else "blocked_backend")


def test_fake_scheduler_whole_stage_preflight_failure_launches_nothing(fake_scheduler):
    s = fake_scheduler
    s.controls.preflight_failure = "smoke/LATENCY"
    assert s.c.run() == 1
    assert not s.starts and not s.probes and not s.c.state["submissions"]
    assert s.c.state["outcome"] == "blocked_integrity"
    assert s.c.state["validation_completed"] == s.preflights[:-1]
    assert s.c.state["validation_job"] == "smoke/LATENCY"


def test_fake_scheduler_failed_worker_closes_pilot_barrier(fake_scheduler):
    s = fake_scheduler
    s.controls.fail_key = "smoke/G_SINGLE"
    assert s.c.run() == 1
    assert s.starts == ["smoke/G_SINGLE", "smoke/G_GREEDY"]
    assert not s.verified and s.c.state["outcome"] == "blocked_execution"
    assert ("technical-go", "pilot") not in s.events


@pytest.mark.parametrize("existing", ["running", "failed", "uncertain"])
def test_fake_scheduler_paused_resume_never_restarts_existing_slots(m, fake_scheduler, existing):
    s = fake_scheduler
    # A paused coordinator has one completed slot and one acknowledged detached worker.
    first, second = m.plan("full")[:2]
    s.persist(first, "completed")
    state = s.persist(second, "failed" if existing == "failed" else "running")
    if existing == "uncertain":
        state["startup_ack"] = False
        write_json(Path(state["output"]) / "status.json", state)
    s.c.state["submissions"] = {j["key"]: dict(status="acknowledged") for j in (first, second)}
    s.c.save("paused")
    assert s.c.run() == (0 if existing == "running" else 1)
    assert first["key"] not in s.starts and second["key"] not in s.starts
    if existing == "running":
        assert s.starts == [j["key"] for j in m.plan("full")[2:]]
        assert len(s.verified) == 21 and s.c.state["outcome"] == "completed"
    else:
        assert not s.starts
        assert s.c.state["outcome"] == ("blocked_startup" if existing == "uncertain" else "blocked_execution")


@pytest.fixture
def collection_observer(operator, monkeypatch):
    """Resume synthetic owned work without allocation, submission or external tools."""
    s = operator
    s.backend.endpoint = "owned-a100"
    s.backend.rows.append(dict(endpoint="owned-a100", accelerator="A100"))
    s.op.state["before_endpoints"] = ["unknown-t4"]
    s.op.record_owner(s.backend.identity())
    s.op.marker = {"run_id": s.op.state["run_id"]}
    s.op.state.update(remote_marker=s.op.marker, remote_startup_ack=True,
                      clock_boot_id=s.m.BOOT_ID, deadline_monotonic=s.op.monotonic_deadline)
    monkeypatch.setattr(s.op, "allocate", forbidden)
    monkeypatch.setattr(s.op, "submit", forbidden)
    monkeypatch.setattr(s.backend, "call", forbidden)
    monkeypatch.setattr(s.m, "bounded_call", forbidden)
    monkeypatch.setattr(s.m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(s.m.subprocess, "run", forbidden)
    monkeypatch.setattr(s.m, "host_gate", forbidden)
    monkeypatch.setattr(s.op, "observe_budget", lambda: None)
    monkeypatch.setattr(s.op, "finish_budget", lambda released: None)
    monkeypatch.setattr(s.op, "release", lambda: True)
    monkeypatch.setattr(s.op, "guard_healthy", lambda: True)
    return s


@pytest.mark.parametrize("failure,operation", [("timeout", "download"), ("transport", "remote_snapshot")])
def test_collection_periodic_degrades_then_reobserves_without_resubmit(collection_observer, monkeypatch, failure, operation):
    s = collection_observer
    sequence, verifications = [], []
    def health(code, name, **kwargs):
        assert name == "health"
        sequence.append(name)
        return dict(run_id=s.op.state["run_id"], known=True, alive=len(sequence) == 1,
                    execution=dict(done=True, rc=0))
    def collect(**kwargs):
        sequence.append("collect")
        s.op.state["collection_operation"] = operation
        if len(sequence) == 2:
            if failure == "timeout":
                raise s.m.subprocess.TimeoutExpired("synthetic-download", 37)
            raise s.m.Blocked("blocked_transport", "Synthetic lost snapshot acknowledgement")
        assert sequence == ["health", "collect", "health", "collect"]
    monkeypatch.setattr(s.op, "remote", health)
    monkeypatch.setattr(s.op, "collect", collect)
    monkeypatch.setattr(s.op, "verify", lambda: verifications.append(True))
    assert s.op.run(resume=True) == 0
    assert sequence == ["health", "collect", "health", "collect"] and verifications == [True]
    events = [json.loads(line) for line in (s.out / "events.jsonl").read_text().splitlines()]
    degraded = [e for e in events if e["status"] == "collection_degraded"]
    assert len(degraded) == 1
    assert {k: degraded[0][k] for k in ("collection_stage", "collection_attempt", "collection_operation",
            "collection_error_type", "collection_error_state", "collection_timeout_seconds")} == dict(
        collection_stage="periodic", collection_attempt=1, collection_operation=operation,
        collection_error_type="TimeoutExpired" if failure == "timeout" else "Blocked",
        collection_error_state=None if failure == "timeout" else "blocked_transport",
        collection_timeout_seconds=37 if failure == "timeout" else None)
    assert s.op.state["collection_degraded"] is False and s.clock["now"] == 1000
    assert not any(isinstance(c[0], list) for c in s.backend.calls)


@pytest.mark.parametrize("remaining", [120, 15])
def test_collection_final_three_attempts_bounded_wait_deadline_no_cleanup_fourth(collection_observer, monkeypatch, remaining):
    s = collection_observer
    s.op.work_deadline = s.op.monotonic_work_deadline = s.clock["now"] + remaining
    s.op.deadline = s.op.monotonic_deadline = s.op.work_deadline + s.m.CLEANUP_SECONDS
    pulls, health_checks = [], []
    def health(code, name, **kwargs):
        assert name == "health"
        health_checks.append(s.clock["now"])
        return dict(run_id=s.op.state["run_id"], known=True, alive=False, execution=dict(done=True, rc=0))
    def collect(*, cleanup=False):
        assert cleanup, "Final collection must be able to use original cleanup allowance"
        pulls.append((s.clock["now"], min(120, s.op.remaining(s.m.JOB_SECONDS, True) - 180)))
        s.op.state["collection_operation"] = "download"
        s.clock["now"] += pulls[-1][1]
        raise s.m.subprocess.TimeoutExpired("synthetic-download", pulls[-1][1])
    monkeypatch.setattr(s.op, "remote", health)
    monkeypatch.setattr(s.op, "collect", collect)
    monkeypatch.setattr(s.op, "verify", forbidden)
    assert s.op.run(resume=True) == 1
    assert pulls == [(1000, 120), (1150, 120), (1300, 120)]
    assert health_checks == [1000, 1150, 1300]
    assert s.clock["now"] <= s.op.deadline - 180
    assert s.op.state["final_collection_attempted"] is True
    assert s.op.state["collection_stage"] == "final" and s.op.state["collection_operation"] == "download"
    assert s.op.state["collection_attempt"] == 3
    assert s.op.state["collection_timeout_seconds"] == 120
    assert s.op.state["collection_error_type"] == "TimeoutExpired"
    assert s.op.state["error_type"] == "TimeoutExpired"
    assert s.op.state["verified"] is False and s.op.state["released"] is True


@pytest.mark.parametrize("state", ["blocked_integrity", "blocked_ownership"])
def test_collection_conflicts_fatal_without_retry_or_verify(collection_observer, monkeypatch, state):
    s = collection_observer
    pulls = []
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: dict(run_id=s.op.state["run_id"], known=True,
                        alive=False, execution=dict(done=True, rc=0)))
    def collect(*, cleanup=False):
        pulls.append(cleanup)
        raise s.m.Blocked(state, "Synthetic collection identity conflict")
    monkeypatch.setattr(s.op, "collect", collect)
    monkeypatch.setattr(s.op, "verify", forbidden)
    assert s.op.run(resume=True) == 1
    assert pulls == [True] and s.clock["now"] == 1000
    assert s.op.state["verified"] is False and s.op.state["error_type"] == "Blocked"
    assert "collection_degraded" not in s.op.state
    events = [json.loads(line) for line in (s.out / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["status"] == state]) == 1


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_collection_failed_download_preserves_prior_committed_snapshot(collection_observer, monkeypatch, failure):
    s = collection_observer
    # The operator clock is pre-1980; ZIP member timestamps need a separate clock.
    monkeypatch.setattr(zipfile, "time", SimpleNamespace(time=lambda: 0, localtime=lambda *a: (2000, 1, 1, 0, 0, 0)))
    snapshot = s.op.root / "synthetic-prior-snapshot.zip"
    make_snapshot(snapshot, s.op.marker, {"papermill_v3.log": b"committed-original"})
    s.m.preserve_snapshot(snapshot, s.out, s.op.marker, s.m.digest(snapshot))
    committed = {p.relative_to(s.out): p.read_bytes() for p in s.out.rglob("*") if p.is_file()}
    monkeypatch.setattr(s.op, "remote", lambda *a, **k: dict(snapshot=dict(
        path="/content/jev_v3_collect_" + s.op.state["run_id"] + ".zip", sha256="0" * 64)))
    def download(args, *, timeout):
        assert args[:3] == ["download", "-s", s.op.item["session"]]
        Path(args[-1]).write_bytes(b"synthetic-partial-download")
        if failure == "timeout":
            raise s.m.subprocess.TimeoutExpired("synthetic-download", timeout)
        return 1, ""
    monkeypatch.setattr(s.backend, "call", download)
    assert s.op.collect_with_recovery() is False
    for relative, content in committed.items():
        if relative not in {Path("status.json"), Path("execution_handover.json"), Path("events.jsonl")}:
            assert (s.out / relative).read_bytes() == content
    commit = s.m.read_json(s.out / "collection_commit.json")
    assert s.m.digest(s.out / commit["snapshot"]) == commit["sha256"] == s.m.digest(snapshot)
    assert s.op.state["collection_operation"] == "download" and s.op.state["collection_stage"] == "periodic"
    assert s.op.state["collection_timeout_seconds"] == (60 if failure == "timeout" else None)


def test_smoke_attempt_namespace_preserves_canonical_params_and_non_smoke_jobs(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    canonical = m.Coordinator(tmp_path, "full", authorize_initial=True)
    attempt = m.Coordinator(tmp_path, "full", authorize_initial=True, attempt_prefix="infra2")
    assert attempt.path == canonical.path.with_name("coordinator_full_infra2_state.json")
    assert attempt.state["submissions"] == canonical.state["submissions"] == {}
    for original, retried in zip(canonical.jobs, attempt.jobs):
        assert retried["params"] == original["params"]
        assert retried == {**original, "session": original["session"] + ("-infra2" if original["phase"] == "smoke" else "")}
        assert attempt.job_options(retried)["attempt_prefix"] == ("infra2" if original["phase"] == "smoke" else None)
        expected = (attempt.out / "attempts/infra2" / original["key"] if original["phase"] == "smoke"
                    else canonical.job_directory(original))
        assert attempt.job_directory(retried) == expected
        assert (attempt.job_directory(retried) / "sessions.json" != canonical.job_directory(original) / "sessions.json") == (original["phase"] == "smoke")
    for phase in ("pilot", "test", "latency", "analysis"):
        with pytest.raises(m.Blocked) as error:
            m.Coordinator(tmp_path, phase, authorize_initial=True, attempt_prefix="infra2")
        assert error.value.state == "blocked_user_authorization"


@pytest.mark.parametrize("damage,expected", [(None, None), ("unclosed-release", "blocked_ownership"),
    ("unverified-ledger", "blocked_budget_or_eta"), ("quality-execution", "blocked_execution"),
    ("no-infra-failure", "blocked_execution")])
def test_smoke_retry_gate_requires_closed_verified_infrastructure_lineage(m, tmp_path, monkeypatch, damage, expected):
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    item = m.job("JFINAL", "smoke")
    out = tmp_path / "results/v3" / item["key"]
    state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"),
        job=item, run_id="synthetic-prior", status="failed", released=True, allocation_attempted=True,
        completed_execution=False, remote_startup_ack=True, error_type="TimeoutExpired",
        health=dict(known=True, alive=True, run_id="synthetic-prior"))
    ledger = spent_fixture(m, 10)
    entry = ledger["jobs"][state["run_id"]]
    entry.update(key=item["key"], session=item["session"], output=str(out), release_evidence="backend_absent")
    if damage == "unclosed-release":
        state["released"] = False
    elif damage == "unverified-ledger":
        entry.update(released=False, release_verified=False)
    elif damage == "quality-execution":
        state["completed_execution"] = True
    elif damage == "no-infra-failure":
        state.update(status="completed", completed_execution=True)
    write_json(out / "status.json", state)
    write_json(tmp_path / m.BUDGET_LEDGER, ledger)
    before_status, before_ledger = (out / "status.json").read_bytes(), (tmp_path / m.BUDGET_LEDGER).read_bytes()
    if expected:
        with pytest.raises(m.Blocked) as error:
            m.smoke_retry_gate(tmp_path, "infra2")
        assert error.value.state == expected
    else:
        assert m.smoke_retry_gate(tmp_path, "infra2") == dict(attempt_prefix="infra2", development_only=True,
            predecessors={item["key"]: m.digest(out / "status.json")})
    assert (out / "status.json").read_bytes() == before_status
    assert (tmp_path / m.BUDGET_LEDGER).read_bytes() == before_ledger
    assert not (tmp_path / "results/v3/attempts").exists()


@pytest.mark.parametrize("damage", [None, "subsequent-executed", "unrelated-evalue", "final-archive", "rc0", "wrong-health-run-id"])
def test_smoke_retry_gate_requires_proven_pre_model_parameter_failure(m, tmp_path, monkeypatch, damage):
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    monkeypatch.setattr(m, "host_gate", forbidden)
    item = m.job("JFINAL", "smoke")
    out = tmp_path / "results/v3" / item["key"]
    run_id = "synthetic-prior"
    message = "Explicit dev smoke requires RUN_TAG=smoke and SEEDS=17"
    state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"),
        job=item, run_id=run_id, status="failed", released=True, allocation_attempted=True,
        completed_execution=False, remote_startup_ack=True, error_type="RuntimeError",
        health=dict(known=True, alive=False, run_id=run_id, execution=dict(done=True, rc=1)))
    notebook = dict(metadata=dict(papermill=dict(parameters=dict(SPLIT="dev", RUN_TAG="smoke", SEEDS=17))),
        cells=[dict(cell_type="code", source=[f'raise ValueError("{message}")'], execution_count=1,
                    outputs=[dict(output_type="error", ename="ValueError", evalue=message)]),
               dict(cell_type="code", source=["# Synthetic model stage; never executed."], execution_count=None, outputs=[])])
    ledger = spent_fixture(m, 10)
    ledger["jobs"][run_id].update(key=item["key"], session=item["session"], output=str(out), release_evidence="backend_absent")
    if damage == "subsequent-executed":
        notebook["cells"][-1]["execution_count"] = 2
    elif damage == "unrelated-evalue":
        notebook["cells"][0]["outputs"][0]["evalue"] = "Unrelated synthetic failure"
    elif damage == "final-archive":
        archive = out / "artifacts/JFINAL_synthetic_final.zip"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(nested_zip())
    elif damage == "rc0":
        state["health"]["execution"]["rc"] = 0
    elif damage == "wrong-health-run-id":
        state["health"]["run_id"] = "synthetic-other"
    notebook_path = out / "remote/notebooks" / (m.NOTEBOOKS[item["condition"]] + ".out." + run_id + ".ipynb")
    write_json(notebook_path, notebook)
    write_json(out / "status.json", state)
    write_json(tmp_path / m.BUDGET_LEDGER, ledger)
    paths = [out / "status.json", notebook_path, tmp_path / m.BUDGET_LEDGER]
    before = [p.read_bytes() for p in paths]
    if damage:
        with pytest.raises(m.Blocked) as error:
            m.smoke_retry_gate(tmp_path, "infra2")
        assert error.value.state == "blocked_execution"
    else:
        assert m.smoke_retry_gate(tmp_path, "infra2") == dict(attempt_prefix="infra2", development_only=True,
            predecessors={item["key"]: m.digest(out / "status.json")})
    assert [p.read_bytes() for p in paths] == before
    assert not (tmp_path / "results/v3/attempts").exists()


def test_collection_final_health_timeout_uses_remaining_attempts(collection_observer, monkeypatch):
    s = collection_observer
    pulls, health_checks = [], []
    def health(code, name, **kwargs):
        health_checks.append(name)
        if len(health_checks) == 2:
            raise s.m.subprocess.TimeoutExpired("synthetic-health", 60)
        return dict(run_id=s.op.state["run_id"], known=True, alive=False, execution=dict(done=True, rc=0))
    def collect(**kwargs):
        pulls.append(s.clock["now"])
        if len(pulls) == 1:
            raise s.m.subprocess.TimeoutExpired("synthetic-snapshot", 60)
    monkeypatch.setattr(s.op, "remote", health)
    monkeypatch.setattr(s.op, "collect", collect)
    monkeypatch.setattr(s.op, "verify", lambda: None)
    assert s.op.run(resume=True) == 0
    assert pulls == [1000, 1060] and health_checks == ["health"] * 3
    assert s.op.state["collection_degraded"] is False


@pytest.mark.parametrize("damage", ["quality", "integrity", "success", "live"])
def test_smoke_new_prefix_cannot_bypass_previous_attempt_failure(m, tmp_path, monkeypatch, damage):
    monkeypatch.setattr(m.Backend, "call", forbidden)
    monkeypatch.setattr(m.Backend, "identity", forbidden)
    canonical = m.job("G_SINGLE", "smoke")
    jobs = []
    ledger = dict(experiment_id=m.EXPERIMENT_ID, max_gpu_seconds=m.MAX_GPU_SECONDS, jobs={})
    for prefix in (None, "infra02"):
        item = canonical if prefix is None else m.job("G_SINGLE", "smoke", canonical["session"] + "-" + prefix)
        base = tmp_path / "results/v3"
        out = (base / "attempts" / prefix if prefix else base) / item["key"]
        state = dict(series="v3", root=str(tmp_path), output=str(out), cli_config=str(out / "sessions.json"),
                     job=item, run_id="synthetic-" + str(prefix), status="failed", released=True, allocation_attempted=True,
                     completed_execution=False, remote_startup_ack=True, error_type="TimeoutExpired",
                     health=dict(known=True, alive=True, run_id="synthetic-" + str(prefix)))
        if prefix:
            state.update(error_type="Blocked" if damage == "integrity" else "RuntimeError",
                         completed_execution=damage in {"quality", "success"},
                         status="completed" if damage == "success" else "running" if damage == "live" else "failed",
                         released=damage != "live")
        write_json(out / "status.json", state)
        jobs.append(out / "status.json")
        ledger["jobs"][state["run_id"]] = dict(key=item["key"], session=item["session"], output=str(out),
            reserved_seconds=120, reserved_epoch=1000, released=True, release_verified=True, actual_seconds=10,
            release_evidence="backend_absent")
    write_json(tmp_path / m.BUDGET_LEDGER, ledger)
    before = [p.read_bytes() for p in jobs]
    with pytest.raises(m.Blocked):
        m.smoke_retry_gate(tmp_path, "infra03")
    assert [p.read_bytes() for p in jobs] == before


@pytest.mark.parametrize("state", ["blocked_ownership", "blocked_integrity"])
def test_remote_collection_conflict_is_not_classified_as_transport(operator, monkeypatch, state):
    s = operator
    monkeypatch.setattr(s.backend, "call", lambda *a, **k: (1, 'V3_JSON::' + json.dumps(dict(blocked=state))))
    with pytest.raises(s.m.Blocked) as error:
        s.op.remote("synthetic remote script", "collect")
    assert error.value.state == state
