"""CPU-only protocol fixture; exercise real v3 algorithms with fake engine requests."""

import importlib.util
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from jevlab.algorithms import Ctx
from jevlab.common import append_jsonl, config_hash, read_json, read_jsonl, sha256_file, write_json
from jevlab.engines import ReqResult


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class FakeEngine:
    def __init__(self, role, calls):
        self.role, self.calls, self.counter = role, calls, 0
        self.engine_kwargs = {"enable_prefix_caching": False}

    def new_id(self, prefix):
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def run(self, requests, deadline):
        self.calls.append((self.role, requests))
        now = time.perf_counter()
        return {r.rid: ReqResult(r.rid, len(r.prompt_ids),
                token_ids=list(map(ord, "FINAL: 1\n")), finish_reason="stop",
                t_add=now, t_first=now, t_end=now) for r in requests}


class FakeSelector:
    def decide(self, problem, prefix, candidates, criterion, permutation, deadline):
        return {"status": "ok", "winner_branch": 0, "input_tokens": 10}


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    import jevlab.v3.algorithms as algorithms

    data = tmp_path / "data"
    data.mkdir()
    test = [{"id": f"t{i:03}", "problem": f"Compute {i}+1", "language": "en"}
            for i in range(500)]
    dev = [{"id": f"d{i:02}", "problem": f"Dev {i}", "language": "en"} for i in range(20)]
    for name, rows in (("test", test), ("latency", test[:100]), ("dev", dev), ("pilot", test[100:150])):
        for row in rows:
            append_jsonl(data / f"{name}_inputs.jsonl", row)
    write_json(data / "schedule.json", {"order": [r["id"] for r in reversed(test)]})
    domains = ["arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"]
    write_json(data / "latency_plan.json", {"dataset_version": "v3", "sealed": True,
        "source_split": "test", "split": "test", "quota_axis": "source_sampling_tier",
        "seeds": [17, 29, 43], "repetitions": 3, "conditions": ["JFINAL", "B13_GREEDY"],
        "generation": 0, "expected_rows": 1800, "quota_per_domain": {"easy": 6, "medium": 8, "hard": 6},
        "intrinsic_difficulty_counts": {"easy": 100, "medium": 0, "hard": 0}, "items": [
        {"item_id": row["id"], "problem": row["problem"], "domain": domains[i // 20],
         "difficulty": "easy", "sampling_tier": "easy" if i % 20 < 6 else "medium" if i % 20 < 14 else "hard"}
        for i, row in enumerate(test[:100])]})
    manifest = {"dataset_version": "v3", "sealed": True,
                "quota_axis": "source_sampling_tier",
                "actual_n": {"test": 500, "pilot": 50, "dev": 20},
                "review": {"agent_reviewed_all": True, "human_reviewed": False,
                           "policy_id": "blind-v3", "policy_sha256": "a" * 64},
                "input_sha256": {p.name: sha256_file(p) for p in data.iterdir()}}
    write_json(data / "dataset_manifest.json", manifest)
    state = types.SimpleNamespace(calls=[], lifecycle=[], gpu="GPU-A", close_fail=False,
                                  post_fail=False, loads=0, checks=[], cases=[])

    class Experiment:
        def __init__(self, **params):
            self.p = {"WARMUP_N": 1, "TIMEOUT_S": 600, "CHECKPOINT_EVERY": 10000, **params}
            self.seeds = [int(s) for s in self.p["SEEDS"].split(",")]
            self.root, self.data_dir = tmp_path, data
            self.dir = tmp_path / "results"
            self.dir.mkdir(exist_ok=True)
            self.cfg = {"protocol_version": "3", "code_version": "0.3.0"}
            self.cond, self.chash, self.run_name = "LATENCY", "frozen-config", "latency"
            self.f = {k: str(self.dir / f"{k}.jsonl") for k in
                      ("candidates", "decisions", "rounds", "metrics", "predictions", "proposals", "warmup")}
            self.engines, self.preflight_results = {}, {}
            self.sampler = None
            self.gpu_uuid = "GPU-A"
            self.manifest = {}
            self.phase_epoch, self.phase_id, self.phase_cleanup_confirmed = 0, None, False
            self.paths = {r: f"verified/{r}" for r in self.roles()}
            self.lock = {r: {"ok": True} for r in self.roles()}
            self.event = lambda *args, **kwargs: None

        def stage(self, name):
            return nullcontext()

        def checkpoint(self, final=False):
            pass

        def save_manifest(self):
            pass

        def _validate_dataset(self):
            self.dataset_seals = read_json(data / "dataset_manifest.json")["input_sha256"]

        def _check_authorization(self):
            # This fixture replaces the entire runner; real consent is tested separately.
            pass

        def _validate_identity(self, *args, **kwargs):
            pass

        def _assert_engines_usable(self):
            pass

        def _assert_unpublished(self):
            pass

        def _phase_metadata(self):
            return {"phase_epoch": getattr(self, "_pending_phase_epoch", self.phase_epoch),
                    "phase_id": self.phase_id, "gpu_uuid": self.gpu_uuid,
                    "resident_roles": list(self.engines), "cleanup_confirmed": self.phase_cleanup_confirmed,
                    "outside_T_total": True}

        def _begin_phase(self):
            import uuid
            self.phase_id = str(uuid.uuid4())
            self._pending_phase_epoch = self.phase_epoch + 1

        def _completed_keys(self):
            keys = [r["resume_key"] for r in read_jsonl(self.f["predictions"])]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate prediction completion markers")
            return set(keys)

        def _orphan_keys(self, done):
            return {r["resume_key"] for name, path in self.f.items() if name not in ("predictions", "warmup")
                    for r in read_jsonl(path) if r.get("resume_key")} - done

        def _publish_final(self):
            return self.checkpoint(final=True)

        def dev_problems(self):
            return read_jsonl(data / "dev_inputs.jsonl")

        def assert_same_gpu(self):
            if state.gpu != "GPU-A":
                raise RuntimeError("GPU UUID changed")

        def _provenance(self, condition):
            return {"gpu_uuid": "GPU-A", "condition": condition, "protocol_version": "3"}

        def _case_context(self):
            import uuid
            frame = {"sampling": dict(self.ctx.sampling), "limits": dict(self.ctx.limits),
                     "system_prompt": self.ctx.system_prompt, "crit_final": self.ctx.crit_final}
            return {"context_frame": frame, "context_frame_id": config_hash(frame), "execution_id": str(uuid.uuid4())}

        def _close_roles(self, roles):
            state.lifecycle.append(("close", tuple(roles)))
            if state.close_fail:
                raise RuntimeError("engine processes still alive")
            for r in roles:
                del self.engines[r]
            return {r: {"cleanup_confirmed": True, "pids": [1], "exitcodes": [0]} for r in roles}

        def _load_roles(self, roles):
            assert not self.engines, "overlapping resident phases"
            assert self.ctx is None and self.selector is None
            state.loads += 1
            state.lifecycle.append(("load", tuple(roles)))
            self.engines = {r: FakeEngine(r, state.calls) for r in roles}
            self.toks = {r: Tokenizer() for r in roles}
            self.eos, self.nl = {r: [0] for r in roles if r != "J"}, {r: {10} for r in roles if r != "J"}
            self._frozen_termination = {r: {"eos_ids": [0], "newline_ids": [10]} for r in self.eos}
            self.selector = FakeSelector() if "J" in roles else None

        def _ctx(self, condition):
            role = "B" if condition.startswith("B13") else "G"
            prompts = getattr(self, "prompts", {})
            return Ctx(gen=self.engines[role], tok=self.toks[role], system_prompt=prompts.get("generador", ""),
                       sampling=dict(temperature=.7, top_p=.9, top_k=0, repetition_penalty=1.,
                                     presence_penalty=0., frequency_penalty=0.),
                       limits=dict(n_candidates=4, max_output_tokens=self.p.get("MAX_OUTPUT_TOKENS", 1024), gb_context=4096,
                                   hybrid_candidate_budget=self.p.get("CANDIDATE_BUDGET", 4096)), eos_ids=[0], newline_ids={10},
                       selector=self.selector, profile=self.cfg.get("implementation_profile", "V3"),
                       crit_step=prompts.get("criterio_paso", ""), crit_final=prompts.get("criterio_final", ""))

        def _check_blocking(self, res):
            if any(v.get("blocking") and not v.get("ok") for v in res.values()):
                raise RuntimeError("blocking preflight")

        def _proposal_callback(self, key, extra=None):
            return lambda row: append_jsonl(self.f["proposals"], {**row, **extra, "resume_key": key})

        def _persist(self, out, key, wall_start, extra=None, completion_callback=None):
            common = {"resume_key": key, **extra}
            for name in ("candidates", "decisions", "rounds"):
                for row in getattr(out, name):
                    append_jsonl(self.f[name], {**row, **common})
            append_jsonl(self.f["metrics"], {**out.run, **common})
            if completion_callback:
                completion_callback({**out.run, **common})
            append_jsonl(self.f["predictions"], {**out.run, **common})

    runner = types.ModuleType("jevlab.v3.runner")
    runner.Experiment = Experiment
    monkeypatch.setitem(sys.modules, "jevlab.v3.runner", runner)
    preflight = types.ModuleType("jevlab.v3.preflight")
    preflight.pre_engine_checks = lambda exp, mode: {
        "selector_native_reference": {"ok": True, "blocking": True}}

    def post(exp, mode):
        assert mode == "full" and exp.exp.cond == "LATENCY"
        assert {"ctx", "p", "engines"} <= vars(exp).keys()
        assert exp.ctx.gen.role == ("B" if exp.cond.startswith("B13") else "G")
        state.checks.append((state.loads, exp.cond))
        return {k: {"ok": not state.post_fail, "blocking": True, **exp._phase_metadata()} for k in
                ("continuity", "stress", "selector_equivalence")}

    preflight.post_engine_checks = post
    monkeypatch.setitem(sys.modules, "jevlab.v3.preflight", preflight)
    import jevlab.v3
    monkeypatch.setattr(jevlab.v3, "preflight", preflight, raising=False)
    monkeypatch.setattr(algorithms, "generator_prompt_ids", lambda *args: ([1], "prompt"))
    monkeypatch.setattr(algorithms, "sampling_params", lambda **kwargs: types.SimpleNamespace(**kwargs))
    real_run = algorithms.run_case

    def observed(case, ctx, timeout_s, **kwargs):
        state.cases.append(case)
        return real_run(case, ctx, timeout_s, **kwargs)

    monkeypatch.setattr(algorithms, "run_case", observed)
    path = Path(__file__).resolve().parents[1] / "src/jevlab/v3/latency_study.py"
    spec = importlib.util.spec_from_file_location("jevlab.v3._latency_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LatencyStudy, state, data


def smoke(cls, **kwargs):
    return cls(SPLIT="dev", LAT_N_ITEMS=2, LAT_REPETITIONS=1, SEEDS="17", **kwargs)


def test_confirmatory_actual_calls_traces_and_stable_seeds(protocol):
    cls, state, _ = protocol
    exp = cls()
    exp.start_engines()
    assert not exp.engines
    exp.preflight_before_engines()
    exp.run()
    measured = [c for c in state.cases if c.generation == 0]
    assert len(measured) == 1800
    warm = [c for c in state.cases if c.generation != 0]
    assert len(state.calls) - len(warm) == 1800
    assert len(state.checks) == state.loads == len(warm)
    assert all(c.problem_id.startswith("d") for c in warm)
    predictions = read_jsonl(exp.f["predictions"])
    metrics = read_jsonl(exp.f["latency_metrics"])
    assert len(predictions) == len(metrics) == 1800
    assert all(row["difficulty"] == "easy" and row["sampling_tier"] in ("easy", "medium", "hard")
               for row in predictions + metrics)
    assert all(row["context_frame_id"] == config_hash(row["context_frame"]) for row in predictions + metrics)
    assert len({r["resume_key"] for r in predictions}) == 1800
    assert all(r["output_sha256"] and r["gpu_uuid"] == "GPU-A" for r in predictions)
    assert sum(r["condition"] == "B13_GREEDY" for r in predictions) == 900
    for repetition in range(3):
        for block in range(10):
            first = next(r for r in predictions if r["repetition"] == repetition and r["block"] == block)
            assert first["phase"] == ("B" if (block + repetition) % 2 == 0 else "H")
    proposals = read_jsonl(exp.f["proposals"])
    assert len(proposals) == 900
    assert all(r["status"] == "complete" and len(r["candidates"]) == 4 for r in proposals)
    assert len(read_jsonl(exp.f["decisions"])) == len(read_jsonl(exp.f["rounds"])) == 900
    for name in ("proposals", "candidates", "decisions", "rounds"):
        assert all({"item_id", "seed", "repetition", "phase", "block", "phase_epoch",
                    "case_index", "gpu_uuid"} <= r.keys() for r in read_jsonl(exp.f[name]))
    for branch in range(4):
        rows = [c for r in proposals if r["item_id"] == "t099" and r["seed"] == 17
                for c in r["candidates"] if c["branch"] == branch]
        assert len(rows) == 3 and len({r["seed_branch"] for r in rows}) == 1
    greedy = [reqs for role, reqs in state.calls if role == "B"]
    assert len(greedy) >= 900 and all(reqs[0].params.seed is None for reqs in greedy)
    swaps = read_jsonl(exp.f["swaps"])
    assert all(r["outside_T_total"] and r["cleanup_confirmed"] for r in swaps)
    assert all(tuple(r["resident_roles"]) in (("B",), ("G", "J")) for r in swaps)
    by_phase = {r["phase_id"]: r for r in swaps}
    assert len(by_phase) == len(swaps)
    assert [r["phase_epoch"] for r in swaps] == list(range(1, len(swaps) + 1))
    for row in predictions:
        swap = by_phase[row["phase_id"]]
        assert row["phase_epoch"] == swap["phase_epoch"]
        assert row["resident_roles"] == swap["resident_roles"] and row["cleanup_confirmed"] is True
        assert row["t_start_perf"] >= swap["t_end_perf"]
        assert row["T_total"] == row["t_end_perf"] - row["t_start_perf"]
        assert all(check["phase_id"] == row["phase_id"] and check["phase_epoch"] == row["phase_epoch"]
                   for check in swap["postflight"].values())
    baselines = [r for r in predictions if r["condition"] == "B13_GREEDY"]
    assert len({r["reference_id"] for r in baselines}) == 100
    assert len({r["execution_id"] for r in baselines}) == 900
    before = len(state.calls)
    exp.run()
    assert len(state.calls) == before  # Only final prediction markers permit skipping.
    exp.finalize()
    assert exp.manifest["coverage_complete"] is True
    assert exp.manifest["expected_coverage"]["n_predictions"] == 1800


def test_dev_smoke_reads_only_v3_dev(protocol):
    cls, state, data = protocol
    (data / "latency_inputs.jsonl").unlink()
    (data / "test_inputs.jsonl").unlink()
    exp = smoke(cls)
    exp.run()
    assert len(read_jsonl(exp.f["predictions"])) == 4
    assert all(c.problem_id in ("d00", "d01") for c in state.cases)


@pytest.mark.parametrize("damage", ["missing", "changed", "extra", "schema", "duplicate", "schedule"])
def test_frozen_subset_rejected(protocol, damage):
    cls, _, data = protocol
    path = data / "latency_inputs.jsonl"
    rows = read_jsonl(path)
    if damage == "missing":
        path.unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            cls()
        return
    if damage == "changed":
        rows[0]["problem"] = "not identical"
    elif damage == "extra":
        rows.append(read_jsonl(data / "test_inputs.jsonl")[100])
    elif damage == "schema":
        rows[0]["answer"] = "gold"
    elif damage == "duplicate":
        rows[1] = rows[0]
    else:
        write_json(data / "schedule.json", {"order": ["t000"] * 500})
    if damage != "schedule":
        path.unlink()
        for row in rows:
            append_jsonl(path, row)
    manifest = read_json(data / "dataset_manifest.json")
    # Even with an updated seal, subset/schema/schedule validation must fail.
    manifest["input_sha256"][path.name] = sha256_file(path)
    manifest["input_sha256"]["schedule.json"] = sha256_file(data / "schedule.json")
    write_json(data / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError):
        cls()


def test_changed_sha_rejected(protocol):
    cls, _, data = protocol
    append_jsonl(data / "latency_inputs.jsonl", {"id": "extra"})
    with pytest.raises(ValueError, match="SHA"):
        cls()


@pytest.mark.parametrize("damage", ["old_labels", "tier_quota", "tier_label", "difficulty_label", "domain",
                                     "extra", "intrinsic_counts", "axis", "source_split", "public_hash"])
def test_source_tier_plan_rejects_unbound_or_coerced_metadata(protocol, damage):
    cls, state, data = protocol
    plan = read_json(data / "latency_plan.json")
    if damage == "old_labels":
        for row in plan["items"]:
            row.pop("sampling_tier")
    elif damage == "tier_quota":
        plan["items"][0]["sampling_tier"] = "hard"
    elif damage == "tier_label":
        plan["items"][0]["sampling_tier"] = "source-hard"
    elif damage == "difficulty_label":
        plan["items"][0]["difficulty"] = "intrinsic-hard"
    elif damage == "domain":
        plan["items"][0]["domain"] = "algebra"
    elif damage == "extra":
        plan["items"][0]["answer"] = "forbidden"
    elif damage == "intrinsic_counts":
        plan["intrinsic_difficulty_counts"] = {"easy": 30, "medium": 40, "hard": 30}
    elif damage == "axis":
        plan["quota_axis"] = "intrinsic_difficulty"
    elif damage == "source_split":
        plan["source_split"] = "pilot"
    else:
        plan["items"][0]["difficulty"] = "medium"
    write_json(data / "latency_plan.json", plan)
    manifest = read_json(data / "dataset_manifest.json")
    if damage != "public_hash":
        manifest["input_sha256"]["latency_plan.json"] = sha256_file(data / "latency_plan.json")
        write_json(data / "dataset_manifest.json", manifest)
    with pytest.raises(ValueError):
        cls()
    assert state.loads == 0 and not state.cases


def test_source_tier_quota_never_relabels_intrinsic_difficulty(protocol):
    cls, _, data = protocol
    plan = read_json(data / "latency_plan.json")
    for row in plan["items"]:
        row["difficulty"] = "medium"
    plan["intrinsic_difficulty_counts"] = {"easy": 0, "medium": 100, "hard": 0}
    write_json(data / "latency_plan.json", plan)
    manifest = read_json(data / "dataset_manifest.json")
    manifest["input_sha256"]["latency_plan.json"] = sha256_file(data / "latency_plan.json")
    write_json(data / "dataset_manifest.json", manifest)
    exp = cls()
    rows = exp._frozen_latency_plan({row["id"]: row for row in exp.latency_items()})
    assert all(row["difficulty"] == "medium" for row in rows)
    assert sum(row["sampling_tier"] == "hard" for row in rows) == 30
    assert exp.manifest["expected_coverage"]["n_predictions"] == 1800


def test_close_failure_never_loads_next_phase(protocol):
    cls, state, _ = protocol
    exp = smoke(cls)
    exp._swap("H")
    state.close_fail = True
    with pytest.raises(RuntimeError, match="still alive"):
        exp._swap("B")
    assert state.loads == 1 and set(exp.engines) == {"G", "J"}
    assert exp.phase is None and exp.ctx is None and exp.selector is None
    assert read_jsonl(exp.f["swaps"])[-1]["ok"] is False
    assert exp.phase_epoch == 1
    assert read_jsonl(exp.f["swaps"])[-1]["phase_epoch"] == 1
    assert read_jsonl(exp.f["swaps"])[-1]["attempted_phase_epoch"] == 2


def test_uuid_guard_and_mandatory_postflight(protocol):
    cls, state, _ = protocol
    exp = smoke(cls)
    state.gpu = "GPU-B"
    with pytest.raises(RuntimeError, match="UUID"):
        exp._swap("B")
    assert state.loads == 0
    state.gpu, state.post_fail = "GPU-A", True
    with pytest.raises(RuntimeError, match="preflight"):
        exp.run()
    assert not read_jsonl(exp.f["predictions"])
    assert not state.cases
    assert exp.phase_epoch == 0
    assert read_jsonl(exp.f["swaps"])[-1]["phase_epoch"] == 0


def test_uuid_guard_before_each_measured_case(protocol, monkeypatch):
    cls, state, _ = protocol
    import jevlab.v3.algorithms as algorithms
    original = algorithms.run_case

    def change_uuid(case, *args, **kwargs):
        out = original(case, *args, **kwargs)
        if case.generation == 0:
            state.gpu = "GPU-B"
        return out

    monkeypatch.setattr(algorithms, "run_case", change_uuid)
    exp = smoke(cls)
    with pytest.raises(RuntimeError, match="UUID"):
        exp.run()
    assert len(read_jsonl(exp.f["predictions"])) == 1


def test_model_failures_stay_in_denominator(protocol, monkeypatch):
    cls, _, _ = protocol
    monkeypatch.setattr(FakeEngine, "run", lambda *args: (_ for _ in ()).throw(RuntimeError("model failure")))
    exp = smoke(cls)
    exp.run()
    assert len(read_jsonl(exp.f["latency_metrics"])) == 4
    assert len(read_jsonl(exp.f["predictions"])) == 4
    assert all(r["status"] == "engine_error" for r in read_jsonl(exp.f["predictions"]))


def test_real_runner_constructor_provenance_persistence_and_finalize(protocol, tmp_path, monkeypatch):
    fixture_cls, state, _ = protocol
    src = Path(__file__).resolve().parents[1] / "src/jevlab/v3"
    spec = importlib.util.spec_from_file_location("jevlab.v3._runner_integration_test", src / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setitem(sys.modules, "jevlab.v3.runner", runner)
    spec = importlib.util.spec_from_file_location("jevlab.v3._latency_integration_test", src / "latency_study.py")
    latency = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(latency)
    data = tmp_path / "data/v3"
    data.mkdir()
    for row in read_jsonl(tmp_path / "data/dev_inputs.jsonl"):
        append_jsonl(data / "dev_inputs.jsonl", row)
    config = tmp_path / "config"
    config.mkdir()
    write_json(config / "experiment_v3.json", {
        "protocol_version": "3", "code_version": "0.3.0", "jevk5_runtime": {},
        "implementation_profile": "1COPY-G4-VLLM-GRAPH-V3",
        "models": {r: {"repo": "Qwen/Qwen3.5-9B" if r == "O" else f"test/{r}",
                       "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a" if r == "O" else "b" * 40}
                   for r in ("G", "J", "B", "O")}})
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("generador", "criterio_paso", "criterio_final"):
        (prompts / f"{name}.txt").write_text("Protocol test prompt", encoding="ascii")
    exp = smoke(latency.LatencyStudy, ROOT=str(tmp_path), WARMUP_N=1)
    exp.gpu_uuid = "GPU-A"
    spec = importlib.util.spec_from_file_location("jevlab.v3._preflight_integration_test", src / "preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    import jevlab.v3
    monkeypatch.setattr(jevlab.v3, "preflight", preflight)
    for name in ("selector_equivalence", "batching_audit", "continuity_test", "eager_vs_graph", "trace_audit"):
        monkeypatch.setattr(preflight, name, lambda *args: {"ok": True, "blocking": True})

    def stress(view):
        assert view.p["TIMEOUT_S"] == 600 and view.ctx is not None
        assert view.prompts and view.selector is view.ctx.selector
        assert view.cond in ("JFINAL", "B13_GREEDY")
        return {"ok": True, "blocking": True}

    monkeypatch.setattr(preflight, "stress_test", stress)
    fixture_base = fixture_cls.__bases__[0]
    for name in ("assert_same_gpu", "_load_roles", "_close_roles", "_ctx"):
        monkeypatch.setattr(exp, name, types.MethodType(getattr(fixture_base, name), exp))
    exp.run()
    predictions = read_jsonl(exp.f["predictions"])
    assert len(predictions) == 4 and all(r["gpu_uuid"] == "GPU-A" for r in predictions)
    assert len(read_jsonl(exp.f["proposals"])) == 2
    assert len(read_jsonl(exp.f["decisions"])) == 2
    exp.finalize()
    assert exp.manifest["coverage_complete"] is True
    assert exp.manifest["missing_predictions"] == 0
    assert exp.manifest["expected_coverage"]["n_predictions"] == 4
    assert len(state.calls) == 6  # Four real run_case measurements plus two dev warmups.


def test_phase_resume_epoch_is_recovered_without_reusing_uuid(protocol):
    cls, state, _ = protocol
    exp = smoke(cls)
    exp._swap("B")
    previous = read_jsonl(exp.f["swaps"])[-1]
    exp._close_roles(list(exp.engines))
    resumed = smoke(cls)
    assert resumed.phase_epoch == 1
    resumed._swap("B")
    current = read_jsonl(resumed.f["swaps"])[-1]
    assert current["phase_epoch"] == 2 and current["phase_id"] != previous["phase_id"]


def test_latency_failed_postflight_never_claims_successful_epoch(protocol):
    cls, state, _ = protocol
    exp = smoke(cls)
    state.post_fail = True
    with pytest.raises(RuntimeError, match="preflight"):
        exp._swap("H")
    failed = read_jsonl(exp.f["swaps"])[-1]
    assert failed["phase_epoch"] == exp.phase_epoch == 0
    assert failed["attempted_phase_epoch"] == 1 and failed["phase_id"]
    assert not read_jsonl(exp.f["predictions"]) and not state.cases


@pytest.mark.parametrize("params", [{"LATENCY_SWAP_MODE": "sleep"}, {"LAT_CONDITIONS": "B13,JFINAL"},
    {"LAT_CONDITIONS": "JFINAL"}, {"LAT_CONDITIONS": "JFINAL,JFINAL"},
    {"LAT_CONDITIONS": "JFINAL,B13_GREEDY,G_GREEDY"}, {"SPLIT": "pilot"},
    {"LAT_N_ITEMS": 2}, {"LAT_REPETITIONS": 1}, {"SEEDS": "17"}, {"LAT_BLOCK_SIZE": 0}])
def test_forbidden_parameters(protocol, params):
    cls, state, _ = protocol
    with pytest.raises(ValueError):
        cls(**params)
    assert state.loads == 0
