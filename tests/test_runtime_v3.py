import datetime as dt
import json
import math
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import jevlab
from jevlab import engines, hw
from jevlab.engines import ReqResult
from jevlab.engines import Req
from jevlab.common import append_jsonl, config_hash, read_json, read_jsonl, sha256_file, sha256_text, write_json
from jevlab.v3 import __version__, preflight
from jevlab.v3 import algorithms
from jevlab.v3 import engines as v3_engines
from jevlab.v3.runner import CONDITIONS, DEFAULTS, OFFICIAL_Q9_REVISION, Experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    data = tmp_path / "data/v3"
    data.mkdir(parents=True)
    cfg = read_json(ROOT / "config/experiment_v3.json")
    cfg["dataset_review_policy"] = {"id": "blind-review-v3", "sha256": "f" * 64}
    (tmp_path / "config/experiment_v3.json").write_text(json.dumps(cfg))
    hashes, inputs = {}, {}
    for split, count in (("test", 500), ("pilot", 50), ("dev", 20)):
        rows = [{"id": f"{split}-{i:03d}", "problem": f"What is {i}+1?", "language": "en"}
                for i in range(count)]
        inputs[split] = rows
        path = data / f"{split}_inputs.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        hashes[path.name] = sha256_file(path)
        (data / f"{split}_gold.jsonl").write_text("secret gold: not even valid JSON\n")
    schedule = {"order": [r["id"] for r in reversed(inputs["test"])],
                "pilot_order": [r["id"] for r in reversed(inputs["pilot"])]}
    (data / "schedule.json").write_text(json.dumps(schedule))
    hashes["schedule.json"] = sha256_file(data / "schedule.json")
    manifest = {"dataset_version": "v3", "sealed": True,
                "quota_axis": "source_sampling_tier",
                "actual_n": {"test": 500, "pilot": 50, "dev": 20}, "input_sha256": hashes,
                "review": {"agent_reviewed_all": True, "human_reviewed": False,
                           "source_sampling_tier_is_intrinsic_difficulty": False,
                           "intrinsic_difficulty_counts": {"test": {"easy": 500, "medium": 0, "hard": 0},
                                                           "pilot": {"easy": 50, "medium": 0, "hard": 0}},
                           "policy_id": "blind-review-v3", "policy_sha256": "f" * 64}}
    (data / "dataset_manifest.json").write_text(json.dumps(manifest))
    (data / "SHA256SUMS").write_text("".join(
        f"{sha256_file(data / name)}  {name}\n" for name in (*hashes, "dataset_manifest.json")))
    return tmp_path


def experiment(root, **params):
    return Experiment(ROOT=str(root), **params)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_isolated_defaults_full_seed_counts_and_roles(workspace, condition):
    exp = experiment(workspace, CONDITION=condition)
    assert jevlab.__version__ == "0.2.1" and __version__ == "0.3.0"
    assert exp.config["code_version"] == "0.3.0"
    greedy = "GREEDY" in condition
    assert exp.seeds == ([17] if greedy else [17, 29, 43])
    assert exp.manifest["expected_coverage"]["n_predictions"] == (500 if greedy else 1500)
    assert len(exp.problems()) == 500
    assert exp.roles() == (["O"] if condition == "Q9_GREEDY" else ["B"] if condition.startswith("B13")
                           else ["G", "J"] if condition == "JFINAL" else ["G"])
    assert exp.p["MAX_OUTPUT_TOKENS"] == 2048 and exp.p["CANDIDATE_BUDGET"] == 8192
    assert exp.p["KV_CACHE_GB_O"] == exp.p["KV_CACHE_GB_G"] == 3
    assert exp.dir.is_relative_to(workspace / "results/v3")
    assert DEFAULTS["DATA_SUBDIR"] == "data/v3" and DEFAULTS["N_PROBLEMS"] == 500


def test_prefix_jobs_full_manifest_not_smoke_and_no_gold(workspace):
    exp = experiment(workspace, SPLIT="dev", N_PROBLEMS=1, SEEDS="17", RUN_TAG="smoke")
    assert [r["id"] for r in exp.problems()] == ["dev-000"]
    assert len(exp.manifest["expected_coverage"]["problem_ids"]) == 20
    assert exp.manifest["expected_coverage"]["n_predictions"] == 60
    original_hash = exp.chash
    (workspace / "data/v3/test_gold.jsonl").write_text("changed inaccessible gold")
    assert experiment(workspace, SPLIT="dev", N_PROBLEMS=20, SEEDS="17", RUN_TAG="smoke").chash == original_hash
    assert not any("gold" in name for name in exp.config["data_sha256"])
    with pytest.raises(ValueError, match="smoke"):
        experiment(workspace, N_PROBLEMS=1, RUN_TAG="smoke")
    exp.finalize()
    manifest = read_json(exp.dir / "manifest.json")
    assert manifest["coverage_complete"] is False and manifest["missing_predictions"] == 60


def test_dev_before_seal_only_uses_copied_twenty_inputs(workspace):
    data = workspace / "data/v3"
    for name in ("dataset_manifest.json", "test_inputs.jsonl", "pilot_inputs.jsonl", "schedule.json"):
        (data / name).unlink()
    exp = experiment(workspace, SPLIT="dev", RUN_TAG="smoke", N_PROBLEMS=1, SEEDS="17")
    assert len(exp.problems()) == 1 and len(exp.dev_problems()) == 20
    assert exp.config["data_sha256"].keys() == {"dev_inputs.jsonl"}
    with pytest.raises(ValueError, match="missing sealed"):
        experiment(workspace)


@pytest.mark.parametrize("field,value", [("protocol_version", "2"), ("protocol_version", "3.1"),
                                         ("protocol_version", None), ("code_version", "0.2.1")])
def test_unknown_protocol_and_version_fail_before_artifacts(workspace, field, value):
    path = workspace / "config/experiment_v3.json"
    cfg = read_json(path)
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="protocol_version|code_version"):
        experiment(workspace)
    assert not (workspace / "results").exists()


@pytest.mark.parametrize("field,value", [("sealed", False), ("dataset_version", "v2"),
                                         ("agent_reviewed_all", False), ("human_reviewed", True),
                                         ("policy_id", "wrong"), ("policy_sha256", "f" * 63)])
def test_unsealed_or_unreviewed_gate(workspace, field, value):
    path = workspace / "data/v3/dataset_manifest.json"
    manifest = read_json(path)
    target = manifest if field in ("sealed", "dataset_version") else manifest["review"]
    target[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        experiment(workspace)


def test_full_count_and_input_hash_gate_applies_to_prefix(workspace):
    path = workspace / "data/v3/test_inputs.jsonl"
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:499]))
    with pytest.raises(ValueError, match="500 inputs"):
        experiment(workspace)
    path.write_text("".join(json.dumps({**r, "problem": "different"}) + "\n" for r in rows))
    with pytest.raises(ValueError, match="SHA"):
        experiment(workspace)


def test_seal_required_and_manifest_integrity_checked(workspace):
    path = workspace / "data/v3/SHA256SUMS"
    seal = path.read_text()
    path.unlink()
    with pytest.raises(ValueError, match="SHA256SUMS"):
        experiment(workspace)
    path.write_text(seal)
    manifest_path = workspace / "data/v3/dataset_manifest.json"
    manifest = read_json(manifest_path)
    manifest["unbound_change"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.*seal mismatch"):
        experiment(workspace)


def test_full_prompt_gate_even_for_one_requested_problem(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="Q9_GREEDY")
    exp.toks = {"O": object()}
    monkeypatch.setattr(preflight, "generator_prompt_ids", lambda tok, system, problem:
                        ([1] * (1025 if problem == "What is 0+1?" else 1), "prompt"))
    result = preflight.prompt_length_check(exp)
    assert not result["ok"] and result["n_prompts"] == 500
    assert result["over"] == {"O:test-000": 1025}


def test_cache_check_requires_disabled_prefix_cache(workspace):
    exp = experiment(workspace)
    exp.engines = {"G": SimpleNamespace(engine_kwargs={"enable_prefix_caching": False}),
                   "J": SimpleNamespace(engine_kwargs={"enable_prefix_caching": False})}
    assert preflight.cache_check(exp)["ok"]
    exp.engines["J"].engine_kwargs["enable_prefix_caching"] = True
    result = preflight.cache_check(exp)
    assert result["blocking"] and not result["ok"]


def test_inputs_cannot_change_after_constructor_or_contain_gold(workspace):
    exp = experiment(workspace)
    path = workspace / "data/v3/test_inputs.jsonl"
    rows = read_jsonl(path)
    rows[0]["gold"] = 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="changed"):
        exp.problems()
    with pytest.raises(ValueError, match="no gold"):
        experiment(workspace)


@pytest.mark.parametrize("params", [{"SEEDS": "17,17"}, {"SEEDS": "17,99"}, {"SEEDS": ""},
                                   {"CONDITION": "J64"}, {"SPLIT": "old"}, {"N_CANDIDATES": 3},
                                   {"ENABLE_THINKING": True}, {"HASH_WEIGHTS": False},
                                   {"DATA_SUBDIR": "data/v2"}, {"DATA_SUBDIR": "../outside"},
                                   {"CONDITION": "G_GREEDY", "SEEDS": "29"}])
def test_invalid_settings(workspace, params):
    with pytest.raises(ValueError):
        experiment(workspace, **params)


def test_a10040_uuid_required_even_abort_false_and_required_any(workspace, monkeypatch):
    exp = experiment(workspace, ABORT_ON_PREFLIGHT_FAIL=False, REQUIRED_GPU="any")
    gpu = {"name": "NVIDIA A100-SXM4-40GB", "uuid": "GPU-one", "memory.total": "40960"}
    monkeypatch.setattr(hw, "gpu_info", lambda: [gpu])
    assert exp.assert_same_gpu()["uuid"] == "GPU-one"
    gpu["uuid"] = "GPU-two"
    with pytest.raises(RuntimeError, match="UUID changed"):
        exp.assert_same_gpu()
    exp.gpu_uuid = None
    gpu.update(name="NVIDIA L4", **{"memory.total": "24576"})
    with pytest.raises(RuntimeError, match="A100"):
        exp.assert_same_gpu()
    gpu.update(name="NVIDIA A100-80GB", **{"memory.total": "81920"})
    with pytest.raises(RuntimeError, match="A100"):
        exp.assert_same_gpu()


def test_real_o_role_one_sequence_and_prefix_cache_off(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="Q9_GREEDY")
    called = []

    class FakeEngine:
        def __init__(self, name, path, tok, **kwargs):
            called.append((name, path, kwargs))

    monkeypatch.setattr(v3_engines, "SafeEngine", FakeEngine)
    exp.paths, exp.toks, exp.lock = {"O": "official9"}, {"O": object()}, {"O": {"ok": True}}
    exp.eos, exp.nl = {"O": [0]}, {"O": {1}}
    exp._load_roles(exp.roles())
    assert called[0][0:2] == ("O", "official9")
    assert called[0][2]["max_num_seqs"] == 1
    assert called[0][2]["extra"] == {"enable_prefix_caching": False}
    exp.ctx = exp._ctx()
    assert exp.ctx.gen is exp.engines["O"] and exp.ctx.selector is None
    assert exp._provenance("Q9_GREEDY")["sampling"]["temperature"] == 0


@pytest.mark.parametrize("mode", ["off", "light", "full"])
def test_mandatory_checks_are_fresh_even_off(workspace, monkeypatch, mode):
    exp = experiment(workspace, PREFLIGHT_MODE=mode, ABORT_ON_PREFLIGHT_FAIL=False)
    called = []

    def check(name):
        return lambda *a: called.append(name) or {"ok": True, "blocking": True}

    for name in ("parser_selftest", "lock_check", "template_check", "tokenizer_check",
                 "prompt_length_check", "sampling_semantics_check", "native_reference",
                 "selector_equivalence", "batching_audit", "continuity_test", "stress_test", "cache_check",
                 "eager_vs_graph"):
        monkeypatch.setattr(preflight, name, check(name))
    res = preflight.pre_engine_checks(exp, mode)
    exp.preflight_results.update(res)
    exp.ctx = SimpleNamespace(selector=object())
    preflight.post_engine_checks(exp, mode)
    assert "tokenizer_check" in called and "template_check" in called and "native_reference" in called
    assert "selector_equivalence" in called and "stress_test" in called
    with pytest.raises(RuntimeError, match="blocking"):
        exp._check_blocking({"stress": {"ok": False, "blocking": True}})
    with pytest.raises(RuntimeError, match="blocking"):
        exp._check_blocking({"tokenizer": {"ok": True, "blocking": True, "skipped": True}})


def test_proposal_callback_appends_and_survives_selector_failure(workspace):
    exp = experiment(workspace)
    callback = exp._proposal_callback("case-key")
    for branch in range(4):
        callback({"branch": branch, "status": "final", "public_output": "FINAL: 1\n"})
    assert len(read_jsonl(exp.f["proposals"])) == 4
    assert all(r["resume_key"] == "case-key" for r in read_jsonl(exp.f["proposals"]))
    append_jsonl(exp.f["failures"], {"status": "selector_error"})
    assert len(read_jsonl(exp.f["proposals"])) == 4


def test_orphan_pool_cannot_be_regenerated_or_overwritten(workspace):
    exp = experiment(workspace, SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    key = exp._key(exp.problems()[0]["id"], 17)
    exp._proposal_callback(key)({"candidate_set_id": "durable", "status": "complete", "candidates": []})
    with pytest.raises(RuntimeError, match="uncommitted durable"):
        exp.run()
    assert len(read_jsonl(exp.f["proposals"])) == 1


@pytest.mark.parametrize("condition", CONDITIONS)
def test_runtime_actual_seed_calls_resume_and_traces(workspace, monkeypatch, condition):
    exp = experiment(workspace, CONDITION=condition, SPLIT="dev", N_PROBLEMS=2)
    requests_seen = []
    names = iter(range(100))

    def generate(requests, deadline):
        requests_seen.append(requests)
        return {r.rid: ReqResult(r.rid, len(r.prompt_ids), list(map(ord, "FINAL: 1\n")),
                                finish_reason="final_abort") for r in requests}

    gen = SimpleNamespace(new_id=lambda tag: f"{tag}-{next(names)}", run=generate)
    tok = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))
    exp.engines = {r: gen for r in exp.roles()}
    exp.toks = {r: tok for r in exp.roles()}
    exp.eos = {r: [0] for r in exp.roles() if r != "J"}
    exp.nl = {r: {10} for r in exp.roles() if r != "J"}
    exp.selector = SimpleNamespace(decide=lambda *args:
                                   {"status": "ok", "winner_branch": 0, "input_tokens": 10})
    exp.ctx = exp._ctx()
    monkeypatch.setattr(algorithms, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(algorithms, "generator_prompt_ids", lambda *a: ([1], "prompt"))
    monkeypatch.setattr(exp, "assert_same_gpu", lambda: None)
    monkeypatch.setattr(exp, "checkpoint", lambda *a, **kw: None)
    exp.run()
    n = 2 if "GREEDY" in condition else 6
    assert len(requests_seen) == len(read_jsonl(exp.f["predictions"])) == n
    assert {r["seed"] for r in read_jsonl(exp.f["predictions"])} == set(exp.seeds)
    assert len(read_jsonl(exp.f["candidates"])) == n * (4 if condition == "JFINAL" else 1)
    assert len(read_jsonl(exp.f["decisions"])) == (n if condition == "JFINAL" else 0)
    assert len(read_jsonl(exp.f["proposals"])) == (n if condition == "JFINAL" else 0)
    assert all(len(r["candidates"]) == 4 for r in read_jsonl(exp.f["proposals"]))
    if "GREEDY" in condition:
        assert all(r.params["seed"] is None and r.params["temperature"] == 0
                   for reqs in requests_seen for r in reqs)
        assert len({r["reference_id"] for r in read_jsonl(exp.f["predictions"])}) == 2
    exp.run()
    assert len(requests_seen) == n


@pytest.mark.parametrize("split,n", [("test", 1), ("test", 100), ("test", 501),
                                   ("pilot", 1), ("pilot", 500), ("dev", 21)])
def test_partial_quality_is_dev_only(workspace, split, n):
    with pytest.raises(ValueError, match="N_PROBLEMS"):
        experiment(workspace, SPLIT=split, N_PROBLEMS=n)
    assert experiment(workspace, SPLIT="pilot").p["N_PROBLEMS"] == 50


@pytest.mark.parametrize("key,value", [("MAX_OUTPUT_TOKENS", 1024), ("CANDIDATE_BUDGET", 4096),
    ("MAX_PROMPT_TOKENS", 512), ("GB_CONTEXT", 8192), ("J_MAX_INPUT", 8192), ("TIMEOUT_S", 300),
    ("TEMPERATURE", .5), ("TOP_P", .8), ("PRESENCE_PENALTY", 1), ("FREQUENCY_PENALTY", 1),
    ("KV_CACHE_GB_G", 2), ("ENFORCE_EAGER", True)])
def test_frozen_quality_parameters_cannot_be_overridden(workspace, key, value):
    with pytest.raises(ValueError, match="frozen protocol parameter"):
        experiment(workspace, **{key: value})
    exp = experiment(workspace)
    exp.p[key] = value
    with pytest.raises(ValueError, match="frozen protocol parameter"):
        exp._validate_configuration()


def test_exact_official_pin_and_config_tampering(workspace):
    path = workspace / "config/experiment_v3.json"
    cfg = read_json(path)
    cfg["models"]["O"]["revision"] = "c202236" + "a" * 33
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="exact pinned"):
        experiment(workspace)
    cfg["models"]["O"]["revision"] = OFFICIAL_Q9_REVISION
    path.write_text(json.dumps(cfg))
    exp = experiment(workspace)
    cfg["limits"]["max_output_tokens"] = 1024
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="configuration changed"):
        exp._validate_configuration()


@pytest.mark.parametrize("condition", ["G_SINGLE", "G_GREEDY", "JFINAL", "B13_GREEDY", "Q9_GREEDY"])
def test_synthetic_stress_reaches_real_rendered_limits(workspace, monkeypatch, condition):
    from jevlab import prompts

    exp = experiment(workspace, CONDITION=condition, SPLIT="dev")
    role = "O" if condition == "Q9_GREEDY" else "B" if condition.startswith("B13") else "G"
    calls, problems_seen = [], []
    counter = iter(range(20))

    def generate(requests):
        calls.append(requests)
        assert all(not r.watch_final and r.params["ignore_eos"] for r in requests)
        return {r.rid: ReqResult(r.rid, len(r.prompt_ids), [1] * r.params["max_tokens"]) for r in requests}

    def render(tok, system, problem):
        problems_seen.append(problem)
        assert "What is" not in problem
        n = 200 + problem.count("marker 12345 alpha beta.") * 11
        return [1] * n, "rendered-synthetic-" + str(n)

    tok = SimpleNamespace(pad_token_id=0)
    exp.engines = {role: SimpleNamespace(new_id=lambda tag: str(next(counter)), run=generate)}
    exp.toks = {role: tok, "J": object()}
    exp.eos, exp.nl = {role: [0]}, {role: {10}}
    exp.selector = (SimpleNamespace(letter_logits_from_ids=lambda ids, n: ([0, 1, 2, 3], .01))
                    if condition == "JFINAL" else None)
    exp.ctx = exp._ctx()
    exp.sampler = SimpleNamespace(now_used=lambda: 30 * 2**30, window_peak=lambda *a: 32 * 2**30)
    monkeypatch.setattr(exp, "dev_problems", lambda: (_ for _ in ()).throw(AssertionError("no source stress")))
    monkeypatch.setattr(preflight, "generator_prompt_ids", render)
    monkeypatch.setattr(prompts, "generator_prompt_ids", render)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "selector_prompt_ids", lambda tok, state, question:
                        ([1] * (100 + question["criteria"]["option_0"].count("\n") * 11), "selector"))
    result = preflight.stress_test(exp)
    assert result["ok"] and result["blocking"] and result["synthetic"]
    assert 960 <= result["rendered_prompt_tokens"] <= 1024
    n = 4 if condition == "JFINAL" else 1
    assert result["generator"]["role"] == role and result["generator"]["n"] == n
    assert result["generator"]["tokens"] == [2048] * n
    assert len(calls) == 1 and len(calls[0]) == n
    assert all(960 <= len(r.prompt_ids) <= 1024 for r in calls[0])
    assert result["nvml_peak_gib"] == 32 and result["limits"]["J_MAX_INPUT"] == 16384
    assert read_json(exp.dir / "preflight" / "stress.json") == result
    if n == 4:
        assert 16320 <= result["selector"]["input_tokens"] <= 16384


def test_impossible_synthetic_stress_is_blocking_without_inference(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    tok = object()
    gen = SimpleNamespace(run=lambda *a: pytest.fail("must not infer an undersized/oversized prompt"))
    exp.engines, exp.toks = {"G": gen}, {"G": tok}
    exp.ctx = SimpleNamespace(gen=gen, tok=tok, selector=object())
    monkeypatch.setattr(preflight, "generator_prompt_ids", lambda *a: ([1] * 1500, "too long"))
    result = preflight.stress_test(exp)
    assert not result["ok"] and result["blocking"]


@pytest.mark.parametrize("damage", ["short", "missing", "context", "selector"])
def test_full_stress_failures_never_pass_or_shrink_pressure(workspace, monkeypatch, damage):
    from jevlab import prompts

    exp = experiment(workspace, SPLIT="dev")
    names = iter(range(10))
    calls = []

    def generate(requests):
        calls.append(requests)
        return {r.rid: ReqResult(r.rid, len(r.prompt_ids), [1] * (2047 if damage == "short" else 2048))
                for r in (requests[:3] if damage == "missing" else requests)}

    tok, jtok = object(), object()
    gen = SimpleNamespace(new_id=lambda tag: str(next(names)), run=generate)
    exp.engines, exp.toks = {"G": gen}, {"G": tok, "J": jtok}
    exp.eos, exp.nl = {"G": [0]}, {"G": {10}}
    exp.selector = SimpleNamespace(letter_logits_from_ids=lambda ids, n:
                                   ([float("nan"), 1, 2, 3] if damage == "selector" else [0, 1, 2, 3], .01))
    exp.ctx = exp._ctx()
    if damage == "context":
        exp.p["GB_CONTEXT"] = 2500
    render = lambda tok, system, problem: ([1] * (200 + problem.count("marker 12345 alpha beta.") * 11), "synthetic")
    monkeypatch.setattr(preflight, "generator_prompt_ids", render)
    monkeypatch.setattr(prompts, "generator_prompt_ids", render)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "selector_prompt_ids", lambda tok, state, question:
                        ([1] * (100 + question["criteria"]["option_0"].count("\n") * 11), "selector"))
    result = preflight.stress_test(exp)
    assert not result["ok"] and result["blocking"]
    assert 960 <= result["rendered_prompt_tokens"] <= 1024
    if damage != "context":
        assert len(calls[0]) == 4 and all(r.params["max_tokens"] == 2048 for r in calls[0])
    else:
        assert not calls


@pytest.mark.parametrize("failure", ["close_graph", "close_eager", "construct_eager"])
def test_graph_lifecycle_failure_is_published_and_never_overlaps_weights(workspace, monkeypatch, failure):
    import jevlab.runner as legacy_runner

    exp = experiment(workspace, CONDITION="G_GREEDY", SPLIT="dev")
    tok = object()
    original = {"model": "verified-G", "tokenizer": "verified-G", "max_model_len": 4096,
                "max_num_seqs": 1, "gpu_memory_utilization": .05, "enforce_eager": False}
    run = lambda requests: {r.rid: SimpleNamespace(token_ids=[1] * 64) for r in requests}
    graph = SimpleNamespace(engine_kwargs=original, new_id=lambda tag: tag, run=run)
    active, loads = [graph], []

    class FakeEngine:
        def __init__(self, name, path, tok, **kw):
            assert not active
            loads.append(name)
            if failure == "construct_eager":
                raise RuntimeError("constructor exposes no cleanup handles")
            self.engine_kwargs = {**original, **kw["extra"], "enforce_eager": kw["enforce_eager"]}
            self.new_id, self.run = graph.new_id, run
            active.append(self)

    def close(engine, timeout):
        fail = failure == "close_graph" and engine is graph or failure == "close_eager" and engine is not graph
        if not fail:
            active.remove(engine)
        return {"cleanup_confirmed": not fail}

    exp.engines, exp.paths, exp.toks = {"G": graph}, {"G": "verified-G"}, {"G": tok}
    exp.eos, exp.nl = {"G": [0]}, {"G": {10}}
    exp.ctx = SimpleNamespace(gen=graph, tok=tok)
    monkeypatch.setattr(v3_engines, "SafeEngine", FakeEngine)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(preflight, "generator_prompt_ids", lambda *a: ([1], "prompt"))
    monkeypatch.setattr(legacy_runner, "shutdown_engine_checked", close)
    result = preflight.eager_vs_graph(exp)
    assert result["blocking"] and not result["ok"]
    assert "lifecycle failure" in result["summary"]
    assert "G" not in loads  # Never restore a graph after an unconfirmed eager close/constructor.
    assert len(active) <= 1
    if failure == "close_graph":
        assert not loads and exp.ctx.gen is graph
    else:
        assert exp.ctx.gen is None
    assert read_json(exp.dir / "preflight/eager_vs_graph.json") == result


@pytest.mark.parametrize("condition", ["G_SINGLE", "G_GREEDY", "JFINAL"])
@pytest.mark.parametrize("agree_tokens,n_rows", [(64, 3), (48, 3), (0, 20)])
def test_graph_diagnostic_restores_exact_one_copy_and_contexts(workspace, monkeypatch, condition, agree_tokens, n_rows):
    import jevlab.runner as legacy_runner

    exp = experiment(workspace, CONDITION=condition, SPLIT="dev")
    active, history = [], []

    class FakeEngine:
        def __init__(self, name, path, tok, **kwargs):
            assert not active
            active.append(self)
            self.n = 0
            self.engine_kwargs = {"model": path, "tokenizer": path, "enforce_eager": kwargs["enforce_eager"],
                "max_model_len": kwargs["max_model_len"], "max_num_seqs": kwargs["max_num_seqs"],
                "gpu_memory_utilization": kwargs["gpu_memory_utilization"], "enable_prefix_caching": False,
                "enable_sleep_mode": False, "dtype": "bfloat16", "max_num_batched_tokens": 4096}
            if kwargs.get("kv_cache_gb"):
                self.engine_kwargs["kv_cache_memory_bytes"] = int(kwargs["kv_cache_gb"] * 2**30)
            self.engine_kwargs.update(kwargs.get("extra", {}))
            history.append(dict(self.engine_kwargs))

        def new_id(self, tag):
            self.n += 1
            return f"{tag}-{self.n}"

        def run(self, requests):
            eager = self.engine_kwargs["enforce_eager"]
            tokens = [1] * 64 if not eager else [1] * agree_tokens + [2] * (64 - agree_tokens)
            return {r.rid: SimpleNamespace(token_ids=tokens) for r in requests}

    def close(engine, timeout):
        active.remove(engine)
        return {"cleanup_confirmed": True, "pids": [1], "exitcodes": [0]}

    tok = object()
    seqs = 4 if condition == "JFINAL" else 1
    graph = FakeEngine("G", "verified-G", tok, enforce_eager=False, max_model_len=4096,
                       max_num_seqs=seqs, kv_cache_gb=3, gpu_memory_utilization=.05)
    exp.engines, exp.paths, exp.toks = {"G": graph}, {"G": "verified-G"}, {"G": tok}
    exp.eos, exp.nl = {"G": [0]}, {"G": {10}}
    exp.ctx = SimpleNamespace(gen=graph, tok=tok)
    exp.ctx_H = SimpleNamespace(gen=graph)
    original = dict(graph.engine_kwargs)
    # Simulate the latency owned view while retaining the owner's context references.
    view = SimpleNamespace(**vars(exp))
    view.exp, view.dev_problems = exp, exp.dev_problems
    monkeypatch.setattr(v3_engines, "SafeEngine", FakeEngine)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(preflight, "generator_prompt_ids", lambda *a: ([1], "prompt"))
    monkeypatch.setattr(legacy_runner, "shutdown_engine_checked", close)
    result = preflight.eager_vs_graph(view)
    assert result["blocking"] is False and result["threshold"] == .75
    assert result["ok"] is (agree_tokens >= 48)
    assert len(result["rows"]) == n_rows and result["expanded_to_20"] is (n_rows == 20)
    assert len(active) == 1 and len(history) == 3
    assert all(kw["max_num_seqs"] == seqs for kw in history)
    assert exp.engines["G"].engine_kwargs == original
    assert exp.ctx.gen is exp.ctx_H.gen is active[0]
    assert len(result["process_cleanup"]) == 2
    assert read_json(exp.dir / "preflight/phases/pre_engines/eager_vs_graph.json") == result


@pytest.mark.parametrize("condition,role", [("B13_GREEDY", "B"), ("Q9_GREEDY", "O")])
def test_graph_diagnostic_publishes_memory_skip_for_non_g(workspace, condition, role):
    exp = experiment(workspace, CONDITION=condition, SPLIT="dev")
    tok, gen = object(), object()
    exp.engines, exp.toks = {role: gen}, {role: tok}
    exp.ctx = SimpleNamespace(gen=gen, tok=tok)
    result = preflight.eager_vs_graph(exp)
    assert result["ok"] and result["skipped"] and not result["blocking"]
    assert result["generator_role"] == role and "memory" in result["summary"]
    assert read_json(exp.dir / "preflight/eager_vs_graph.json") == result


def test_tokenizer_preparation_uses_native_o_role(workspace, monkeypatch):
    from jevlab import models, prompts

    snapshot, cache = workspace / "snapshot-O", workspace / "hf"
    snapshot.mkdir()
    cache.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    (snapshot / "tokenizer.json").write_text("{}")
    tok = SimpleNamespace(pad_token_id=0)
    exp = experiment(workspace, CONDITION="Q9_GREEDY", SPLIT="dev", HF_CACHE=str(cache))
    seen = []
    monkeypatch.setattr(models, "snapshot", lambda repo, revision, cache:
                        seen.append((repo, revision)) or str(snapshot))
    monkeypatch.setattr(models, "verify_snapshot", lambda *a: {"ok": True, "mismatches": []})
    monkeypatch.setattr(models, "param_inventory", lambda *a: {"language_params": 9})
    monkeypatch.setattr(models, "eos_ids", lambda tok: [42])
    monkeypatch.setattr(models, "newline_token_ids", lambda tok: {10, 11})
    monkeypatch.setattr(models, "special_ids", lambda tok: {"pad": 0})
    monkeypatch.setattr(prompts, "load_tokenizer", lambda path: tok)
    exp.prepare_models()
    assert seen == [("Qwen/Qwen3.5-9B", OFFICIAL_Q9_REVISION)]
    assert set(exp.toks) == set(exp.manifest["tokens"]) == {"O"}
    assert read_json(cache / f"newline_ids_O_{OFFICIAL_Q9_REVISION[:10]}.json") == [10, 11]
    assert preflight.tokenizer_check(exp)["roles"]["O"]["ok"]


def fake_consent(exp):
    """Synthetic unit-test evidence only; never writes an actual project approval."""
    incoming = exp.root / "incoming"
    incoming.mkdir(exist_ok=True)
    entries = {}
    for condition in CONDITIONS:
        count = 150 if condition in ("G_SINGLE", "JFINAL", "B13") else 50
        cfg = json.loads(json.dumps(exp.config))
        cfg["params"].update(CONDITION=condition, SPLIT="pilot", RUN_TAG="synthetic-pilot-fixture",
                             SEEDS="17,29,43" if count == 150 else "17")
        digest = config_hash(cfg)
        manifest = {"condition": condition, "config": cfg, "config_hash": digest,
                    "coverage_complete": True, "finished_utc": "2026-01-01T00:00:00+00:00",
                    "expected_coverage": {"n_problems": 50, "n_predictions": count}}
        path = incoming / f"{condition}_fixture_final.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
        entries[condition] = {"verified": True, "denominator": count,
                             "rates": {"denominator": count, "timeout_infra_rate": 0},
                             "verification": {"ok": True, "condition": condition, "records": count,
                                 "config_hash": digest, "archive": str(path), "archive_sha256": sha256_file(path)}}
    report = {"protocol_version": "3", "code_version": "0.3.0", "go": True, "decision": "go",
              "errors": [], "budget_approval": False, "conditions": entries,
              "aggregate": {"complete": True, "denominator": 600, "observed_cases": 600, "timeout_infra_rate": 0}}
    report_path = exp.root / "synthetic-pilot-fixture.json"
    report_path.write_text(json.dumps(report))
    approval = {"approved": True, "approved_by": "unit-test-fixture-not-real-consent",
                "approved_utc": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat(),
                "scope": "test_and_latency", "max_gpu_hours": 1,
                "config_sha256": exp.config["experiment_config_sha256"],
                "pilot_report_sha256": sha256_file(report_path)}
    approval_path = exp.root / "synthetic-approval-fixture.json"
    approval_path.write_text(json.dumps(approval))
    exp.p.update(APPROVAL_FILE=str(approval_path), PILOT_REPORT_FILE=str(report_path), CONFIRMATORY_AUTHORIZED=True)
    return approval_path, report_path


def test_missing_runtime_consent_blocks_before_gpu_work(workspace, monkeypatch):
    exp = experiment(workspace)
    monkeypatch.setattr(exp, "capture_environment", lambda: pytest.fail("must not probe/allocate GPU"))
    with pytest.raises(RuntimeError, match="APPROVAL_FILE"):
        exp.run_all()
    assert not exp.engines and exp.manifest["authorization"]["checked"] is False


def test_runtime_consent_binds_existing_operator_fields_and_archives(workspace):
    exp = experiment(workspace)
    fake_consent(exp)
    exp._check_authorization()
    assert exp.manifest["authorization"]["checked"] and exp.manifest["authorization"]["required"]
    assert len(exp.manifest["authorization"]["pilot_archive_sha256"]) == 6
    assert exp.manifest["authorization"]["gpu_hour_meter"] == "operator-owned"
    exp.p["CONFIRMATORY_AUTHORIZED"] = False
    with pytest.raises(RuntimeError, match="CONFIRMATORY_AUTHORIZED"):
        exp._check_authorization()


@pytest.mark.parametrize("damage", ["config", "report", "scope", "hours", "future", "no-go", "missing-archive", "changed-archive"])
def test_runtime_consent_rejects_unbound_or_incomplete_evidence(workspace, damage):
    exp = experiment(workspace)
    approval_path, report_path = fake_consent(exp)
    approval = read_json(approval_path)
    if damage == "config":
        approval["config_sha256"] = "0" * 64
    elif damage == "report":
        approval["pilot_report_sha256"] = "0" * 64
    elif damage == "scope":
        approval["scope"] = "latency"
    elif damage == "hours":
        approval["max_gpu_hours"] = True
    elif damage == "future":
        approval["approved_utc"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat()
    elif damage == "no-go":
        report = read_json(report_path)
        report["go"] = False
        report_path.write_text(json.dumps(report))
        approval["pilot_report_sha256"] = sha256_file(report_path)
    else:
        path = exp.root / "incoming/G_SINGLE_fixture_final.zip"
        path.unlink()
        if damage == "changed-archive":
            path.write_text("changed")
    approval_path.write_text(json.dumps(approval))
    with pytest.raises((ValueError, FileNotFoundError)):
        exp._check_authorization()


@pytest.mark.parametrize("damage", ["data", "config", "coverage", "seeds"])
def test_pilot_provenance_still_rejected_when_outer_hashes_are_updated(workspace, damage):
    exp = experiment(workspace)
    approval_path, report_path = fake_consent(exp)
    report = read_json(report_path)
    evidence = report["conditions"]["G_SINGLE"]["verification"]
    path = Path(evidence["archive"])
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if damage == "data":
        manifest["config"]["data_sha256"]["pilot_inputs.jsonl"] = "0" * 64
    elif damage == "config":
        manifest["config"]["experiment_config_sha256"] = "0" * 64
    elif damage == "coverage":
        manifest["expected_coverage"]["n_problems"] = 49
    else:
        manifest["config"]["params"]["SEEDS"] = "17"
    digest = config_hash(manifest["config"])
    manifest["config_hash"] = evidence["config_hash"] = digest
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    evidence["archive_sha256"] = sha256_file(path)
    report_path.write_text(json.dumps(report))
    approval = read_json(approval_path)
    approval["pilot_report_sha256"] = sha256_file(report_path)
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="binding mismatch"):
        exp._check_authorization()


def safe_engine(core, callback=None):
    engine = object.__new__(v3_engines.SafeEngine)
    engine.name, engine.engine, engine._n = "fake", core, 0
    engine.eos_ids, engine.newline_ids = [], set()
    engine.tok = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))
    engine.usable, engine.last_cleanup = True, None
    engine.cleanup_timeout_s, engine.cleanup_max_steps = .05, 3
    engine.failure_callback = callback
    return engine


@pytest.mark.parametrize("failure", ["add", "step"])
def test_safe_engine_aborts_all_attempted_ids_before_reuse(failure):
    order, active, aborted = [], set(), []

    def add(rid, *args):
        active.add(rid)
        if failure == "add" and rid == "two":
            raise RuntimeError("accepted then failed")

    def step():
        raise RuntimeError("step failed")

    def abort(ids):
        order.append("abort")
        aborted.extend(ids)
        active.difference_update(ids)

    engine = safe_engine(SimpleNamespace(add_request=add, step=step, abort_request=abort,
                                         has_unfinished_requests=lambda: bool(active)),
                         lambda row: order.append("failure" if row["cleanup_pending"] else "cleanup"))
    with pytest.raises(RuntimeError) as caught:
        engine.run([Req("one", [1], {}), Req("two", [1], {})])
    assert aborted == ["one", "two"] and not active
    assert order == ["failure", "abort", "cleanup"]
    assert caught.value.engine_cleanup["cleanup_confirmed"] is True
    assert caught.value.engine_cleanup["unfinished"] is False and engine.usable
    assert engine.engine.add_request is add  # The tracking proxy is never left installed.


@pytest.mark.parametrize("unfinished", [True, 0, None])
def test_safe_engine_unknown_cleanup_permanently_stops_reuse(unfinished):
    calls = []
    core = SimpleNamespace(add_request=lambda rid, *a: calls.append(rid),
                           step=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
                           abort_request=lambda ids: None,
                           has_unfinished_requests=lambda: unfinished)
    engine = safe_engine(core)
    with pytest.raises(RuntimeError) as caught:
        engine.run([Req("one", [1], {})])
    assert caught.value.engine_cleanup["cleanup_confirmed"] is False and engine.usable is False
    with pytest.raises(RuntimeError, match="unusable"):
        engine.run([Req("two", [1], {})])
    assert calls == ["one"]


def test_safe_engine_cleanup_hang_is_bounded():
    release = threading.Event()
    core = SimpleNamespace(add_request=lambda *a: (_ for _ in ()).throw(RuntimeError("add failed")),
                           abort_request=lambda ids: release.wait(1), has_unfinished_requests=lambda: False)
    engine = safe_engine(core)
    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError) as caught:
            engine.run([Req("one", [1], {})])
        assert time.perf_counter() - started < .5
        assert caught.value.engine_cleanup["timed_out"] is True and not engine.usable
    finally:
        release.set()


def install_case_context(exp, monkeypatch, gen=None):
    counter = iter(range(100))
    gen = gen or SimpleNamespace(new_id=lambda tag: f"{tag}-{next(counter)}",
        run=lambda reqs, deadline: {r.rid: ReqResult(r.rid, 1, list(map(ord, "FINAL: 1\n")),
                                                   finish_reason="final_abort") for r in reqs})
    tok = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))
    exp.engines, exp.toks = {"G": gen}, {"G": tok}
    exp.eos, exp.nl = {"G": [0]}, {"G": {10}}
    if hasattr(gen, "eos_ids"):
        gen.eos_ids, gen.newline_ids = [0], {10}
    exp.ctx = exp._ctx()
    monkeypatch.setattr(algorithms, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(algorithms, "generator_prompt_ids", lambda *a: ([1], "prompt"))
    monkeypatch.setattr(exp, "assert_same_gpu", lambda: None)
    monkeypatch.setattr(exp, "checkpoint", lambda *a, **kw: None)
    return gen


@pytest.mark.parametrize("confirmed", [True, False])
def test_case_failure_is_committed_before_unknown_cleanup_stops(workspace, monkeypatch, confirmed):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    adds = []
    active = set()

    def add(rid, *a):
        adds.append(rid)
        active.add(rid)
        raise RuntimeError("accepted then failed")

    engine = safe_engine(SimpleNamespace(add_request=add,
        abort_request=lambda ids: active.clear() if confirmed else None,
        has_unfinished_requests=lambda: bool(active),
        step=lambda: (_ for _ in ()).throw(RuntimeError("still broken"))), exp._engine_failure)
    install_case_context(exp, monkeypatch, engine)
    if confirmed:
        exp.run()
    else:
        with pytest.raises(RuntimeError, match="unusable"):
            exp.run()
    predictions = read_jsonl(exp.f["predictions"])
    assert len(predictions) == len(adds) == (2 if confirmed else 1)
    assert all(r["status"] == "engine_error" and r["error_traceback"] for r in predictions)
    assert all(r["engine_cleanup"]["cleanup_confirmed"] is confirmed for r in predictions)
    assert all(r["context_frame_id"] == config_hash(r["context_frame"]) and r["execution_id"] for r in predictions)
    assert all(r["no_candidates_produced"] and r["candidate_count_observed"] == 0 for r in predictions)
    assert all(not r["proposal_expected"] and not r["proposal_persisted"] for r in predictions)
    assert all(r["request_ids"] == r["engine_cleanup"]["attempted_ids"] for r in predictions)
    assert all(r["candidate_trace_absence"] == "no_observed_engine_results" for r in predictions)
    assert read_jsonl(exp.f["failures"])[0]["cleanup_pending"] is True
    before = len(adds)
    if confirmed:
        exp.run()
    assert len(adds) == before  # Confirmed ordinary failures are not retried for quality.


@pytest.mark.parametrize("confirmed", [True, False])
def test_selector_engine_cleanup_protects_all_active_engines(workspace, monkeypatch, confirmed):
    exp = experiment(workspace, CONDITION="JFINAL", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    install_case_context(exp, monkeypatch)
    active = set()

    def add(rid, *args):
        active.add(rid)
        raise RuntimeError("J accepted request then failed")

    selector_engine = safe_engine(SimpleNamespace(add_request=add,
        abort_request=lambda ids: active.clear() if confirmed else None,
        has_unfinished_requests=lambda: bool(active),
        step=lambda: (_ for _ in ()).throw(RuntimeError("J still broken"))), exp._engine_failure)
    exp.engines["J"] = selector_engine

    def decide(*args):
        return selector_engine.run([Req(selector_engine.new_id("sel"), [1], {})], args[-1])

    exp.selector = SimpleNamespace(engine=selector_engine, decide=decide)
    exp.ctx = exp._ctx()
    if confirmed:
        exp.run()
    else:
        with pytest.raises(RuntimeError, match="unusable.*J"):
            exp.run()
    rows = read_jsonl(exp.f["predictions"])
    assert len(rows) == (2 if confirmed else 1)
    assert all(r["status"] == "selector_error" and r["engine_cleanup"]["cleanup_confirmed"] is confirmed for r in rows)
    assert len(read_jsonl(exp.f["proposals"])) == len(rows)
    assert len(read_jsonl(exp.f["candidates"])) == len(rows) * 4
    assert all(r["raw_output"] == "" for r in rows)


def test_j_readout_probe_never_falls_back_on_an_unusable_engine(monkeypatch):
    from jevlab.selector import Selector

    adds = []
    core = SimpleNamespace(add_request=lambda rid, *a: adds.append(rid) or
                           (_ for _ in ()).throw(RuntimeError("readout failed")),
                           abort_request=lambda ids: None, has_unfinished_requests=lambda: True,
                           step=lambda: (_ for _ in ()).throw(RuntimeError("broken cleanup")))
    engine = safe_engine(core)
    tok = SimpleNamespace(encode=lambda text, **kw: [ord(text)] if len(text) == 1 else [1])
    selector = Selector(engine, tok, 1.316)
    monkeypatch.setattr(selector, "_params", lambda n: {})
    with pytest.raises(RuntimeError, match="unusable") as caught:
        selector.probe()
    assert len(adds) == 1 and not engine.usable
    assert caught.value.engine_cleanup["cleanup_confirmed"] is False


@pytest.mark.parametrize("drift", ["sampling", "limits", "system_prompt", "crit_final", "params", "prompt_file", "config", "code"])
def test_identity_drift_rejects_before_any_run_artifact(workspace, monkeypatch, drift):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    install_case_context(exp, monkeypatch)
    if drift in ("sampling", "limits"):
        getattr(exp.ctx, drift)["temperature" if drift == "sampling" else "max_output_tokens"] = 5
    elif drift in ("system_prompt", "crit_final"):
        setattr(exp.ctx, drift, "changed")
    elif drift == "params":
        exp.p["TEMPERATURE"] = .2
    elif drift == "prompt_file":
        (workspace / "prompts/generador.txt").write_text("changed")
    elif drift == "config":
        exp.config["profile"] = "changed"
    else:
        monkeypatch.setattr(exp, "_code_hashes", lambda: {"changed": "0" * 64})
    before = {p.relative_to(exp.dir): p.read_bytes() for p in exp.dir.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="drift|identity|parameter"):
        exp.run()
    assert {p.relative_to(exp.dir): p.read_bytes() for p in exp.dir.rglob("*") if p.is_file()} == before
    assert not read_jsonl(exp.f["predictions"]) and not read_jsonl(exp.f["proposals"])


@pytest.mark.parametrize("target", ["ctx", "engine", "prepared"])
@pytest.mark.parametrize("field", ["eos_ids", "newline_ids"])
def test_termination_id_mutation_rejected_in_dev(workspace, monkeypatch, target, field):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    gen = install_case_context(exp, monkeypatch)
    gen.eos_ids, gen.newline_ids = [0], {10}
    if target == "prepared":
        (exp.eos["G"] if field == "eos_ids" else exp.nl["G"]).clear()
    else:
        getattr(exp.ctx if target == "ctx" else gen, field).clear()
    with pytest.raises(ValueError, match="termination ID drift"):
        exp.run()
    assert not read_jsonl(exp.f["predictions"])


@pytest.mark.parametrize("reason", ["deadline", "final"])
def test_safe_engine_termination_abort_hang_is_bounded_without_second_abort(reason):
    active, calls = set(), []
    release = threading.Event()

    def abort(ids):
        calls.append(list(ids))
        release.wait(1)

    core = SimpleNamespace(add_request=lambda rid, *a: active.add(rid), abort_request=abort,
        has_unfinished_requests=lambda: bool(active),
        step=lambda: [SimpleNamespace(request_id="one", finished=False,
                                      outputs=[SimpleNamespace(token_ids=list(map(ord, "FINAL: 1\n")))])])
    engine = safe_engine(core)
    engine.newline_ids = {10}
    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError) as caught:
            engine.run([Req("one", [1], {})], deadline=0 if reason == "deadline" else None)
        assert time.perf_counter() - started < .5
        assert not engine.usable and caught.value.engine_cleanup["timed_out"] is True
        assert calls == [["one"]]
    finally:
        release.set()


def test_context_drift_between_cases_does_not_rehash_or_infer(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    gen = install_case_context(exp, monkeypatch)
    original = gen.run

    def drift(reqs, deadline):
        result = original(reqs, deadline)
        exp.ctx.sampling["temperature"] = .2
        return result

    gen.run = drift
    before = exp.chash, exp.dir
    with pytest.raises(ValueError, match="context drift"):
        exp.run()
    assert len(read_jsonl(exp.f["predictions"])) == 1
    assert (exp.chash, exp.dir) == before


@pytest.mark.parametrize("target", ["raw_ctx_extra", "engine_extra"])
def test_cross_case_raw_safety_snapshot_rejects_extra_drift(workspace, monkeypatch, target):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    gen = install_case_context(exp, monkeypatch)
    original = gen.run

    def change_extra(reqs, deadline):
        result = original(reqs, deadline)
        if target == "raw_ctx_extra":
            exp.ctx.untracked_override = {"nested": ["changed"]}
        else:
            gen.engine_kwargs = {"enable_prefix_caching": True}
        return result

    gen.run = change_extra
    with pytest.raises(ValueError, match="extras drift"):
        exp.run()
    assert len(read_jsonl(exp.f["predictions"])) == 1


def test_irrelevant_selector_seed_changes_do_not_change_context_identity(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="JFINAL", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    install_case_context(exp, monkeypatch)
    jengine = SimpleNamespace(engine_kwargs={"enable_prefix_caching": False, "seed": 0})
    exp.engines["J"] = jengine
    exp.manifest["selector"] = {"readout_used": exp.p["J_READOUT"]}
    selector = SimpleNamespace(engine=jengine, temperature=1.316, max_input=exp.p["J_MAX_INPUT"],
                               readout=exp.p["J_READOUT"], letter_ids=[1, 2, 3, 4], seed=0)

    def decide(*args):
        selector.seed += 1
        jengine.engine_kwargs["seed"] += 1
        return {"status": "ok", "winner_branch": 0, "input_tokens": 10}

    selector.decide = decide
    exp.selector, exp.ctx = selector, None
    exp.ctx = exp._ctx()
    exp.run()
    rows = read_jsonl(exp.f["predictions"])
    assert len(rows) == 2 and selector.seed == 2
    assert len({row["context_frame_id"] for row in rows}) == 1


@pytest.mark.parametrize("status", ["selector_context_limit", "selector_error", "timeout"])
def test_selector_failure_frame_and_actual_decision_precede_prediction_without_retry(workspace, monkeypatch, status):
    import jevlab.v3.runner as runner

    exp = experiment(workspace, CONDITION="JFINAL", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    install_case_context(exp, monkeypatch)
    calls, writes = [], []

    def decide(*args):
        calls.append(args)
        if status == "selector_context_limit":
            return {"status": status, "input_tokens": 16385}
        raise TimeoutError("selector deadline") if status == "timeout" else RuntimeError("selector error")

    exp.selector = SimpleNamespace(decide=decide)
    exp.ctx = exp._ctx()
    original_append = runner.append_jsonl

    def observed_write(path, row):
        writes.append((Path(path).stem, row.get("resume_key")))
        original_append(path, row)

    monkeypatch.setattr(runner, "append_jsonl", observed_write)
    exp.run()
    predictions = read_jsonl(exp.f["predictions"])
    failures = read_jsonl(exp.f["failures"])
    assert len(predictions) == len(failures) == len(calls) == 2
    for pred, failure in zip(predictions, failures):
        assert pred["status"] == failure["status"] == status
        assert failure["context_frame_id"] == pred["context_frame_id"] == config_hash(pred["context_frame"])
        assert failure["execution_id"] == pred["execution_id"]
        assert failure["decision"]["status"] == status
        assert len(failure["request_ids"]) == 4
        assert failure["candidate_count_observed"] == 4 and not failure["no_candidates_produced"]
        assert failure["proposal_expected"] and failure["proposal_persisted"]
        key = pred["resume_key"]
        assert writes.index(("failures", key)) < writes.index(("decisions", key)) < writes.index(("predictions", key))
    assert all(row["raw_output"] == "" and "winner_branch" not in row for row in predictions)
    exp.run()
    assert len(calls) == 2


def test_public_manifest_source_tier_axis_accepts_actual_zero_hard_counts(workspace):
    exp = experiment(workspace, SPLIT="pilot")
    assert exp.manifest["quota_axis"] == "source_sampling_tier"
    assert exp.manifest["intrinsic_difficulty_counts"]["test"] == {"easy": 500, "medium": 0, "hard": 0}
    assert exp.manifest["intrinsic_difficulty_counts"]["pilot"] == {"easy": 50, "medium": 0, "hard": 0}


@pytest.mark.parametrize("marker", ["manifest", "zip"])
def test_existing_final_refused_before_writes_or_environment(workspace, monkeypatch, marker):
    exp = experiment(workspace, SPLIT="dev", SEEDS="17", N_PROBLEMS=1)
    if marker == "manifest":
        manifest = read_json(exp.dir / "manifest.json")
        manifest["finished_utc"] = "published"
        (exp.dir / "manifest.json").write_text(json.dumps(manifest))
    else:
        (exp.artifact_root / f"{exp.run_name}_final.zip").write_bytes(b"published bytes")
    before = {p.relative_to(exp.artifact_root): p.read_bytes() for p in exp.artifact_root.rglob("*") if p.is_file()}
    monkeypatch.setattr(hw, "gpu_info", lambda: pytest.fail("must not probe GPU"))
    monkeypatch.setattr("jevlab.v3.runner.setup_logging", lambda *a: pytest.fail("must not configure logging"))
    with pytest.raises(FileExistsError, match="immutable"):
        experiment(workspace, SPLIT="dev", SEEDS="17", N_PROBLEMS=1)
    assert {p.relative_to(exp.artifact_root): p.read_bytes() for p in exp.artifact_root.rglob("*") if p.is_file()} == before


def test_partial_resume_preserves_manifest_and_strict_prediction_multiplicity(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    install_case_context(exp, monkeypatch)
    exp.run()
    created = exp.manifest["created_utc"]
    resumed = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=2, SEEDS="17")
    assert resumed.chash == exp.chash and resumed.manifest["created_utc"] == created
    install_case_context(resumed, monkeypatch)
    resumed.run()
    rows = read_jsonl(resumed.f["predictions"])
    assert len(rows) == 2
    append_jsonl(resumed.f["predictions"], rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        resumed.run()
    with pytest.raises(ValueError, match="duplicate"):
        resumed.finalize()


def test_partial_single_engine_traces_cannot_be_regenerated(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    gen = install_case_context(exp, monkeypatch)
    gen.run = lambda *args: pytest.fail("must not regenerate an uncommitted case")
    key = exp._key(exp.problems()[0]["id"], 17)
    append_jsonl(exp.f["metrics"], {"resume_key": key, "status": "engine_error"})
    with pytest.raises(RuntimeError, match="uncommitted durable"):
        exp.run()


def test_final_counts_reject_duplicate_companion_metrics(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    install_case_context(exp, monkeypatch)
    exp.run()
    append_jsonl(exp.f["metrics"], read_jsonl(exp.f["metrics"])[0])
    with pytest.raises(ValueError, match="metrics multiplicity"):
        exp.finalize()
    assert not exp.manifest.get("finished_utc")


def test_final_zip_exclusive_publication_never_replaces_competing_bytes(workspace, monkeypatch):
    import os

    exp = experiment(workspace, SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    original = os.link
    final = exp.artifact_root / f"{exp.run_name}_final.zip"

    def competing_publish(source, target):
        Path(target).write_bytes(b"different published bytes")
        return original(source, target)

    monkeypatch.setattr("jevlab.v3.runner.os.link", competing_publish)
    with pytest.raises(FileExistsError):
        exp.finalize()
    assert final.read_bytes() == b"different published bytes"
    assert not read_json(exp.dir / "manifest.json").get("finished_utc")


def test_final_publication_uses_unique_closed_inode_and_unlink_failure_is_not_rollback(workspace, monkeypatch):
    import os

    exp = experiment(workspace, SPLIT="dev", N_PROBLEMS=1, SEEDS="17")
    original_link, original_unlink = os.link, os.unlink
    observed = []

    def publish(source, target):
        source = Path(source)
        assert source.name != Path(str(target) + ".tmp").name
        assert zipfile.is_zipfile(source)  # Writer is closed before exclusive publication.
        observed.append(source)
        original_link(source, target)
        # A second publisher's former shared staging name cannot truncate final bytes.
        Path(str(target) + ".tmp").write_bytes(b"another publisher")

    def unlink(path, *args, **kwargs):
        if Path(path) in observed:
            raise PermissionError("injected post-publication temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr("jevlab.v3.runner.os.link", publish)
    monkeypatch.setattr("jevlab.v3.runner.os.unlink", unlink)
    final = exp.finalize()
    assert exp._published and zipfile.is_zipfile(final)
    manifest = read_json(exp.dir / "manifest.json")
    assert manifest["finished_utc"]
    with zipfile.ZipFile(final) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/manifest.json"))
        assert json.loads(archive.read(name))["finished_utc"] == manifest["finished_utc"]


def test_incremental_preflight_and_false_skipped_flags_survive_later_exception(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    monkeypatch.setattr(preflight, "parser_selftest", lambda: {"ok": True, "blocking": True, "skipped": False})
    monkeypatch.setattr(preflight, "lock_check", lambda *a: (_ for _ in ()).throw(RuntimeError("later failure")))
    with pytest.raises(RuntimeError, match="later failure"):
        preflight.pre_engine_checks(exp)
    summary = read_json(exp.dir / "preflight/summary.json")
    assert summary["parser_selftest"]["skipped"] is False
    assert summary["lock"]["ok"] is False and "later failure" in summary["lock"]["traceback"]
    manifest = read_json(exp.dir / "manifest.json")
    assert manifest["preflight"]["parser_selftest"]["skipped"] is False
    assert manifest["preflight"]["lock"]["ok"] is False


def test_v3_continuity_false_is_blocking_and_stops_later_checks(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="G_SINGLE", SPLIT="dev")
    exp.ctx = SimpleNamespace(selector=None)
    monkeypatch.setattr(preflight, "eager_vs_graph", lambda *a: {"ok": True, "blocking": False})
    monkeypatch.setattr(preflight, "cache_check", lambda *a: {"ok": True, "blocking": True})
    monkeypatch.setattr(preflight.legacy, "continuity_test", lambda *a: {"ok": False, "blocking": False})
    monkeypatch.setattr(preflight, "stress_test", lambda *a: pytest.fail("must stop after continuity"))
    result = preflight.post_engine_checks(exp)
    assert result["continuity"]["blocking"] is True
    with pytest.raises(RuntimeError, match="continuity"):
        exp._check_blocking(result)


def selector_reference(exp, monkeypatch, count=2):
    items = [{"state": "state", "question": {"criteria": {f"option_{i}": str(i) for i in range(4)}}}
             for _ in range(count)]
    exp.paths, exp.toks = {"J": "fake-native-snapshot"}, {"J": object()}
    exp.engines = {"G": object(), "J": object()}
    exp.gpu_uuid = "GPU-fake"
    monkeypatch.setattr(preflight.legacy, "selector_items", lambda *a: items)
    monkeypatch.setattr("jevlab.prompts.selector_prompt_ids", lambda *a: ([1, 2], "prompt"))
    logits = [2., 1., 0., -1.]
    weights = [math.exp(x / 1.316) for x in logits]
    probs = [x / sum(weights) for x in weights]
    exp.selector = SimpleNamespace(letter_logits_from_ids=lambda *a: (logits, .01), probabilities=lambda *a: probs)
    exp.ctx = SimpleNamespace(selector=exp.selector)
    identity = preflight._identity(exp, items)
    native = {"ok": True, "blocking": True, "identity": identity,
              "native_execution_id": "fake-native-execution",
              "reference_id": config_hash(identity), "phase_epoch": 0, "phase_id": None,
              "gpu_uuid": exp.gpu_uuid, "resident_roles": [],
              "native": {"identity": identity, "reference_id": config_hash(identity),
                  "native_execution_id": "fake-native-execution",
                  "runtime_commit": exp.cfg["jevk5_runtime"]["commit"], "temperature": 1.316,
                  "items": [{"index": index, "reference_id": config_hash(identity),
                             "ids": [1, 2], "ids_sha256": sha256_text(json.dumps([1, 2])),
                             "ids_equal_prompt_text": True, "raw_logits": list(logits),
                             "probabilities": list(probs), "seconds": .01} for index, _ in enumerate(items)]}}
    exp._begin_phase()
    exp.phase_cleanup_confirmed = True
    return native


def test_equivalence_rows_persist_before_later_readout_exception(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch)
    original = exp.selector.letter_logits_from_ids
    calls = []

    def readout(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("second readout failed")
        return original(*args)

    exp.selector.letter_logits_from_ids = readout
    with pytest.raises(RuntimeError, match="second readout"):
        preflight.selector_equivalence(exp, native)
    result = read_json(exp.dir / "preflight/phases" / exp.phase_id / "selector_equivalence.json")
    assert result["ok"] is False and len(result["rows"]) == 1
    assert result["rows"][0]["ok"] is True
    assert result["rows"][0]["native_probs"] == native["native"]["items"][0]["probabilities"]
    assert result["rows"][0]["phase_id"] == exp.phase_id
    assert "second readout failed" in result["traceback"]
    assert read_json(exp.dir / "manifest.json")["preflight"]["selector_equivalence"]["ok"] is False


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf")])
@pytest.mark.parametrize("vector", ["raw_logits", "probabilities"])
def test_native_equivalence_rejects_boolean_or_nonfinite_evidence(workspace, monkeypatch, invalid, vector):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch, count=1)
    native["native"]["items"][0][vector][0] = invalid
    result = preflight.selector_equivalence(exp, native)
    assert result["ok"] is False and result["rows"][0]["ok"] is False
    assert "non-bool" in result["rows"][0]["error"]


def test_native_identity_drift_rejected_without_readout(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch)
    native["identity"] = {**native["identity"], "config_hash": "different"}
    exp.selector.letter_logits_from_ids = lambda *a: pytest.fail("must not use stale reference")
    result = preflight.selector_equivalence(exp, native)
    assert result["ok"] is False and not result["rows"]


def test_trace_audit_requires_only_current_phase_logs_and_native_json(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch)
    log_path = exp.dir / "logs/vllm.log"
    log_path.write_text("old phase log\n")
    exp._phase_log_start = log_path.stat().st_size
    result = preflight.selector_equivalence(exp, native)
    assert result["ok"] and result["reference_phase"]["phase_epoch"] == 0
    assert not preflight.trace_audit(exp)["ok"]  # An old nonempty worker log is not evidence.
    with log_path.open("a") as stream:
        stream.write("current engine arguments, compile and memory\n")
    assert preflight.trace_audit(exp)["ok"]
    phase_file = exp.dir / "preflight/phases" / exp.phase_id / "native_reference_trace.json"
    trace = read_json(phase_file)
    trace["phase_id"] = "old-phase"
    write_json(phase_file, trace)
    assert not preflight.trace_audit(exp)["ok"]


@pytest.mark.parametrize("damage", ["probabilities", "raw_logits", "boolean", "missing"])
def test_trace_audit_binds_actual_native_numbers_not_only_identity_labels(workspace, monkeypatch, damage):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch)
    (exp.dir / "logs/vllm.log").write_text("current phase engine log\n")
    assert preflight.selector_equivalence(exp, native)["ok"]
    assert preflight.trace_audit(exp)["ok"]
    path = exp.dir / "preflight/phases" / exp.phase_id / "native_reference_trace.json"
    trace = read_json(path)
    if damage == "missing":
        trace["output"]["items"] = []
    elif damage == "boolean":
        trace["output"]["items"][0]["probabilities"][0] = True
    else:
        trace["output"]["items"][0][damage][0] += .01
    write_json(path, trace)
    assert not preflight.trace_audit(exp)["ok"]


def test_native_subprocess_rejects_stale_output_and_keeps_full_diagnostics(workspace, monkeypatch):
    exp = experiment(workspace, SPLIT="dev")
    selector_reference(exp, monkeypatch, count=1)
    out = exp.dir / "preflight/phases" / exp.phase_id / "native_ref_out.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"items": ["stale"]})

    def subprocess_run(cmd, **kwargs):
        assert not Path(cmd[4]).exists()  # Output removed before this reference execution.
        return SimpleNamespace(returncode=1, stdout="s" * 30000, stderr="e" * 30000)

    monkeypatch.setattr(preflight.subprocess, "run", subprocess_run)
    result = preflight.native_reference(exp)
    assert result["ok"] is False and "native" not in result
    assert len(result["stdout"]) == len(result["stderr"]) == 30000
    assert (out.parent / "native_ref_stderr.txt").read_text() == "e" * 30000


@pytest.mark.parametrize("damage", [None, "cpu", "dtype", "uuid", "boolean", "execution"])
def test_native_subprocess_validates_actual_hardware_dtype_and_execution_identity(workspace, monkeypatch, damage):
    exp = experiment(workspace, SPLIT="dev")
    native = selector_reference(exp, monkeypatch, count=1)

    def subprocess_run(cmd, **kwargs):
        payload = read_json(cmd[3])
        assert not Path(cmd[4]).exists()
        reference = json.loads(json.dumps(native["native"]))
        reference.update(native_execution_id=payload["native_execution_id"],
                         runtime_version="0.3.0", snapshot=payload["identity"]["snapshot"],
                         model_device="cuda:0", model_dtype="torch.bfloat16",
                         cuda_hardware={"returncode": 0, "stdout": "GPU-fake,NVIDIA A100,driver,40960", "stderr": ""})
        if damage == "cpu":
            reference["model_device"] = "cpu"
        elif damage == "dtype":
            reference["model_dtype"] = "torch.float32"
        elif damage == "uuid":
            reference["cuda_hardware"]["stdout"] = "GPU-other,NVIDIA A100,driver,40960"
        elif damage == "boolean":
            reference["items"][0]["probabilities"][0] = True
        elif damage == "execution":
            reference["native_execution_id"] = "previous-execution"
        write_json(cmd[4], reference)
        return SimpleNamespace(returncode=0, stdout="fake stdout", stderr="fake stderr")

    monkeypatch.setattr(preflight.subprocess, "run", subprocess_run)
    result = preflight.native_reference(exp)
    assert result["ok"] is (damage is None)
    assert result["native"]["model_device"] == ("cpu" if damage == "cpu" else "cuda:0")


def test_native_wrapper_records_true_device_dtype_and_explicit_kwargs(workspace, monkeypatch):
    import types
    from jevlab.v3 import native_ref

    identity = {"snapshot": str((workspace / "snapshot").resolve()), "runtime": {"commit": "a" * 40}}
    inp, out = workspace / "native-in.json", workspace / "native-out.json"
    write_json(inp, {"identity": identity, "items": [], "native_execution_id": "fake-execution",
                     "phase_epoch": 0, "phase_id": None, "gpu_uuid": "GPU-fake", "resident_roles": [],
                     "cleanup_confirmed": False, "outside_T_total": True})
    torch = types.ModuleType("torch")
    torch.__version__, torch.bfloat16 = "fake", "fake-bfloat16"
    torch.version = SimpleNamespace(cuda="fake-cuda")
    torch.cuda = SimpleNamespace(is_initialized=lambda: False)
    runtime = types.ModuleType("jevk5")
    runtime.__file__ = str(workspace / "fake-runtime.py")
    observed = []

    def model(snapshot, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(temperature=1.316, model=SimpleNamespace(device="cpu", dtype="torch.float32"))

    runtime.JevK5 = model
    prompt = types.ModuleType("jevk5.prompt")
    prompt.decision_options, prompt.prompt_text = lambda q: [], lambda *a: ""
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "jevk5", runtime)
    monkeypatch.setitem(sys.modules, "jevk5.prompt", prompt)
    monkeypatch.setattr(native_ref.importlib.metadata, "distribution", lambda name:
                        SimpleNamespace(version="0.3.0", read_text=lambda name:
                                        json.dumps({"vcs_info": {"commit_id": "a" * 40}})))
    monkeypatch.setattr(native_ref.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="GPU-fake,A100,driver,40960", stderr=""))
    native_ref.main(str(inp), str(out), identity["snapshot"])
    result = read_json(out)
    assert observed == [{"device": "cuda", "dtype": "fake-bfloat16", "graphs": False}]
    assert result["model_device"] == "cpu" and result["model_dtype"] == "torch.float32"
    assert result["constructor_kwargs"]["dtype"] == "torch.bfloat16"
    assert result["cuda_hardware"]["stdout"].startswith("GPU-fake")
    assert result["runtime_commit"] == "a" * 40 and result["peak_mem_gib"] is None
