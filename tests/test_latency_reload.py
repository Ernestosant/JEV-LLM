import ast
import json
import sys
import zipfile
from collections import Counter
from types import SimpleNamespace

import pytest

from jevlab import algorithms, engines, preflight, prompts, selector
from jevlab.common import read_json, read_jsonl
from jevlab.latency_study import LatencyStudy
from jevlab.runner import Experiment, shutdown_engine_checked
from test_runtime_v2 import ROOT, tool, workspace


@pytest.fixture
def reload_study(workspace, monkeypatch):
    study = LatencyStudy(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2_reload.json", DATA_SUBDIR="data/v2",
                         WARMUP_N=1)
    study.paths = {r: str(workspace / "models" / r) for r in study.roles()}
    for path in study.paths.values():
        from pathlib import Path
        Path(path).mkdir(parents=True)
    from pathlib import Path
    Path(study.paths["J"]).joinpath("jevk5_config.json").write_text(json.dumps({"temperature": 1.316}))
    study.toks = dict.fromkeys(study.roles(), object())
    study.lock = {r: {"ok": True, "architectures": [], "params": {}} for r in study.roles()}
    study.eos, study.nl = {"G": [0], "B": [0]}, {"G": {1}, "B": {1}}
    study.eos_ids, study.newline_ids = study.eos["G"], study.nl["G"]
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("time.perf_counter", lambda: clock.now)
    live, events, probes = {}, [], []

    class Process:
        next_pid = 100

        def __init__(self, engine):
            self.pid = Process.next_pid
            Process.next_pid += 1
            self.exitcode, self.engine, self.joins = None, engine, []

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return self.engine.name in live

    class Core:
        def __init__(self, engine):
            self.owner = engine
            self.resources = SimpleNamespace(engine_manager=SimpleNamespace(processes=[Process(engine)]))

        def shutdown(self, timeout):
            events.append(("close", self.owner.name, clock.now))
            clock.now += 3
            if self.owner.stuck:
                return
            live.pop(self.owner.name)
            self.resources.engine_manager.processes[0].exitcode = 0

    class Engine:
        def __init__(self, name, model_dir, tok, **kwargs):
            assert name not in live
            assert not kwargs.get("sleep_mode")
            assert not (name == "B" and live)
            assert not (name != "B" and "B" in live)
            assert not (name.startswith("G") and any(r.startswith("G") for r in live))
            self.name, self.kwargs, self.stuck = name, kwargs, False
            self.engine_kwargs = {"enable_sleep_mode": False}
            self.engine = SimpleNamespace(engine_core=Core(self))
            self.process = self.engine.engine_core.resources.engine_manager.processes[0]
            live[name] = self
            events.append(("load", name, clock.now))
            clock.now += 5

        def sleep(self, *args):
            raise AssertionError("reload must never sleep")

        def wake(self):
            raise AssertionError("reload must never wake")

        def new_id(self, tag):
            return tag

        def run(self, reqs):
            assert self.engine is not None and self.name in live
            return {r.rid: SimpleNamespace(token_ids=[2 if self.kwargs["enforce_eager"] else 1] * r.params["max_tokens"])
                    for r in reqs}

    class Selector:
        def __init__(self, engine, tok, temperature, *args):
            assert temperature == 1.316
            self.engine = engine

        def probe(self):
            assert live["J"] is self.engine
            probes.append(self.engine.process.pid)
            clock.now += 2
            return "logprob_token_ids"

    monkeypatch.setattr(engines, "Engine", Engine)
    monkeypatch.setattr(selector, "Selector", Selector)
    return SimpleNamespace(study=study, live=live, events=events, probes=probes, clock=clock)


def test_reload_config_is_exact_amendment_and_hash_is_new(workspace):
    old = read_json(workspace / "config/experiment_v2.json")
    new = read_json(workspace / "config/experiment_v2_reload.json")
    assert new.pop("code_version") == "0.2.1"
    assert new.pop("amendment")["approved_by"] == "user"
    assert new["runtime_defaults"].pop("LATENCY_SWAP_MODE") == "reload"
    assert new == old
    args = dict(ROOT=str(workspace), DATA_SUBDIR="data/v2")
    frozen = LatencyStudy(CONFIG_FILE="config/experiment_v2.json", **args)
    amended = LatencyStudy(CONFIG_FILE="config/experiment_v2_reload.json", **args)
    assert frozen.swap_mode == "sleep" and amended.swap_mode == "reload"
    assert amended.chash != frozen.chash
    assert amended.config["code_version"] == "0.2.1"


def test_reload_phase_exclusivity_contexts_and_j_rebuild(reload_study):
    s, live = reload_study.study, reload_study.live
    s.start_engines()
    assert set(live) == {"G", "J"}
    assert s.ctx_B is None and s.ctx_H.gen is live["G"]
    g, j = live["G"], live["J"]
    assert g.kwargs["max_num_seqs"] == 4
    assert g.kwargs["kv_cache_gb"] == 3.0 and j.kwargs["kv_cache_gb"] == 4.0
    single = s._condition_context("G_SINGLE")
    assert set(live) == {"G"} and live["G"] is g
    assert single.selector is None and j.engine is None
    hybrid = s._condition_context("J64")
    assert set(live) == {"G", "J"} and live["G"] is g
    assert hybrid.selector.engine is live["J"] and live["J"] is not j
    s._swap("B")
    assert set(live) == {"B"} and s.ctx_H is None and s.selector is None
    assert s.ctx_B.gen is live["B"] and g.engine is None
    assert live["B"].kwargs["kv_cache_gb"] == 3.0
    s._swap("H")
    assert set(live) == {"G", "J"} and s.ctx_B is None
    assert s.ctx_H.gen is live["G"] and live["G"] is not g
    assert len(reload_study.probes) == 3
    swaps = read_jsonl(s.f["swaps"])
    assert all(r["outside_T_total"] and r["ok"] for r in swaps)
    assert all(c["cleanup_confirmed"] for r in swaps for c in r["closed"].values())
    s.shutdown()
    assert not live


def test_unconfirmed_cleanup_aborts_before_loading_next_phase(reload_study):
    s = reload_study.study
    s.start_engines()
    reload_study.live["J"].stuck = True
    with pytest.raises(RuntimeError, match="still alive"):
        s._swap("B")
    assert "B" not in reload_study.live
    assert "G" not in reload_study.live
    assert s.phase is None and s.ctx_H is None and s.ctx_B is None
    assert not read_jsonl(s.f["swaps"])[-1]["ok"]
    reload_study.live["J"].stuck = False
    s.shutdown()


def test_shutdown_requires_process_proof():
    core = SimpleNamespace(resources=SimpleNamespace(engine_manager=SimpleNamespace(processes=[])))
    with pytest.raises(RuntimeError, match="process handles"):
        shutdown_engine_checked(SimpleNamespace(engine=SimpleNamespace(engine_core=core)))


def test_reload_preflight_native_h_graph_then_b_stress(reload_study, monkeypatch):
    s, live = reload_study.study, reload_study.live
    checked = []
    ok = lambda *args: {"ok": True, "blocking": True, "summary": "fake check"}
    for name in ("template_check", "prompt_length_check", "sampling_semantics_check", "tokenizer_check"):
        monkeypatch.setattr(preflight, name, ok)

    def native(exp):
        assert not live
        checked.append("native")
        return ok()

    def equivalence(exp, nat):
        assert set(live) == {"G", "J"}
        assert exp.selector.engine is live["J"]
        checked.append("equivalence")
        return ok()

    def stress(exp):
        expected = {"B"} if exp.cond == "B13" else {"G", "J"}
        assert set(live) == expected
        assert exp.ctx.gen is live["B" if exp.cond == "B13" else "G"]
        assert exp.ctx.gen.engine is not None
        checked.append("stress_B" if exp.cond == "B13" else "stress_H")
        return ok()

    monkeypatch.setattr(preflight, "native_reference", native)
    monkeypatch.setattr(preflight, "selector_equivalence", equivalence)
    monkeypatch.setattr(preflight, "stress_test", stress)
    monkeypatch.setattr(preflight, "continuity_test", ok)
    monkeypatch.setattr(preflight, "batching_audit", ok)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda *args: ([1], "prompt"))
    s.p["PREFLIGHT_MODE"] = "off"
    s.preflight_before_engines()
    s.start_engines()
    s.preflight_after_engines()
    assert checked == ["native", "equivalence", "stress_H", "stress_B"]
    graph = s.preflight_results["eager_vs_graph"]
    assert graph["expanded_to_20"] and len(graph["rows"]) == 20 and not graph["blocking"]
    assert all(c["cleanup_confirmed"] for c in graph["process_cleanup"])
    assert set(live) == {"G", "J"} and s.ctx_H.gen is live["G"]
    assert s.ctx is s.ctx_H and s.ctx_B is None and s.p["PREFLIGHT_MODE"] == "off"
    s.shutdown()


def test_reload_full_192_plan_swaps_and_rewarm_outside_case_clock(reload_study, monkeypatch):
    s, live, clock = reload_study.study, reload_study.live, reload_study.clock
    intervals, warmed = [], set()

    def run(case, ctx, timeout):
        roles = {"B"} if case.condition.startswith("B13") else ({"G"} if case.condition == "G_SINGLE" else {"G", "J"})
        assert set(live) == roles
        gen_role = "B" if "B" in roles else "G"
        assert ctx.gen is live[gen_role] and ctx.gen.engine is not None
        if case.condition.startswith("J"):
            assert ctx.selector.engine is live["J"]
        key = (case.condition, s._reload_epoch)
        if case.seed == 0:
            warmed.add(key)
        else:
            assert key in warmed
        start = clock.now
        clock.now += 1
        if case.seed != 0:
            intervals.append((start, clock.now))
        return SimpleNamespace(run={"problem_id": case.problem_id, "condition": case.condition, "seed": case.seed,
            "profile": ctx.profile, "status": "ok", "public_output": "FINAL: 2\n", "raw_output": "FINAL: 2\n",
            "T_total": 1.0, "t_start_perf": start, "t_end_perf": clock.now})

    monkeypatch.setattr(algorithms, "run_case", run)
    s.start_engines()
    s.run()
    rows = read_jsonl(s.f["latency_metrics"])
    assert len(rows) == len(intervals) == 192
    assert Counter(r["condition"] for r in rows) == dict.fromkeys(s.CONDS_H + s.CONDS_B, 32)
    assert all(r["T_total"] == 1.0 and r["seed"] == 17 for r in rows)
    assert len(read_json(s.dir / "latency_plan.json")["items"]) == 32
    for swap in read_jsonl(s.f["swaps"]):
        assert all(swap["t_end_perf"] <= start or swap["t_start_perf"] >= end for start, end in intervals)
    s.shutdown()


def test_reload_builders_preserve_frozen_v2_and_no_inference_metadata(workspace, monkeypatch):
    builder, bundler = tool("build_notebooks"), tool("make_bundle")
    monkeypatch.setattr(builder, "NB", workspace / "notebooks")
    monkeypatch.setattr(bundler, "ROOT", workspace)
    monkeypatch.setattr(bundler, "DIST", workspace / "dist")
    frozen = {}
    for subdir, name in (("dist/v2", "jev_llm_v2_bundle.zip"), ("notebooks/v2", "08_estudio_latencia.ipynb")):
        path = workspace / subdir / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"frozen 0.2.0 bytes")
        frozen[path] = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["build_notebooks.py", "--series", "v2-reload"])
    builder.main()
    nb = builder.latency_notebook("v2-reload")
    params = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    env = {}
    exec(params, env)
    assert env["CONFIG_FILE"] == "config/experiment_v2_reload.json"
    assert env["ROOT"] == "/content/jev_llm_v2" and env["DATA_SUBDIR"] == "data/v2"
    assert env["LATENCY_SWAP_MODE"] == "reload"
    create = next(c.source for c in nb.cells if "exp = LatencyStudy" in c.source)
    assert "LATENCY_SWAP_MODE=LATENCY_SWAP_MODE" in create
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    for gold in (False, True):
        path = bundler.build("jev_llm_v2_analysis_bundle.zip" if gold else "jev_llm_v2_bundle.zip", gold, "v2-reload")
        assert path.parent == workspace / "dist/v2-reload"
        with zipfile.ZipFile(path) as z:
            assert {"config/experiment_v2.json", "config/experiment_v2_reload.json", "config/diagnostic_08b_a100.json"} <= set(z.namelist())
            if not gold:
                assert not any("gold" in n or "dataset_manifest" in n or "review" in n for n in z.namelist())
            assert json.loads(z.read("BUNDLE_MANIFEST.json"))["code_version"] == "0.2.1"
    assert all(path.read_bytes() == before for path, before in frozen.items())


def test_actual_frozen_020_bundle_and_config_remain_consistent():
    path = ROOT / "dist/v2/jev_llm_v2_bundle.zip"
    if not path.exists():
        pytest.skip("frozen bundle not present")
    with zipfile.ZipFile(path) as z:
        assert b'__version__ = "0.2.0"' in z.read("src/jevlab/__init__.py")
        assert z.read("config/experiment_v2.json") == (ROOT / "config/experiment_v2.json").read_bytes()


def test_amended_quality_graph_uses_checked_cleanup(reload_study, monkeypatch, workspace):
    s = reload_study.study
    s.start_engines()
    quality = Experiment(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2_reload.json", DATA_SUBDIR="data/v2",
                         CONDITION="JFINAL")
    quality.engines, quality.ctx = s.engines, s.ctx_H
    quality.paths, quality.toks = s.paths, s.toks
    quality.eos_ids, quality.newline_ids = s.eos_ids, s.newline_ids
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda *args: ([1], "prompt"))
    result = preflight.eager_vs_graph_v2(quality)
    assert result["expanded_to_20"] and len(result["process_cleanup"]) == 2
    assert all(r["cleanup_confirmed"] for r in result["process_cleanup"])
    assert quality.ctx.gen is reload_study.live["G"]
    quality.shutdown()
    assert not reload_study.live


def test_reload_graph_infrastructure_error_blocks(reload_study, monkeypatch):
    s = reload_study.study
    s.start_engines()
    ok = lambda *args: {"ok": True, "blocking": True, "summary": "fake check"}
    for name in ("selector_equivalence", "continuity_test", "batching_audit", "stress_test"):
        monkeypatch.setattr(preflight, name, ok)

    def failed_graph(exp):
        raise RuntimeError("engine cleanup or inference failed")

    monkeypatch.setattr(preflight, "eager_vs_graph", failed_graph)
    result = preflight.post_engine_checks(s, "full")
    assert result["eager_vs_graph"]["blocking"] and not result["eager_vs_graph"]["ok"]
    with pytest.raises(RuntimeError, match="blocking"):
        s._check_blocking(result)
    s.shutdown()


def test_failed_eager_constructor_cannot_load_unproven_replacement(reload_study, monkeypatch):
    s = reload_study.study
    s.start_engines()
    original = engines.Engine

    def constructor(name, *args, **kw):
        if name == "G_eager":
            raise RuntimeError("eager constructor failed without returning process handles")
        raise AssertionError("must not load a replacement after unproven constructor cleanup")

    monkeypatch.setattr(engines, "Engine", constructor)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda *args: ([1], "prompt"))
    with pytest.raises(RuntimeError, match="constructor failed"):
        preflight.eager_vs_graph_v2(s)
    assert set(reload_study.live) == {"J"}
    assert s.ctx_H.gen is None and "G" not in s.engines
    monkeypatch.setattr(engines, "Engine", original)
    s.shutdown()
