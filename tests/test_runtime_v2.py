import ast
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from jevlab import __version__
from jevlab import engines, preflight, prompts
from jevlab.common import sha256_text
from jevlab.latency_study import LatencyStudy, build_items
from jevlab.runner import DEFAULTS, Experiment

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path):
    for name in ("config", "prompts"):
        shutil.copytree(ROOT / name, tmp_path / name)
    for subdir in ("data", "data/v2"):
        directory = tmp_path / subdir
        directory.mkdir(exist_ok=True)
        for split, count in (("test", 3), ("pilot", 2), ("dev", 20)):
            rows = [{"id": f"{split}-{i}", "problem": f"What is {i}+1?", "language": "en"} for i in range(count)]
            (directory / f"{split}_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (directory / f"{split}_gold.jsonl").write_text("secret gold\n")
        (directory / "schedule.json").write_text(json.dumps({"order": ["test-2", "test-0", "test-1"],
                                                           "pilot_order": ["pilot-1", "pilot-0"]}))
    return tmp_path


def experiment(root, **params):
    return Experiment(ROOT=str(root), CONFIG_FILE="config/experiment_v2.json", DATA_SUBDIR="data/v2", **params)


def test_config_models_aliases_and_old_defaults(workspace):
    cfg = json.loads((workspace / "config/experiment_v2.json").read_text())
    old = json.loads((workspace / "config/experiment.json").read_text())
    assert cfg["models"]["G"] == {"repo": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
    for role in ("J", "B"):
        assert cfg["models"][role] == old["models"][role]
    assert cfg["jevk5_runtime"] == old["jevk5_runtime"]
    assert cfg["sampling"] == old["sampling"]
    assert cfg["limits"] == old["limits"]
    assert cfg["conditions"] == old["conditions"]
    assert set(cfg["condition_labels"]) == set(cfg["conditions"])
    assert cfg["hardware"]["generator_copies"] == 1
    assert cfg["hardware"]["generator_max_num_seqs"] == 4
    assert cfg["software_pins"]["vllm"] == "0.30.0"
    for alias, code in cfg["condition_aliases"].items():
        exp = experiment(workspace, CONDITION=alias)
        assert exp.cond == code
        assert exp.p["KV_CACHE_GB_G"] == 3.0
        assert exp.p["KV_CACHE_GB_J"] == 4.0
        assert exp.p["REQUIRED_GPU"] == "A100"
        assert exp.config["code_version"] == __version__ == "0.2.1"
    old_exp = Experiment(ROOT=str(workspace))
    assert old_exp.data_dir == workspace / "data"
    assert old_exp.cfg["models"]["G"] == old["models"]["G"]
    assert old_exp.p["KV_CACHE_GB_G"] == DEFAULTS["KV_CACHE_GB_G"] == 2.0
    assert DEFAULTS["CONFIG_FILE"] == "config/experiment.json"
    assert DEFAULTS["DATA_SUBDIR"] == "data"


@pytest.mark.parametrize("split,want", [("test", ["test-2", "test-0"]), ("pilot", ["pilot-1", "pilot-0"]),
                                      ("dev", ["dev-0", "dev-1"])])
def test_path_split_schedule_and_no_gold(workspace, split, want):
    exp = experiment(workspace, SPLIT=split, N_PROBLEMS=2)
    assert [r["id"] for r in exp.problems()] == want
    assert len(exp.dev_problems()) == 20
    assert not any("gold" in k for k in exp.config["data_sha256"])
    original = exp.chash
    (workspace / "data/v2/test_gold.jsonl").write_text("different secret gold\n")
    assert experiment(workspace, SPLIT=split, N_PROBLEMS=2).chash == original
    row = {"id": "leak", "problem": "1+1?", "language": "en", "gold_numerator": 2}
    (workspace / f"data/v2/{split}_inputs.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="no gold"):
        exp.problems()


def test_hash_and_invalid_paths(workspace):
    exp = experiment(workspace)
    assert experiment(workspace, SPLIT="pilot").chash != exp.chash
    cfg_path = workspace / "config/experiment_v2.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["condition_labels"]["J64"] += " changed"
    cfg_path.write_text(json.dumps(cfg))
    assert experiment(workspace).chash != exp.chash
    for key in ("CONFIG_FILE", "DATA_SUBDIR"):
        with pytest.raises(ValueError, match="relative"):
            Experiment(ROOT=str(workspace), **{key: "../outside"})
    with pytest.raises(ValueError, match="SPLIT"):
        experiment(workspace, SPLIT="test_v2")
    exp = experiment(workspace, ABORT_ON_PREFLIGHT_FAIL=False)
    with pytest.raises(RuntimeError, match="blocking"):
        exp._check_blocking({"stress": {"ok": False, "blocking": True}})


def test_latency_counts_defaults_and_smoke(workspace):
    full = LatencyStudy(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2.json", DATA_SUBDIR="data/v2")
    smoke = LatencyStudy(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2.json", DATA_SUBDIR="data/v2",
                         LAT_N_DEV=2, LAT_SYNTH_PER_BAND=1, RUN_TAG="smoke")
    assert full.p["SPLIT"] == smoke.p["SPLIT"] == "dev"
    assert len(build_items(full.dev_problems(), full.p["LAT_N_DEV"], full.p["LAT_SYNTH_PER_BAND"])) == 32
    assert len(build_items(smoke.dev_problems(), smoke.p["LAT_N_DEV"], smoke.p["LAT_SYNTH_PER_BAND"])) == 6
    assert full.chash != smoke.chash
    with pytest.raises(ValueError, match="level 2"):
        LatencyStudy(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2.json", SLEEP_LEVEL=2)
    with pytest.raises(ValueError):
        build_items(full.dev_problems(), -1, 3)


@pytest.mark.parametrize("series", ["v2", "v2-reload"])
def test_generated_forms_extract_and_pass_parameters(workspace, series):
    builder = tool("build_notebooks")
    notebooks = [builder.condition_notebook(c, series) for c in builder.COND_INFO]
    notebooks += [builder.latency_notebook(series), builder.analysis_notebook(series)]
    for nb in notebooks:
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, "generated_notebook", "exec")
        cell = next(c for c in nb.cells if "parameters" in c.metadata.get("tags", []))
        assigned = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(cell.source).body
                    if isinstance(node, ast.Assign)}
        assert assigned["CONFIG_FILE"] == ("config/experiment_v2_reload.json" if series == "v2-reload" else "config/experiment_v2.json")
        assert assigned["DATA_SUBDIR"] == "data/v2"
        assert "/content/jev_llm_v2" in "\n".join(c.source for c in nb.cells)
        if "CONDITION" not in assigned:
            assert 'ROOT / DATA_SUBDIR / f"{SPLIT}_gold.jsonl"' in "\n".join(c.source for c in nb.cells)
            continue
        assert assigned["ROOT"] == "/content/jev_llm_v2"
        assert assigned["KV_CACHE_GB_G"] == 3.0
        runtime = next(c for c in nb.cells if c.cell_type == "code" and "exp = " in c.source)
        tree = ast.parse(runtime.source)
        passed = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict")
        assert {k.arg for k in passed.keywords} <= assigned.keys()
        params = {k.arg: assigned[k.arg] for k in passed.keywords}
        params["ROOT"] = str(workspace)
        cls = LatencyStudy if assigned["CONDITION"] == "LATENCY" else Experiment
        assert cls(**params).data_dir == workspace / "data/v2"
    old_params = builder.params_cell("G_SINGLE", "L4")
    assert 'CONFIG_FILE = "config/experiment.json"' in old_params
    assert 'DATA_SUBDIR = "data"' in old_params


def test_builders_isolate_outputs_and_bundle_gold(workspace, monkeypatch):
    builder, bundler = tool("build_notebooks"), tool("make_bundle")
    monkeypatch.setattr(builder, "NB", workspace / "notebooks")
    monkeypatch.setattr(bundler, "ROOT", workspace)
    monkeypatch.setattr(bundler, "DIST", workspace / "dist")
    for directory in ("notebooks", "dist"):
        (workspace / directory).mkdir()
        (workspace / directory / "original").write_text("unchanged")
    monkeypatch.setattr(sys, "argv", ["build_notebooks.py", "--series", "v2"])
    builder.main()
    assert len(list((workspace / "notebooks/v2").glob("*.ipynb"))) == 8
    assert (workspace / "notebooks/original").read_text() == "unchanged"
    for gold in (False, True):
        path = bundler.build("analysis.zip" if gold else "inference.zip", gold, "v2")
        assert path.parent == workspace / "dist/v2"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert not any(n.startswith("data/") and not n.startswith("data/v2/") for n in names)
            assert "config/experiment_v2.json" in names
            assert "config/experiment.json" not in names
            assert any(n.endswith("_gold.jsonl") for n in names) == gold
            manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
            assert manifest["contains_gold"] == gold
            for name, digest in manifest["files"].items():
                import hashlib
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert (workspace / "dist/original").read_text() == "unchanged"


def test_selector_reference_cannot_pass_missing_rows(workspace, monkeypatch):
    exp = experiment(workspace)
    monkeypatch.setattr(preflight, "selector_items", lambda exp: [{"state": {}, "question": {"criteria": {}}}])
    result = preflight.selector_equivalence(exp, {"ok": True, "native": {"items": [], "temperature": 1.316}})
    assert not result["ok"] and result["blocking"]


@pytest.mark.parametrize("eager_token,expected_count", [(1, 3), (2, 20)])
def test_graph_expansion_is_nonblocking_and_one_copy(workspace, monkeypatch, eager_token, expected_count):
    exp = experiment(workspace)
    active = []

    class FakeEngine:
        def __init__(self, name, path, tok, enforce_eager, **kwargs):
            assert not active
            active.append(self)
            self.eager = enforce_eager
            self.engine_kwargs = {"enable_sleep_mode": False}

        def new_id(self, tag):
            return tag

        def run(self, requests):
            return {r.rid: SimpleNamespace(token_ids=[eager_token if self.eager else 1] * 64) for r in requests}

        def shutdown(self):
            active.remove(self)

    monkeypatch.setattr(engines, "Engine", FakeEngine)
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda *args: ([1], "prompt"))
    exp.paths, exp.toks, exp.eos_ids, exp.newline_ids = {"G": "unused"}, {"G": object()}, [0], {1}
    exp.engines["G"] = FakeEngine("G", "unused", None, False)
    exp.ctx = SimpleNamespace(gen=exp.engines["G"])
    result = preflight.eager_vs_graph(exp)
    assert result["blocking"] is False
    assert len(result["rows"]) == expected_count
    assert result["expanded_to_20"] == (expected_count == 20)
    assert (exp.dir / "preflight/eager_vs_graph.json").is_file()
    assert exp.ctx.gen is exp.engines["G"]
    assert len(active) == 1
    exp.shutdown()


def test_thinking_is_disabled_explicitly():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return prompts.THINK_OFF_SUFFIX

    text = prompts.render_generator_prompt(Tokenizer(), "system", "problem")
    assert text.endswith("<think>\n\n</think>\n\n")
    assert sha256_text(text)


@pytest.mark.parametrize("native_logits,vllm_logits,want", [([10, 9.9375, 0, 0], [0, 0, 1, 0], False),
                                                          ([10, 9.9375, 0, 0], [0, 1, 0, 0], True),
                                                          ([10, 9, 0, 0], [0, 1, 0, 0], False),
                                                          ([10, 9, 0, 0], [1, 0, 0, 0], True)])
def test_selector_bf16_tolerance_does_not_accept_unrelated_options(workspace, monkeypatch, native_logits, vllm_logits, want):
    import hashlib

    exp = experiment(workspace)
    ids = [1, 2, 3]
    monkeypatch.setattr(preflight, "selector_items", lambda exp: [{"state": {}, "question": {"criteria": dict.fromkeys(range(4))}}])
    monkeypatch.setattr(prompts, "selector_prompt_ids", lambda *args: (ids, "prompt"))
    exp.toks = {"J": object()}
    exp.selector = SimpleNamespace(letter_logits_from_ids=lambda *args: (vllm_logits, 0),
                                   probabilities=lambda logits: [0.7 if x == max(logits) else 0.1 for x in logits])
    reference = {"ok": True, "native": {"temperature": 1.316, "items": [
        {"ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "raw_logits": native_logits,
         "probabilities": [0.7, 0.1, 0.1, 0.1], "seconds": 0}]}}
    result = preflight.selector_equivalence(exp, reference)
    assert result["ok"] == want
    assert result["blocking"]


@pytest.mark.parametrize("missing_request", [False, True])
def test_stress_requires_four_full_outputs_and_near_limit_selector(workspace, monkeypatch, missing_request):
    exp = experiment(workspace, CONDITION="JFINAL")
    monkeypatch.setattr(engines, "sampling_params", lambda **kw: kw)
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda *args: ([1, 2], "prompt"))
    monkeypatch.setattr(prompts, "selector_prompt_ids", lambda tok, state, question:
                        ([1] * (100 + question["criteria"]["option_0"].count("\n") * 15), "prompt"))

    def run(reqs):
        return {r.rid: SimpleNamespace(token_ids=[1] * r.params["max_tokens"])
                for r in (reqs[:-1] if missing_request else reqs)}

    names = iter(range(4))
    exp.ctx = SimpleNamespace(tok=object(), gen=SimpleNamespace(new_id=lambda tag: str(next(names)), run=run))
    exp.toks = {"J": object()}
    exp.selector = SimpleNamespace(letter_logits_from_ids=lambda ids, n: ([0, 1, 2, 3], 0))
    result = preflight.stress_test(exp)
    assert result["ok"] == (not missing_request)
    assert result["blocking"]
    assert 16320 <= result["selector"]["input_tokens"] <= 16384
    assert result["nvml_peak_gib"] is None


def test_v2_off_cannot_skip_mandatory_checks(workspace, monkeypatch):
    exp = experiment(workspace, CONDITION="JFINAL")
    check = lambda *args: {"ok": True, "blocking": True, "summary": "mocked"}
    for name in ("parser_selftest", "lock_check", "template_check", "prompt_length_check", "sampling_semantics_check", "tokenizer_check"):
        monkeypatch.setattr(preflight, name, check)
    called = []
    monkeypatch.setattr(preflight, "native_reference", lambda exp: called.append("native") or check())
    preflight.pre_engine_checks(exp, "off")
    assert called == ["native"]
    exp.selector = object()
    monkeypatch.setattr(preflight, "selector_equivalence", lambda *args: called.append("equivalence") or check())
    monkeypatch.setattr(preflight, "stress_test", lambda *args: called.append("stress") or check())
    monkeypatch.setattr(preflight, "eager_vs_graph", lambda *args: called.append("graph") or {"ok": False, "blocking": False})
    result = preflight.post_engine_checks(exp, "off")
    assert called == ["native", "equivalence", "graph", "stress"]
    assert result["stress"]["blocking"]
    assert not result["eager_vs_graph"]["blocking"]


def test_bundle_loader_preserves_root_and_requires_correct_gold(tmp_path):
    builder = tool("build_notebooks")
    tmp_path.joinpath("BUNDLE_MANIFEST.json").write_text(json.dumps({
        "series": "v2", "contains_gold": False, "bundle_version": "test", "created_utc": "test", "files": {}}))
    nb = builder.condition_notebook("J64", "v2")
    loader = next(c.source for c in nb.cells if c.cell_type == "code" and "BUNDLE_MANIFEST.json" in c.source)
    env = {"ROOT": str(tmp_path)}
    exec(loader, env)
    assert env["ROOT"] == tmp_path
    assert env["expected_gold"] is False
    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_gold.jsonl").write_text("secret")
    with pytest.raises(SystemExit, match="contains gold"):
        exec(loader, {"ROOT": str(tmp_path)})


def test_latency_prompt_limit_checks_baseline_and_synthetics(workspace, monkeypatch):
    exp = LatencyStudy(ROOT=str(workspace), CONFIG_FILE="config/experiment_v2.json", DATA_SUBDIR="data/v2")
    exp.toks = {"G": "generator", "B": "baseline"}
    monkeypatch.setattr(prompts, "generator_prompt_ids", lambda tok, *args: ([1] * (513 if tok == "baseline" else 100), "prompt"))
    result = preflight.prompt_length_check(exp)
    assert not result["ok"] and result["blocking"]
    assert len(result["over"]) == 32
    assert all(key.startswith("B:") for key in result["over"])
