"""CPU-only postpilot loader regression checks; never publish actual notebooks."""

import ast
import inspect
import json
import subprocess
import sys

import nbformat
import pytest

from test_notebooks_v3 import (
    bundles, notebooks, public_full, publisher, rewrite, source,
)


@pytest.mark.parametrize("gold", [False, True])
def test_postpilot_loader_executes_real_full_bundle(public_full, tmp_path, gold):
    bundle = bundles.build(root=public_full, gold=gold)
    manifest = bundles.inspect_bundle(bundle, check_local=False)
    assert manifest["development_only"] is False
    assert manifest["operational_policy_id"] == bundles.OPERATIONAL_POLICY_ID
    target = tmp_path / "isolated-runtime"
    loader = notebooks.bundle_cell(gold=gold)
    compile(loader, "postpilot-loader", "exec")
    # A fresh interpreter exercises the actual embedded dependencies and imports.
    script = (
        f"ROOT = {str(target)!r}\nBUNDLE_FILE = {str(bundle)!r}\n"
        "SPLIT = 'test'\nRUN_TAG = 'main'\nSEEDS = '17,29,43'\n"
        + loader + "\n" + loader + "\n"
        "assert man['development_only'] is False\n"
        f"assert man['contains_gold'] is {gold!r}\n"
        "assert man['operational_policy_id'] == OPERATIONAL_POLICY_ID\n"
        "assert inspect_bundle(ZIP, check_local=False) == man\n"
        "assert all(sha((ROOT / name).read_bytes()) == digest for name, digest in man['files'].items())\n"
        "print('postpilot-loader-ok')\n"
    )
    result = subprocess.run([sys.executable, "-I", "-"], input=script,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "postpilot-loader-ok"
    assert json.loads((target / "BUNDLE_MANIFEST.json").read_bytes()) == manifest


@pytest.mark.parametrize("gold", [False, True])
@pytest.mark.parametrize("field", ["operational_policy_id", "operational_module_sha256", "source_mapping"])
def test_postpilot_loader_rejects_operational_binding_changes(public_full, tmp_path, gold, field):
    bundle = bundles.build(root=public_full, gold=gold)
    def corrupt(raw, manifest):
        if field == "source_mapping":
            member = "src/jevlab/v3/operational.py"
            manifest[field][member] = "src/jevlab/v3/foreign.py"
            manifest["source_sha256"][manifest[field][member]] = manifest["files"][member]
        else:
            manifest[field] = "wrong-policy" if field.endswith("id") else "0" * 64
    rewrite(bundle, corrupt)
    target = tmp_path / "rejected-runtime"
    with pytest.raises(ValueError, match="Operational policy/module source binding mismatch"):
        exec(notebooks.bundle_cell(gold=gold), dict(
            ROOT=str(target), BUNDLE_FILE=str(bundle), SPLIT="test", RUN_TAG="main", SEEDS="17,29,43"))
    assert not target.exists()


def test_postpilot_output_dir_preserves_originals_and_scientific_cells(tmp_path):
    root = tmp_path / "project"
    original = root / "notebooks/v3/01_G_SINGLE.ipynb"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"historical pilot notebook: must remain untouched\n")
    seal = root / "dist/v3/jev_llm_v3_bundle.zip"
    seal.parent.mkdir(parents=True)
    seal.write_bytes(b"historical seal sentinel\n")
    output = root / "notebooks/v3/postpilot"
    paths = notebooks.build(root=root, output_dir=output)
    assert len(paths) == 8 and all(p.parent == output for p in paths)
    assert original.read_bytes() == b"historical pilot notebook: must remain untouched\n"
    assert seal.read_bytes() == b"historical seal sentinel\n"
    first = [p.read_bytes() for p in paths]
    assert notebooks.build(root=root, output_dir=output) == paths
    assert [p.read_bytes() for p in paths] == first
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        expected = (notebooks.analysis_notebook() if "ANALYSIS" in path.name
                    else notebooks.condition_notebook(path.stem.split("_", 1)[1]))
        assert [c.source for c in nb.cells] == [c.source for c in expected.cells]
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                assert cell.outputs == [] and cell.execution_count is None
        loader = nb.cells[3].source
        assignments = {n.targets[0].id: ast.literal_eval(n.value)
                       for n in ast.parse(loader).body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
        assert assignments["OPERATIONAL_POLICY_ID"] == bundles.OPERATIONAL_POLICY_ID
        assert inspect.getsource(bundles.inspect_bundle) in loader
    with pytest.raises(ValueError, match="Refusing notebook source changes"):
        notebooks.build(root=root)
    assert original.read_bytes() == b"historical pilot notebook: must remain untouched\n"


def test_postpilot_cli_forwards_explicit_output_dir(monkeypatch, tmp_path, capsys):
    output = tmp_path / "notebooks/v3/postpilot"
    calls = []
    def build(*, output_dir):
        calls.append(output_dir)
        return [output_dir / "01_G_SINGLE.ipynb"]
    monkeypatch.setattr(notebooks, "build", build)
    notebooks.main(["--output-dir", str(output)])
    assert calls == [output]
    assert capsys.readouterr().out.strip() == str(output / "01_G_SINGLE.ipynb")
    assert not output.exists()
