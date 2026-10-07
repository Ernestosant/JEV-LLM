"""Offline notebook/bundle checks. Fixtures never authorize actual dataset publication."""

import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_notebooks_v3 as notebooks
import make_bundle_v3 as bundles


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "project"
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src/jevlab").glob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "src/jevlab/v3").rglob("*.py")]
    names += [bundles.CONFIG, "tools/verify_run_v3.py", "data/dev_inputs.jsonl", "data/v3/REVIEW_CONTRACT.md",
              "data/v3/ELIGIBILITY_POLICY.md", "data/v3/staging/review_policy.json",
              "data/build_dataset_v3.py", "data/build_dataset.py"]
    names += list(bundles.TOOL_SOURCES)
    names += [f"prompts/{n}.txt" for n in ("generador", "criterio_paso", "criterio_final")]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    cfg_path = root / bundles.CONFIG
    cfg = json.loads(cfg_path.read_bytes())
    policy = json.loads((root / "data/v3/staging/review_policy.json").read_bytes())
    cfg["dataset_review_policy"] = {"id": policy["policy_id"], "sha256": policy["contract_sha256"]}
    cfg_path.write_bytes(bundles.encoded(cfg))
    prep = root / "data/v3/staging/preparation_manifest.json"
    prep.write_bytes(bundles.encoded({"source_sha256": {}, "previous_artifact_sha256": {},
                                    "source_lock_path": bundles.CONFIG, "tokenizers": {}}))
    return root


@pytest.fixture
def publisher():
    spec = importlib.util.spec_from_file_location("packaging_publisher_schema", ROOT / "data/build_dataset_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite(path, change):
    with zipfile.ZipFile(path) as archive:
        raw = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(raw["BUNDLE_MANIFEST.json"])
    change(raw, manifest)
    raw["BUNDLE_MANIFEST.json"] = bundles.encoded(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in raw.items():
            info = zipfile.ZipInfo(name)
            info.filename = name  # Windows ZipInfo normalizes backslashes at construction.
            archive.writestr(info, content)


def test_eight_isolated_valid_notebooks(tmp_path):
    paths = notebooks.build(output_dir=tmp_path / "notebooks")
    assert [p.name for p in paths] == [f"{i:02d}_{c}.ipynb" for i, c in enumerate(notebooks.CONDITIONS, 1)] + ["07_ANALYSIS.ipynb", "08_LATENCY.ipynb"]
    first = [p.read_bytes() for p in paths]
    notebooks.build(output_dir=tmp_path / "notebooks")
    assert [p.read_bytes() for p in paths] == first
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        params = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
        assert len(params) == 1
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(params[0].source).body if isinstance(n, ast.Assign)}
        assert values["CONFIG_FILE"] == bundles.CONFIG
        assert values["DATA_SUBDIR"] == "data/v3"
        assert values["ROOT"] == "/content/jev_llm_v3"
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs and cell.execution_count is None
        text = "\n".join(c.source for c in nb.cells)
        assert "from jevlab.runner import" not in text
        assert "from jevlab.latency_study import" not in text
        assert "/content/jev_llm_v2" not in text
        if "ANALYSIS" in path.name:
            assert "accelerator" not in nb.metadata
            assert "run_analysis as run_analysis_v3" in text
            assert "approval=Path(APPROVAL_FILE)" in text
        else:
            condition = values["CONDITION"]
            assert values["N_PROBLEMS"] == 500
            assert values["SEEDS"] == ("17" if condition.endswith("GREEDY") else "17,29,43")
            assert re.match(r"^(?:[0-9]+_)?" + condition + r"(?:_|\.)", path.stem + ".out.main.ipynb")
            assert "finally:\n    exp.shutdown()" in text
            assert notebooks.INSTALL in text
            if condition == "LATENCY":
                assert values["LAT_N_ITEMS"] * values["LAT_REPETITIONS"] * 3 * 2 == 1800
                assert values["LAT_CONDITIONS"] == "JFINAL,B13_GREEDY"
                assert values["LATENCY_SWAP_MODE"] == "reload"


def test_dev_api_source_mapping_and_no_fake_full_data(source):
    path = bundles.build(dev=True, root=source)
    manifest = bundles.inspect_bundle(path, root=source)
    assert manifest["series"] == "v3" and manifest["code_version"] == "0.3.0"
    assert manifest["legacy_code_version"] == "0.2.1"
    assert manifest["development_only"] and not manifest["contains_gold"]
    assert manifest["source_mapping"]["data/v3/dev_inputs.jsonl"] == "data/dev_inputs.jsonl"
    assert set(n for n in manifest["files"] if n.startswith("data/")) == {"data/v3/dev_inputs.jsonl", "data/v3/dataset_manifest.json"}
    assert not (source / "data/v3/test_inputs.jsonl").exists()
    assert bundles.build(dev=True, root=source) == path
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("data/v3/dataset_manifest.json"))
        assert metadata["sealed"] is False and metadata["actual_n"] == {"dev": 20}
        assert "tools/verify_run_v3.py" in archive.namelist()
    assert bundles.inspect_bundle(path, check_local=False) == manifest


def test_explicit_staging_dev_preferred(source):
    original = source / "data/v3/staging/dev_inputs.jsonl"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes((source / "data/dev_inputs.jsonl").read_bytes())
    manifest = bundles.inspect_bundle(bundles.build(dev=True, root=source), root=source)
    assert manifest["source_mapping"]["data/v3/dev_inputs.jsonl"] == "data/v3/staging/dev_inputs.jsonl"


def test_full_requires_actual_dataset_seal_and_dev_never_gold(source):
    with pytest.raises(ValueError, match="SHA256SUMS"):
        bundles.validate_dataset(source)
    with pytest.raises(ValueError, match="SHA256SUMS"):
        bundles.build(root=source)
    with pytest.raises(ValueError, match="never contain gold"):
        bundles.build(dev=True, gold=True, root=source)
    assert not (source / "dist").exists()


def test_unsealed_extra_member_refused_by_actual_dataset_api(source):
    directory = source / "data/v3"
    digest = bundles.sha((directory / "REVIEW_CONTRACT.md").read_bytes())
    (directory / "SHA256SUMS").write_text(f"{digest}  REVIEW_CONTRACT.md\n")
    (directory / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unsealed"):
        bundles.validate_dataset(source)


@pytest.mark.parametrize("change,match", [
    (lambda raw, m: m.update(series="v2"), "series/code"),
    (lambda raw, m: m.update(code_version="0.2.1"), "series/code"),
    (lambda raw, m: m["files"].pop("tools/verify_run_v3.py"), "hashes"),
    (lambda raw, m: raw.update({"extra.txt": b"extra"}), "hashes"),
    (lambda raw, m: raw.update({"../escape": b"x"}), "Unsafe"),
    (lambda raw, m: raw.update({"C:/escape": b"x"}), "Unsafe"),
    (lambda raw, m: raw.update({"data\\escape": b"x"}), "Unsafe"),
    (lambda raw, m: raw.update({"prompts/generador.txt": b"altered"}), "hash mismatch"),
    (lambda raw, m: m.update(config_sha256="a" * 64), "pin mismatch"),
    (lambda raw, m: m["source_mapping"].update({"data/v3/dev_inputs.jsonl": "../dev"}), "Unsafe"),
])
def test_strict_archive_rejections(source, change, match):
    path = bundles.build(dev=True, root=source)
    rewrite(path, change)
    with pytest.raises(ValueError, match=match):
        bundles.inspect_bundle(path, check_local=False)


@pytest.mark.parametrize("name", ["data/v3/test_gold.jsonl", "data/v3/solutions.jsonl", "data/v3/staging/reviews/evidence.jsonl"])
def test_inference_rejects_even_hashed_private_payload(source, name):
    path = bundles.build(dev=True, root=source)
    def add(raw, m):
        raw[name] = b"private payload"
        m["files"][name] = bundles.sha(raw[name])
    rewrite(path, add)
    with pytest.raises(ValueError, match="Unapproved"):
        bundles.inspect_bundle(path, check_local=False)


def test_gold_hidden_in_inputs_rejected(source):
    path = source / "data/dev_inputs.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    rows[0]["solution"] = "not blind"
    path.write_bytes(bundles.encoded(rows[0]) + b"".join((json.dumps(r) + "\n").encode() for r in rows[1:]))
    with pytest.raises(ValueError):
        bundles.build(dev=True, root=source)


def test_dev_source_changes_require_explicit_development_rebuild(source):
    path = bundles.build(dev=True, root=source)
    before = path.read_bytes()
    changed = source / "src/jevlab/v3/runner.py"
    changed.write_bytes(changed.read_bytes() + b"\n# local change\n")
    with pytest.raises(ValueError, match="local source"):
        bundles.inspect_bundle(path, root=source)
    assert bundles.build(dev=True, root=source) == path
    assert path.read_bytes() != before
    assert bundles.inspect_bundle(path, root=source)["development_only"] is True


def test_duplicate_and_symlink_zip_members(source):
    path = bundles.build(dev=True, root=source)
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning):
            archive.writestr("BUNDLE_MANIFEST.json", b"{}")
    with pytest.raises(ValueError, match="Duplicate"):
        bundles.inspect_bundle(path, check_local=False)
    path.unlink()
    path = bundles.build(dev=True, root=source)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "outside")
    with pytest.raises(ValueError, match="symlink"):
        bundles.inspect_bundle(path, check_local=False)


def test_embedded_loader_rejects_stale_root_and_unknown_series(source, tmp_path):
    path = bundles.build(dev=True, root=source)
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "old.py").write_text("stale")
    env = dict(ROOT=str(target), BUNDLE_FILE=str(path), SPLIT="dev", RUN_TAG="smoke", SEEDS="17")
    with pytest.raises(ValueError, match="Stale/foreign"):
        exec(notebooks.bundle_cell(), env)
    rewrite(path, lambda raw, m: m.update(series="v1"))
    with pytest.raises(ValueError, match="series/code"):
        exec(notebooks.bundle_cell(), env)


def test_embedded_loader_empty_root_loads_exact_dev_bundle(source, tmp_path, monkeypatch):
    path = bundles.build(dev=True, root=source)
    target = tmp_path / "runtime"
    for name in ("jevlab", "jevlab.v3"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", sys.path[:])
    env = dict(ROOT=str(target), BUNDLE_FILE=str(path), SPLIT="dev", RUN_TAG="smoke", SEEDS="17")
    exec(notebooks.bundle_cell(), env)
    exec(notebooks.bundle_cell(), env)
    (target / "data/v3/hidden_gold.jsonl").write_text("private")
    with pytest.raises(ValueError, match="Extra/stale"):
        exec(notebooks.bundle_cell(), env)
    for name in ("jevlab", "jevlab.v3"):
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture
def public_full(source, monkeypatch, publisher):
    """Synthetic packaging fixture only; bypasses no real dataset publication gate."""
    monkeypatch.setattr(bundles, "_DATASET_CERTIFICATES", {})
    domains = publisher.DOMAINS
    test = [{"id": f"fixture-{i}", "problem": f"Packaging fixture number {i}, not a real dataset.", "language": "en"} for i in range(500)]
    pilot = [{"id": f"pilot-fixture-{i}", "problem": "Packaging fixture only.", "language": "en"} for i in range(50)]
    labeled = {}
    for split, rows, per_domain in (("test", test, 100), ("pilot", pilot, 10)):
        labeled[split] = [{**r, "domain": domains[i // per_domain],
            "sampling_tier": "easy" if i % per_domain < per_domain * 3 // 10 else
                             "medium" if i % per_domain < per_domain * 7 // 10 else "hard",
            "difficulty": "easy" if i % 2 == 0 else "medium",
            "gold_numerator": 1, "gold_denominator": 1, "reference_solution": "Synthetic fixture only"}
            for i, r in enumerate(rows)]
    gold = labeled["test"]
    latency_ids = publisher.latency_subset(gold)
    by_id = {r["id"]: r for r in test}
    subset = [by_id[pid] for pid in latency_ids]
    directory = source / "data/v3"
    (directory / "review_policy.json").write_bytes((directory / "staging/review_policy.json").read_bytes())
    cfg = json.loads((source / bundles.CONFIG).read_bytes())
    policy = json.loads((directory / "review_policy.json").read_bytes())
    cfg["dataset_review_policy"] = {"id": policy["policy_id"], "sha256": policy["contract_sha256"]}
    (source / bundles.CONFIG).write_bytes(bundles.encoded(cfg))
    for name, value in (("test_inputs.jsonl", test), ("pilot_inputs.jsonl", pilot), ("latency_inputs.jsonl", subset),
                        ("test_gold.jsonl", gold), ("pilot_gold.jsonl", labeled["pilot"]), ("dev_gold.jsonl", [])):
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in value))
    (directory / "dev_inputs.jsonl").write_bytes((source / "data/dev_inputs.jsonl").read_bytes())
    schedule = publisher.schedule_for(gold, labeled["pilot"])
    (directory / "schedule.json").write_bytes(bundles.encoded(schedule))
    (directory / "latency_plan.json").write_bytes(bundles.encoded(publisher.latency_plan_for(gold, latency_ids, schedule)))
    seals = {p.name: bundles.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
    public = publisher.public_manifest(
        {"builder_sha256": bundles.sha((source / "data/build_dataset_v3.py").read_bytes()),
         "review_policy_json_sha256": seals["review_policy.json"], "sources": {}},
        {n: seals[n] for n in ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "latency_inputs.jsonl", "schedule.json", "latency_plan.json")},
        json.loads((directory / "review_policy.json").read_bytes()), analysis_hash="c" * 64, complete=True,
        intrinsic_counts={split: {d: sum(r["difficulty"] == d for r in rows) for d in publisher.TEST_QUOTA}
                          for split, rows in labeled.items()})
    (directory / "dataset_manifest.json").write_bytes(bundles.encoded(public))
    seals["dataset_manifest.json"] = bundles.sha((directory / "dataset_manifest.json").read_bytes())
    (directory / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in seals.items()))
    def synthetic_validation(root):
        assert root == source
        current_seals = {line.split()[1]: line.split()[0]
                         for line in (directory / "SHA256SUMS").read_text().splitlines()}
        return {"manifest": json.loads((directory / "dataset_manifest.json").read_bytes()),
                "seals": current_seals, "lineage_inventory": {}}
    synthetic_validation.__wrapped__ = bundles._validate_dataset_full
    monkeypatch.setattr(bundles, "_validate_dataset_full", synthetic_validation)
    return source


def test_actual_public_bytes_and_seal_shared_gold_visibility_recursive_sources(public_full):
    root = public_full
    nested = root / "src/jevlab/v3/submodule/fixture.py"
    nested.parent.mkdir()
    nested.write_text("# packaging fixture\n")
    inference = bundles.build(root=root)
    analysis = bundles.build(root=root, gold=True)
    bundles.inspect_bundle(inference, root=root)
    bundles.inspect_bundle(analysis, root=root)
    with zipfile.ZipFile(inference) as a, zipfile.ZipFile(analysis) as b:
        assert "src/jevlab/v3/submodule/fixture.py" in a.namelist()
        assert not any(n.endswith("_gold.jsonl") or "/staging/" in n for n in a.namelist())
        assert all("data/v3/" + n in b.namelist() for n in bundles.GOLD_DATA)
        for name in bundles.PUBLIC_DATA:
            assert a.read("data/v3/" + name) == b.read("data/v3/" + name)
            assert a.read("data/v3/" + name) == (root / "data/v3" / name).read_bytes()
        manifest = json.loads(a.read("data/v3/dataset_manifest.json"))
        assert "review_evidence_sha256" not in manifest
        assert "reference_solution" not in a.read("data/v3/latency_plan.json").decode()
        assert "agent_reviewed_all" in a.read("data/v3/dataset_manifest.json").decode()


def test_full_bundle_is_immutable_after_source_change(public_full):
    path = bundles.build(root=public_full)
    before = path.read_bytes()
    changed = public_full / "src/jevlab/v3/runner.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed packaging fixture\n")
    with pytest.raises(ValueError, match="certificate bytes/path inventory changed"):
        bundles.inspect_bundle(path, root=public_full)
    with pytest.raises(ValueError, match="certificate bytes/path inventory changed"):
        bundles.build(root=public_full)
    assert path.read_bytes() == before


def test_full_bundle_archive_existing_preserves_original_and_packaged_bytes(public_full):
    path = bundles.build(root=public_full)
    before = path.read_bytes()
    original = bundles.inspect_bundle(path, check_local=False)
    changed = public_full / "tools/colab/run_v3.py"
    assert "tools/colab/run_v3.py" not in original["files"]
    changed.write_bytes(changed.read_bytes() + b"\n# changed packaging fixture\n")
    bundles._DATASET_CERTIFICATES.clear()
    with pytest.raises(ValueError, match="Refusing to overwrite frozen bundle"):
        bundles.build(root=public_full)
    assert path.read_bytes() == before
    assert not (path.parent / "archive").exists()
    assert bundles.build(root=public_full, archive_existing=True) == path
    archived = path.parent / "archive" / f"{path.stem}_{hashlib.sha256(before).hexdigest()}.zip"
    assert list(archived.parent.iterdir()) == [archived]
    assert archived.read_bytes() == before
    rebuilt = bundles.inspect_bundle(path, root=public_full)
    assert rebuilt["files"] == original["files"]
    assert rebuilt["source_sha256"]["tools/colab/run_v3.py"] == bundles.sha(changed.read_bytes())
    assert rebuilt["source_sha256"] != original["source_sha256"]
    with zipfile.ZipFile(archived) as old, zipfile.ZipFile(path) as new:
        assert old.namelist() == new.namelist()
        assert all(old.read(name) == new.read(name) for name in original["files"])


def test_full_bundle_archive_existing_rejects_packaged_payload_change(public_full):
    path = bundles.build(root=public_full)
    before = path.read_bytes()
    changed = public_full / "src/jevlab/v3/runner.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed packaging fixture\n")
    bundles._DATASET_CERTIFICATES.clear()
    with pytest.raises(ValueError, match="must preserve every packaged experiment byte"):
        bundles.build(root=public_full, archive_existing=True)
    assert path.read_bytes() == before
    assert not (path.parent / "archive").exists()


@pytest.mark.parametrize("field", ["id", "sha256"])
def test_full_bundle_requires_current_parent_config_policy_pin(public_full, field):
    path = bundles.build(root=public_full)
    def corrupt(raw, manifest):
        cfg = json.loads(raw[bundles.CONFIG])
        cfg["dataset_review_policy"][field] = "historical-policy" if field == "id" else "0" * 64
        raw[bundles.CONFIG] = bundles.encoded(cfg)
        manifest["files"][bundles.CONFIG] = bundles.sha(raw[bundles.CONFIG])
        manifest["source_sha256"][manifest["source_mapping"][bundles.CONFIG]] = bundles.sha(raw[bundles.CONFIG])
        manifest["config_sha256"] = bundles.sha(raw[bundles.CONFIG])
    rewrite(path, corrupt)
    with pytest.raises(ValueError, match="review policy pin mismatch"):
        bundles.inspect_bundle(path, check_local=False)


def test_current_publisher_public_schema_and_balanced_plan_package_without_projection(public_full, publisher):
    """Publisher API schema fixture, not actual reviews or real dataset publication."""
    directory = public_full / "data/v3"
    rows = [json.loads(line) for line in (directory / "test_inputs.jsonl").read_text().splitlines()]
    pilot = [json.loads(line) for line in (directory / "pilot_inputs.jsonl").read_text().splitlines()]
    labeled = [json.loads(line) for line in (directory / "test_gold.jsonl").read_text().splitlines()]
    labeled_pilot = [json.loads(line) for line in (directory / "pilot_gold.jsonl").read_text().splitlines()]
    schedule = publisher.schedule_for(labeled, labeled_pilot)
    assert schedule["order"] != [r["id"] for r in rows]
    subset = [json.loads(line)["id"] for line in (directory / "latency_inputs.jsonl").read_text().splitlines()]
    plan = publisher.latency_plan_for(labeled, subset, schedule)
    assert all(set(i) == {"item_id", "problem", "domain", "difficulty", "sampling_tier"} for i in plan["items"])
    assert plan["quota_axis"] == "source_sampling_tier"
    assert plan["intrinsic_difficulty_counts"]["hard"] == 0
    (directory / "schedule.json").write_bytes(bundles.encoded(schedule))
    (directory / "latency_plan.json").write_bytes(bundles.encoded(plan))
    policy = json.loads((directory / "review_policy.json").read_bytes())
    hashes = {n: bundles.sha((directory / n).read_bytes()) for n in
              ("test_inputs.jsonl", "pilot_inputs.jsonl", "dev_inputs.jsonl", "latency_inputs.jsonl", "schedule.json", "latency_plan.json")}
    metadata = {"builder_sha256": "a" * 64, "review_policy_json_sha256": bundles.sha((directory / "review_policy.json").read_bytes()),
                "sources": {"synthetic_schema": {"repo": "synthetic/source", "revision": "b" * 40, "license": "unit fixture only"}}}
    counts = {split: {d: sum(r["difficulty"] == d for r in values) for d in publisher.TEST_QUOTA}
              for split, values in (("test", labeled), ("pilot", labeled_pilot))}
    with pytest.raises(ValueError, match="intrinsic difficulty counts"):
        publisher.public_manifest(metadata, hashes, policy, analysis_hash="c" * 64, complete=True)
    public = publisher.public_manifest(metadata, hashes, policy, analysis_hash="c" * 64, complete=True, intrinsic_counts=counts)
    assert public["review"]["intrinsic_difficulty_counts"] == {
        "test": {"easy": 250, "medium": 250, "hard": 0}, "pilot": {"easy": 25, "medium": 25, "hard": 0}}
    assert public["review"]["source_sampling_tier_is_intrinsic_difficulty"] is False
    (directory / "dataset_manifest.json").write_bytes(bundles.encoded(public))
    seals = {p.name: bundles.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
    (directory / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in seals.items()))
    bundle = bundles.build(root=public_full)
    with zipfile.ZipFile(bundle) as z:
        for name in bundles.PUBLIC_DATA:
            assert z.read("data/v3/" + name) == (directory / name).read_bytes()
        assert "data/v3/review_manifest.json" not in z.namelist()
        assert all(set(json.loads(line)) == {"id", "problem", "language"}
                   for line in z.read("data/v3/latency_inputs.jsonl").decode().splitlines())
    bundles.inspect_bundle(bundle, root=public_full)


@pytest.mark.parametrize("field", ["seeds", "repetitions", "conditions", "expected_rows", "blocks"])
def test_public_plan_cannot_drop_mandatory_execution_controls(public_full, field):
    raw = {"data/v3/" + name: (public_full / "data/v3" / name).read_bytes() for name in bundles.PUBLIC_DATA}
    plan = json.loads(raw["data/v3/latency_plan.json"])
    plan.pop(field)
    raw["data/v3/latency_plan.json"] = bundles.encoded(plan)
    metadata = json.loads(raw["data/v3/dataset_manifest.json"])
    metadata["input_sha256"]["latency_plan.json"] = bundles.sha(raw["data/v3/latency_plan.json"])
    raw["data/v3/dataset_manifest.json"] = bundles.encoded(metadata)
    seals = {line.split()[1]: line.split()[0] for line in raw["data/v3/SHA256SUMS"].decode().splitlines()}
    for name in ("latency_plan.json", "dataset_manifest.json"):
        seals[name] = bundles.sha(raw["data/v3/" + name])
    raw["data/v3/SHA256SUMS"] = "".join(f"{h}  {n}\n" for n, h in seals.items()).encode()
    with pytest.raises(ValueError, match="Missing.*latency plan"):
        bundles.validate_public(raw, False)


def test_legacy_script_hashes_match_previous_development_bundle():
    archive = ROOT / "dist/v3/jev_llm_v3_dev_bundle.zip"
    if not archive.is_file():
        pytest.skip("No prior development bundle available for legacy hash comparison")
    with zipfile.ZipFile(archive) as z:
        sources = json.loads(z.read("BUNDLE_MANIFEST.json"))["source_sha256"]
    for name in ("tools/colab/session_guard.py", "tools/colab/session_guard.sh", "tools/colab/run_v2.py", "tools/build_notebooks.py", "tools/verify_run.py"):
        assert bundles.sha((ROOT / name).read_bytes()) == sources[name]


def test_public_manifest_cannot_smuggle_review_payload(source):
    path = bundles.build(dev=True, root=source)
    def corrupt(raw, m):
        name = "data/v3/dataset_manifest.json"
        metadata = json.loads(raw[name])
        metadata["review_payload"] = {"solution": "hidden"}
        raw[name] = bundles.encoded(metadata)
        m["files"][name] = bundles.sha(raw[name])
    rewrite(path, corrupt)
    with pytest.raises(ValueError, match="Private/unapproved"):
        bundles.inspect_bundle(path, check_local=False)


@pytest.mark.parametrize("failure", ["missing-pilot", "mismatched-pilot", "no-go", "missing-budget", "not-authorized"])
def test_notebook_approval_requires_user_flag_pinned_pilot_and_budget(tmp_path, failure):
    entries = {c: {"verified": True, "denominator": 150 if c in {"G_SINGLE", "JFINAL", "B13"} else 50,
                  "verification": {"ok": True, "condition": c, "records": 150 if c in {"G_SINGLE", "JFINAL", "B13"} else 50},
                  "rates": {"denominator": 150 if c in {"G_SINGLE", "JFINAL", "B13"} else 50, "timeout_infra_rate": 0}}
               for c in notebooks.CONDITIONS}
    pilot = dict(protocol_version="3", code_version="0.3.0", go=True, decision="go", errors=[], budget_approval=False,
                 conditions=entries, aggregate=dict(complete=True, denominator=600, observed_cases=600, timeout_infra_rate=0))
    if failure == "no-go":
        pilot["go"] = False
    pilot_path = tmp_path / "fixture_pilot.json"
    pilot_path.write_text(json.dumps(pilot))
    approval = dict(approved=True, scope="test_and_latency", config_sha256="a" * 64,
                    pilot_report_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                    approved_by="synthetic-schema-fixture-only", approved_utc="2020-01-01T00:00:00+00:00", max_gpu_hours=1)
    if failure == "mismatched-pilot":
        approval["pilot_report_sha256"] = "b" * 64
    if failure == "missing-budget":
        approval.pop("max_gpu_hours")
    approval_path = tmp_path / "fixture_approval.json"
    approval_path.write_text(json.dumps(approval))
    env = dict(CONFIG_FILE=bundles.CONFIG, DATA_SUBDIR="data/v3", SPLIT="test", CONDITION="JFINAL",
               CONFIRMATORY_AUTHORIZED=failure != "not-authorized", APPROVAL_FILE=str(approval_path),
               PILOT_REPORT_FILE="" if failure == "missing-pilot" else str(pilot_path),
               man={"config_sha256": "a" * 64}, Path=Path, json=json, hashlib=hashlib)
    with pytest.raises(ValueError):
        exec(notebooks.APPROVAL, env)


@pytest.mark.parametrize("condition", ["G_SINGLE", "G_GREEDY", "JFINAL", "B13", "B13_GREEDY", "Q9_GREEDY", "LATENCY"])
@pytest.mark.parametrize("split", ["test", "dev"])
def test_constructor_forwards_actual_consent_arguments(condition, split, tmp_path, monkeypatch):
    from jevlab.v3 import runner, latency_study
    seen = {}
    class FakeRuntime:
        def __init__(self, **params):
            seen.update(params)
            self.dir = "synthetic-constructor-fixture"
    module, name = (latency_study, "LatencyStudy") if condition == "LATENCY" else (runner, "Experiment")
    monkeypatch.setattr(module, name, FakeRuntime)
    nb = notebooks.condition_notebook(condition)
    env = {}
    exec(next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", [])), env)
    approval = tmp_path / "user-supplied-fixture.json"
    report = tmp_path / "user-pilot-fixture.json"
    approval.write_text('{"fixture":"supplied input only, not actual funding"}')
    report.write_text('{"fixture":"supplied input only, not an actual pilot GO"}')
    env.update(SPLIT=split, APPROVAL_FILE=str(approval) if split == "test" else "",
               PILOT_REPORT_FILE=str(report) if split == "test" else "", CONFIRMATORY_AUTHORIZED=split == "test")
    create = next(c.source for c in nb.cells if "exp = " in c.source)
    exec(create, env)
    assert seen["APPROVAL_FILE"] == env["APPROVAL_FILE"]
    assert seen["PILOT_REPORT_FILE"] == env["PILOT_REPORT_FILE"]
    assert seen["CONFIRMATORY_AUTHORIZED"] is (split == "test")
    assert "BUNDLE_FILE" not in seen
    if split == "dev":
        assert runner.check_budget_approval(type("DevContext", (), {"p": seen})()) == {"required": False, "scope": "dev"}


def test_same_prefix_wrong_official_revision_refused_by_build_and_embedded_loader(source):
    path = bundles.build(dev=True, root=source)
    def wrong_pin(raw, manifest):
        cfg = json.loads(raw[bundles.CONFIG])
        cfg["models"]["O"]["revision"] = bundles.OFFICIAL_Q9_REVISION[:7] + "0" * 33
        raw[bundles.CONFIG] = bundles.encoded(cfg)
        manifest["files"][bundles.CONFIG] = bundles.sha(raw[bundles.CONFIG])
        manifest["config_sha256"] = bundles.sha(raw[bundles.CONFIG])
        manifest["models"] = cfg["models"]
        manifest["source_sha256"][bundles.CONFIG] = bundles.sha(raw[bundles.CONFIG])
    rewrite(path, wrong_pin)
    with pytest.raises(ValueError, match="exact pinned official"):
        bundles.inspect_bundle(path, check_local=False)
    env = dict(ROOT=str(source / "runtime"), BUNDLE_FILE=str(path), SPLIT="dev", RUN_TAG="smoke", SEEDS="17")
    with pytest.raises(ValueError, match="exact pinned official"):
        exec(notebooks.bundle_cell(), env)
    cfg_path = source / bundles.CONFIG
    cfg = json.loads(cfg_path.read_bytes())
    cfg["models"]["O"]["revision"] = bundles.OFFICIAL_Q9_REVISION[:7] + "0" * 33
    cfg_path.write_bytes(bundles.encoded(cfg))
    with pytest.raises(ValueError, match="exact pinned official"):
        bundles.build(root=source, dev=True, name="wrong-pin-dev.zip")


@pytest.mark.parametrize("embedded", [False, True])
def test_zip_size_bound_precedes_any_member_read(source, monkeypatch, embedded):
    path = bundles.build(dev=True, root=source)
    original = zipfile.ZipFile.infolist
    def oversized(archive):
        infos = original(archive)
        infos[0].file_size = 512 * 1024 * 1024 + 1
        return infos
    monkeypatch.setattr(zipfile.ZipFile, "infolist", oversized)
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized ZIP must be rejected before reads or testzip")
    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "testzip", forbidden)
    with pytest.raises(ValueError, match="uncompressed size"):
        if embedded:
            exec(notebooks.bundle_cell(), dict(ROOT=str(source / "runtime"), BUNDLE_FILE=str(path),
                                              SPLIT="dev", RUN_TAG="smoke", SEEDS="17"))
        else:
            bundles.inspect_bundle(path, check_local=False)


@pytest.mark.parametrize("nested", ["domain_policy", "difficulty_policy", "adjudication", "eligibility", "seeds", "quota", "regex_evidence", "method"])
def test_policy_nested_private_payload_rejected(public_full, nested):
    raw = {"data/v3/" + n: (public_full / "data/v3" / n).read_bytes() for n in bundles.PUBLIC_DATA}
    policy = json.loads(raw["data/v3/review_policy.json"])
    if nested == "method":
        policy[nested] = {"private_solution": "secret"}
    else:
        target = policy["domain_policy"][nested] if nested == "regex_evidence" else policy[nested]
        target["private_solution"] = {"answer": "secret"}
    raw["data/v3/review_policy.json"] = bundles.encoded(policy)
    with pytest.raises(ValueError, match="policy"):
        bundles.validate_public(raw, False)


@pytest.mark.parametrize("corruption", [None, "agent_reviews", "review_decision", "domain", "difficulty",
                                       "sampling_tier", "accepted_via_adjudication", "provisional_domain", "builder",
                                        "archive-original", "archive-intermediate", "intrinsic-counts", "source-lock"])
def test_synthetic_builder_final_labels_and_migration_lineage(public_full, monkeypatch, corruption):
    """Mock publisher integration only, never authorization or real dataset resealing."""
    root = public_full
    directory, stage = root / "data/v3", root / "data/v3/staging"
    real_validate = bundles.validate_dataset
    monkeypatch.setattr(bundles, "_validate_dataset_full", bundles._validate_dataset_full.__wrapped__)
    rows, selected, decisions = [], {}, {}
    for split in ("test", "pilot"):
        inputs = [json.loads(l) for l in (directory / f"{split}_inputs.jsonl").read_text().splitlines()]
        gold = [json.loads(l) for l in (directory / f"{split}_gold.jsonl").read_text().splitlines()]
        selected[split] = []
        for i, g in zip(inputs, gold):
            candidate = {**i, "id": "candidate-" + i["id"], "provisional_domain": "algebra",
                         "provisional_difficulty": g["sampling_tier"], "candidate_sha256": "a" * 64}
            decision = {"reviews": [{"synthetic": True}], "final_domain": g.get("domain", "arithmetic"),
                        "final_difficulty": g["difficulty"], "sampling_tier": candidate["provisional_difficulty"],
                        "accepted_via_adjudication": True}
            decisions[candidate["id"]] = decision
            rows.append(candidate)
            selected[split].append(candidate)
            g.update({k: v for k, v in candidate.items() if k != "id"})
            g.update(candidate_id=candidate["id"], agent_reviews=decision["reviews"], review_decision=decision,
                      domain=decision["final_domain"], difficulty=decision["final_difficulty"],
                      sampling_tier=candidate["provisional_difficulty"],
                      accepted_via_adjudication=True, agent_reviewed=True, human_reviewed=False)
        assert any(g["difficulty"] != g["sampling_tier"] for g in gold)
        if corruption in gold[0]:
            gold[0][corruption] = "corrupt synthetic field"
        (directory / f"{split}_gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
    dev = [json.loads(l) for l in (directory / "dev_inputs.jsonl").read_text().splitlines()]
    (directory / "dev_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
    archive_names = {era: f"data/v3/policy_amendments/synthetic-{era}/data/build_dataset_v3.py"
                     for era in ("original", "intermediate")}
    protected = {}
    for era, name in archive_names.items():
        archive = root / name
        archive.parent.mkdir(parents=True)
        archive.write_bytes(f"synthetic {era} builder, not executable\n".encode())
        protected[name] = bundles.sha(archive.read_bytes())
    historical = {"data/build_dataset_v3.py": protected[archive_names["original"]]}
    intermediate = {"data/build_dataset_v3.py": protected[archive_names["intermediate"]]}
    policy = json.loads((directory / "review_policy.json").read_bytes())
    cfg_path = root / bundles.CONFIG
    cfg = json.loads(cfg_path.read_bytes())
    cfg["dataset_review_policy"] = {"id": policy["policy_id"], "sha256": policy["contract_sha256"]}
    cfg_path.write_bytes(bundles.encoded(cfg))
    lock = root / "data/synthetic-source-lock.json"
    lock.write_bytes(b'{"synthetic-source":"immutable-test-only"}')
    metadata = {"builder_sha256": bundles.sha((root / "data/build_dataset_v3.py").read_bytes()),
                "construction_seed": 20261002, "latency_seed": 20261003,
                "review_policy_json_sha256": bundles.sha((directory / "review_policy.json").read_bytes()),
                 "previous_artifact_sha256": historical,
                 "synthetic_intermediate_artifact_sha256": intermediate,
                "source_sha256": {"data/dev_inputs.jsonl": bundles.sha((root / "data/dev_inputs.jsonl").read_bytes())},
                "source_lock_path": lock.relative_to(root).as_posix(), "source_lock_sha256": bundles.sha(lock.read_bytes()),
                "tokenizers": {}, "generator_prompt_sha256": bundles.sha((root / "prompts/generador.txt").read_bytes()),
                "packets": []}
    reference = stage / "synthetic_reference.jsonl"
    blind = stage / "synthetic_blind.jsonl"
    reference.write_text("".join(json.dumps(r) + "\n" for r in rows))
    blind.write_text("".join(json.dumps({k: r[k] for k in ("id", "problem", "language")}) + "\n" for r in rows))
    metadata["packets"] = [{"blind": blind.relative_to(root).as_posix(), "reference": reference.relative_to(root).as_posix(),
                            "blind_sha256": bundles.sha(blind.read_bytes()), "reference_sha256": bundles.sha(reference.read_bytes()),
                            "candidate_ids": [r["id"] for r in rows]}]
    (stage / "preparation_manifest.json").write_bytes(bundles.encoded(metadata))
    (directory / "review_manifest.json").write_bytes(b"{}")
    public = json.loads((directory / "dataset_manifest.json").read_bytes())
    public["analysis_manifest_sha256"] = bundles.sha(b"{}")
    if corruption == "intrinsic-counts":
        public["review"]["intrinsic_difficulty_counts"]["test"]["easy"] -= 1
        public["review"]["intrinsic_difficulty_counts"]["test"]["medium"] += 1
    (directory / "dataset_manifest.json").write_bytes(bundles.encoded(public))
    def inventory(path):
        return {p.relative_to(path).as_posix(): bundles.sha(p.read_bytes()) for p in path.rglob("*")
                if p.is_file() and p != path / "SHA256SUMS"}
    seals = inventory(directory)
    (directory / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in seals.items()))
    calls = []
    def lineage(actual_stage, actual_metadata):
        assert actual_stage == stage and actual_metadata == metadata
        assert actual_metadata["previous_artifact_sha256"] == historical
        assert actual_metadata["synthetic_intermediate_artifact_sha256"] == intermediate
        calls.append("lineage")
        return protected.copy()
    def select(actual_rows, approved, actual_decisions):
        assert actual_rows == rows and approved == set(decisions) and actual_decisions == decisions
        calls.append("selection")
        return selected
    def read_rows(path):
        return rows if path.name == "candidate_pool.jsonl" else [json.loads(l) for l in path.read_text().splitlines()]
    def load(name, path):
        assert path == root / "data/build_dataset_v3.py" and path != bundles.ROOT / "data/build_dataset_v3.py"
        def execute(module):
            module.__dict__.update(SEED=20261002, LATENCY_SEED=20261003, verify_seal=inventory,
                read_json=lambda p: json.loads(p.read_bytes()), read_jsonl=read_rows,
                verify_preparation_lineage=lineage, review_policy=lambda: policy, candidate_sha=lambda r: "a" * 64,
                load_reviews=lambda *args: (set(decisions), decisions, {}), select_approved=select,
                latency_subset=lambda gold: [r["id"] for r in read_rows(directory / "latency_inputs.jsonl")],
                schedule_for=lambda *args: json.loads((directory / "schedule.json").read_bytes()),
                latency_plan_for=lambda *args: json.loads((directory / "latency_plan.json").read_bytes()))
        return importlib.util.spec_from_loader(name, SimpleNamespace(create_module=lambda spec: None, exec_module=execute))
    monkeypatch.setattr(bundles.importlib.util, "spec_from_file_location", load)
    if corruption == "builder":
        (root / "data/build_dataset_v3.py").write_bytes(b"changed synthetic root builder")
    if corruption and corruption.startswith("archive-"):
        (root / archive_names[corruption.removeprefix("archive-")]).write_bytes(b"changed historical builder")
    if corruption == "source-lock":
        lock.write_bytes(b'{"synthetic-source":"changed-test-only"}')
    if corruption:
        with pytest.raises(ValueError, match="builder/seed|evidence changed|gold differs|intrinsic difficulty counts"):
            real_validate(root)
    else:
        validated = real_validate(root)
        assert calls == ["lineage", "selection"]
        for name, digest in protected.items():
            assert validated["lineage_inventory"][name] == digest
        assert validated["lineage_inventory"]["data/build_dataset_v3.py"] == metadata["builder_sha256"]
        assert validated["lineage_inventory"][metadata["source_lock_path"]] == metadata["source_lock_sha256"]
        monkeypatch.setattr(bundles, "validate_dataset", real_validate)
        payload, manifest = bundles._payload(root, False, False)
        for name, digest in protected.items():
            assert manifest["source_sha256"][name] == digest
            assert name not in payload
        assert manifest["source_sha256"]["data/build_dataset_v3.py"] == metadata["builder_sha256"]
        assert manifest["source_sha256"][metadata["source_lock_path"]] == metadata["source_lock_sha256"]
        assert metadata["source_lock_path"] not in payload
        assert "data/build_dataset_v3.py" not in payload
        assert metadata["previous_artifact_sha256"] == historical
        assert metadata["synthetic_intermediate_artifact_sha256"] == intermediate
        assert calls == ["lineage", "selection"]  # Payload reuse must not replay the full proof.


@pytest.fixture
def certificate_root(tmp_path, monkeypatch):
    """Small, entirely invented proof inputs; never read project datasets or answers."""
    root = tmp_path / "synthetic-certificate-only"
    external = {"fixtures/source.jsonl", "fixtures/previous.py", "fixtures/source-lock.json",
                "fixtures/tokenizer/tokenizer.json"}
    names = {bundles.CONFIG, "data/build_dataset_v3.py", "data/build_dataset.py",
             "data/dev_inputs.jsonl", "prompts/generador.txt", "src/jevlab/__init__.py",
             "src/jevlab/v3/nested/fixture.py", "data/fixture_builder.py", *bundles.TOOL_SOURCES,
             *external, "data/v3/test_inputs.jsonl", "data/v3/review_policy.json",
             "data/v3/REVIEW_CONTRACT.md", "data/v3/staging/reviews/fixture.json"}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic fixture only: " + name + "\n").encode())
    metadata = {"source_sha256": {"fixtures/source.jsonl": bundles.sha((root / "fixtures/source.jsonl").read_bytes())},
                "previous_artifact_sha256": {"fixtures/previous.py": bundles.sha((root / "fixtures/previous.py").read_bytes())},
                "source_lock_path": "fixtures/source-lock.json",
                "tokenizers": {"synthetic": {"directory": "fixtures/tokenizer", "sha256": {
                    "tokenizer.json": bundles.sha((root / "fixtures/tokenizer/tokenizer.json").read_bytes())}}}}
    (root / "data/v3/staging/preparation_manifest.json").write_bytes(bundles.encoded(metadata))
    seals = {p.relative_to(root / "data/v3").as_posix(): bundles.sha(p.read_bytes())
             for p in (root / "data/v3").rglob("*") if p.is_file()}
    (root / "data/v3/SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(seals.items())))
    # Snapshot every file in the miniature tree, including nested review evidence.
    snapshot = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    validated = {"manifest": {"synthetic_fixture_only": True, "review": {"replay_count": 1}},
                 "seals": seals, "lineage_inventory": {n: bundles.sha(snapshot[n]) for n in external}}
    calls = []
    def full(actual_root):
        assert actual_root == root and actual_root != ROOT
        calls.append(actual_root)
        return json.loads(json.dumps(validated))
    monkeypatch.setattr(bundles, "_DATASET_CERTIFICATES", {})
    monkeypatch.setattr(bundles, "_validate_dataset_full", full)
    return SimpleNamespace(root=root, snapshot=snapshot, validated=validated, calls=calls, full=full)


def test_dataset_certificate_full_once_reuse_rehashes_every_byte_and_deepcopies(certificate_root, monkeypatch):
    s = certificate_root
    first = bundles.validate_dataset(s.root)
    assert first == s.validated and s.calls == [s.root]
    reads, original_open = [], Path.open
    def opened(path, mode="r", *args, **kwargs):
        assert path.is_relative_to(s.root), "Certificate reuse must read only synthetic inputs"
        if mode == "rb":
            reads.append(path.relative_to(s.root).as_posix())
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    first["manifest"]["review"]["replay_count"] = 999
    first["seals"].clear()
    second = bundles.validate_dataset(s.root)
    assert second == s.validated and second is not first
    assert Counter(reads) == Counter({name: 1 for name in s.snapshot})
    second["manifest"]["review"]["replay_count"] = 0
    second["lineage_inventory"].clear()
    assert bundles.validate_dataset(s.root) == s.validated
    assert s.calls == [s.root]  # Successful replay is reused, not skipped on the first call.


@pytest.mark.parametrize("name", ["data/v3/test_inputs.jsonl", "data/v3/staging/reviews/fixture.json",
    "data/build_dataset_v3.py", "data/build_dataset.py", "data/fixture_builder.py",
    "src/jevlab/v3/nested/fixture.py", bundles.CONFIG, "data/v3/review_policy.json",
    "data/v3/REVIEW_CONTRACT.md", "fixtures/source.jsonl", "fixtures/previous.py",
    "fixtures/source-lock.json", "fixtures/tokenizer/tokenizer.json", "prompts/generador.txt",
    "tools/make_bundle_v3.py", "data/v3/SHA256SUMS", "data/v3/staging/preparation_manifest.json"])
def test_dataset_certificate_same_length_restored_mtime_mutation_fails(certificate_root, name):
    s = certificate_root
    bundles.validate_dataset(s.root)
    path = s.root / name
    before = path.stat()
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="certificate bytes/path inventory changed"):
        bundles.validate_dataset(s.root)
    assert s.calls == [s.root]


@pytest.mark.parametrize("damage", ["extra-file", "missing-file", "extra-directory", "missing-directory", "source-directory"])
def test_dataset_certificate_changed_inventory_fails(certificate_root, damage):
    s = certificate_root
    empty = s.root / "data/v3/staging/empty"
    empty.mkdir()
    bundles.validate_dataset(s.root)
    if damage == "extra-file":
        (s.root / "data/v3/staging/extra.json").write_bytes(b"synthetic extra")
    elif damage == "missing-file":
        (s.root / "data/v3/test_inputs.jsonl").unlink()
    elif damage == "extra-directory":
        (s.root / "data/v3/extra").mkdir()
    elif damage == "missing-directory":
        empty.rmdir()
    else:
        path = s.root / "fixtures/source.jsonl"
        path.unlink()
        path.mkdir()
    with pytest.raises((ValueError, OSError)):
        bundles.validate_dataset(s.root)
    assert s.calls == [s.root]


@pytest.mark.parametrize("kind", ["file", "source-parent", "dataset-directory"])
def test_dataset_certificate_native_symlinks_rejected(certificate_root, kind):
    s = certificate_root
    bundles.validate_dataset(s.root)
    if kind == "file":
        path, target = s.root / "fixtures/source.jsonl", s.root / "fixtures/previous.py"
        path.unlink()
    elif kind == "source-parent":
        path, target = s.root / "fixtures/tokenizer", s.root / "fixtures/other-tokenizer"
        path.rename(target)
    else:
        path, target = s.root / "data/v3/linked", s.root / "fixtures"
    try:
        path.symlink_to(target, target_is_directory=kind != "file")
    except OSError as exc:
        pytest.skip(f"Native symlink privileges unavailable: {exc}")
    with pytest.raises(ValueError, match="[Ss]ymlink|Non-regular"):
        bundles.validate_dataset(s.root)
    assert s.calls == [s.root]


@pytest.mark.parametrize("failure", ["mutation", "seal-binding", "lineage-binding", "replay-error"])
def test_dataset_certificate_failed_first_proof_never_publishes(certificate_root, monkeypatch, failure):
    s = certificate_root
    def full(root):
        result = s.full(root)
        if failure == "mutation":
            (root / "data/v3/test_inputs.jsonl").write_bytes(b"mutated during synthetic proof")
        elif failure == "seal-binding":
            result["seals"]["test_inputs.jsonl"] = "0" * 64
        elif failure == "lineage-binding":
            result["lineage_inventory"]["fixtures/source.jsonl"] = "0" * 64
        else:
            raise ValueError("Synthetic full replay rejected")
        return result
    monkeypatch.setattr(bundles, "_validate_dataset_full", full)
    with pytest.raises(ValueError, match="proof changed|full replay rejected"):
        bundles.validate_dataset(s.root)
    assert not bundles._DATASET_CERTIFICATES and s.calls == [s.root]
    for name, raw in s.snapshot.items():
        (s.root / name).write_bytes(raw)
    monkeypatch.setattr(bundles, "_validate_dataset_full", s.full)
    assert bundles.validate_dataset(s.root) == s.validated
    assert s.calls == [s.root, s.root]


def test_dataset_fingerprint_matches_complete_synthetic_snapshot(certificate_root):
    s = certificate_root
    hashes, directories = bundles.dataset_fingerprint(s.root, s.snapshot)
    assert hashes == {name: bundles.sha(raw) for name, raw in s.snapshot.items()}
    assert directories == {p.relative_to(s.root).as_posix() for p in (s.root / "data/v3").rglob("*") if p.is_dir()}
    assert not s.calls


def test_synthetic_real_publisher_selection_uses_reviewed_domain_and_original_tier(publisher):
    """Exercise real selection on synthetic rows, not review authorization or migration."""
    rows, decisions = [], {}
    for domain in publisher.DOMAINS:
        for tier in publisher.TEST_QUOTA:
            for i in range(publisher.TEST_QUOTA[tier] + publisher.PILOT_QUOTA[tier]):
                row = {"id": f"synthetic-{domain}-{tier}-{i}", "provisional_domain": "not-reviewed-domain",
                       "provisional_difficulty": tier, "source_family": "fixture-a" if i % 2 else "fixture-b"}
                rows.append(row)
                decisions[row["id"]] = {"accepted": True, "final_domain": domain,
                                        "final_difficulty": "easy" if i % 2 else "medium", "sampling_tier": tier}
    selected = publisher.select_approved(rows, set(decisions), decisions)
    for split, quota in (("test", publisher.TEST_QUOTA), ("pilot", publisher.PILOT_QUOTA)):
        for domain in publisher.DOMAINS:
            assert Counter(r["provisional_difficulty"] for r in selected[split]
                           if decisions[r["id"]]["final_domain"] == domain) == quota
        assert not any(decisions[r["id"]]["final_difficulty"] == "hard" for r in selected[split])
    assert {r["id"] for r in selected["test"]}.isdisjoint(r["id"] for r in selected["pilot"])
    rows[0]["provisional_difficulty"] = "medium"
    with pytest.raises(ValueError, match="Insufficient twice-reviewed strata"):
        publisher.select_approved(rows, set(decisions), decisions)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("corruption", [None, "all-medium", "missing-tier", "wrong-tier-quota", "wrong-latency-counts",
    "invalid-label", "missing-public-counts", "wrong-public-counts", "extra-public-counts", "missing-quota-axis",
    "tier-is-intrinsic", "gold", "adjudication", "private-field", "language", "policy-id", "contract-sha"])
def test_synthetic_public_and_embedded_source_tier_contract(public_full, embedded, corruption):
    """Resealed synthetic public payloads test validation, not actual dataset publication."""
    raw = {"data/v3/" + n: (public_full / "data/v3" / n).read_bytes() for n in bundles.PUBLIC_DATA}
    plan = json.loads(raw["data/v3/latency_plan.json"])
    metadata = json.loads(raw["data/v3/dataset_manifest.json"])
    assert metadata["review"]["intrinsic_difficulty_counts"] == {
        "test": {"easy": 250, "medium": 250, "hard": 0}, "pilot": {"easy": 25, "medium": 25, "hard": 0}}
    assert plan["intrinsic_difficulty_counts"] == {
        d: sum(i["difficulty"] == d for i in plan["items"]) for d in ("easy", "medium", "hard")}
    for domain in {i["domain"] for i in plan["items"]}:
        assert Counter(i["sampling_tier"] for i in plan["items"] if i["domain"] == domain) == {"easy": 6, "medium": 8, "hard": 6}
    if corruption == "all-medium":
        for item in plan["items"]:
            item["difficulty"] = "medium"
        plan["intrinsic_difficulty_counts"] = {"easy": 0, "medium": 100, "hard": 0}
    elif corruption == "missing-tier":
        plan["items"][0].pop("sampling_tier")
    elif corruption == "wrong-tier-quota":
        item = plan["items"][0]
        item["sampling_tier"] = "medium" if item["sampling_tier"] != "medium" else "easy"
    elif corruption == "wrong-latency-counts":
        plan["intrinsic_difficulty_counts"]["easy"] += 1
        plan["intrinsic_difficulty_counts"]["medium"] -= 1
    elif corruption == "invalid-label":
        plan["items"][0]["difficulty"] = "unknown"
    elif corruption == "missing-public-counts":
        metadata["review"].pop("intrinsic_difficulty_counts")
    elif corruption == "wrong-public-counts":
        metadata["review"]["intrinsic_difficulty_counts"]["test"]["easy"] -= 1
    elif corruption == "extra-public-counts":
        metadata["review"]["intrinsic_difficulty_counts"]["pilot"]["unknown"] = 0
    elif corruption == "missing-quota-axis":
        metadata.pop("quota_axis")
    elif corruption == "tier-is-intrinsic":
        metadata["review"]["source_sampling_tier_is_intrinsic_difficulty"] = True
    elif corruption in {"gold", "adjudication", "private-field", "language"}:
        field = {"gold": "gold_numerator", "adjudication": "accepted_via_adjudication",
                 "private-field": "reference_solution", "language": "language"}[corruption]
        plan["items"][0][field] = "synthetic leaked field"
    elif corruption in {"policy-id", "contract-sha"}:
        policy = json.loads(raw["data/v3/review_policy.json"])
        if corruption == "policy-id":
            policy["policy_id"] = metadata["review"]["policy_id"] = "synthetic-unauthorized-policy"
        else:
            raw["data/v3/REVIEW_CONTRACT.md"] += b"\nSynthetic unauthorized amendment.\n"
            policy["contract_sha256"] = metadata["review"]["policy_sha256"] = bundles.sha(raw["data/v3/REVIEW_CONTRACT.md"])
        raw["data/v3/review_policy.json"] = bundles.encoded(policy)
        metadata["policy_artifact_sha256"]["review_policy.json"] = bundles.sha(raw["data/v3/review_policy.json"])
    raw["data/v3/latency_plan.json"] = bundles.encoded(plan)
    metadata["input_sha256"]["latency_plan.json"] = bundles.sha(raw["data/v3/latency_plan.json"])
    raw["data/v3/dataset_manifest.json"] = bundles.encoded(metadata)
    seals = {line.split()[1]: line.split()[0] for line in raw["data/v3/SHA256SUMS"].decode().splitlines()}
    seals.update({n.removeprefix("data/v3/"): bundles.sha(content) for n, content in raw.items()
                  if n != "data/v3/SHA256SUMS"})
    raw["data/v3/SHA256SUMS"] = "".join(f"{h}  {n}\n" for n, h in seals.items()).encode()
    validate = bundles.validate_public
    if embedded:
        env = {"ROOT": public_full}
        exec(notebooks.bundle_cell().split("\nROOT = Path(ROOT)", 1)[0], env)
        validate = env["validate_public"]
    if corruption in {None, "all-medium"}:
        validate(raw, False)
    else:
        with pytest.raises(ValueError):
            validate(raw, False)
