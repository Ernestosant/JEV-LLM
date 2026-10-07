"""Generated v3 evidence only: offline CPU tests, never real data or gold."""

import copy
from contextlib import nullcontext
import json
import random
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_run_v3 as verify
import pilot_decision_v3 as pilot
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def lines(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def evidence(tmp_path):
    inputs = tmp_path / "pilot_inputs.jsonl"
    inputs.write_text(lines([{"id": f"p{i}", "problem": f"Problem {i}", "language": "en"} for i in range(50)]))
    metadata = tmp_path / "dataset_manifest.json"
    policy = tmp_path / "REVIEW_CONTRACT.md"
    policy.write_text("Synthetic fixture policy, never actual agent review.\n")
    (tmp_path / "review_policy.json").write_text(json.dumps({"policy_id": verify.REVIEW_POLICY_ID,
        "contract_sha256": verify.sha256(policy.read_bytes()), "quota_axis": verify.QUOTA_AXIS,
        "quota": verify.SAMPLING_QUOTAS}))
    (tmp_path / "ELIGIBILITY_POLICY.md").write_text("Synthetic fixture eligibility only.\n")
    metadata.write_text(json.dumps({"dataset_version": "v3", "sealed": True,
                                   "status": "FROZEN", "construction_seed": 20261002, "latency_seed": 20261003,
                                   "test_reserved_before_pilot": True, "latency_test_subset_only": True,
                                   "builder_sha256": "a" * 64, "analysis_manifest_sha256": "b" * 64,
                                   "quota_axis": verify.QUOTA_AXIS, "quota": verify.SAMPLING_QUOTAS,
                                   "source_revisions": {"fixture": {"repo": "synthetic", "revision": "a" * 40}},
                                   "policy_artifact_sha256": {n: verify.sha256((tmp_path / n).read_bytes()) for n in
                                                             ("review_policy.json", "ELIGIBILITY_POLICY.md")},
                                   "actual_n": {"test": 500, "pilot": 50, "dev": 20},
                                    "review": {"agent_reviewed_all": True, "human_reviewed": False,
                                                "policy_id": verify.REVIEW_POLICY_ID, "policy_sha256": verify.sha256(policy.read_bytes()),
                                                "source_sampling_tier_is_intrinsic_difficulty": False,
                                                "intrinsic_difficulty_counts": {"test": {"easy": 500, "medium": 0, "hard": 0},
                                                                                "pilot": {"easy": 50, "medium": 0, "hard": 0}}}}))
    seal_public_fixture(tmp_path, (inputs, metadata, policy))
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("generador", "criterio_paso", "criterio_final"):
        (prompts / (name + ".txt")).write_text(name + "\n")
    frozen = {"protocol_version": "3", "code_version": "0.3.0", "models": {
        role: {"repo": "fake-" + role if role != "O" else "Qwen/Qwen3.5-9B",
               "revision": "a" * 40 if role != "O" else "c202236235762e1c871ad0ccb60c8ee5ba337b9a"} for role in "GJBO"},
               "implementation_profile": "v3-mock", "jevk5_runtime": {"commit": "mock"},
              "selector": {"calibration_temperature_expected": 1.0},
              "limits": {"n_candidates": 4, "max_output_tokens": 1024}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(frozen))
    return tmp_path, inputs, config, prompts, frozen


def seal_public_fixture(root, paths):
    extra = (root / "review_policy.json", root / "ELIGIBILITY_POLICY.md")
    metadata = json.loads((root / "dataset_manifest.json").read_bytes())
    metadata["input_sha256"] = {p.name: verify.sha256(p.read_bytes()) for p in paths
                                if p.name.endswith("_inputs.jsonl") or p.name == "schedule.json"}
    (root / "dataset_manifest.json").write_text(json.dumps(metadata))
    (root / "SHA256SUMS").write_text("".join(f"{verify.sha256(p.read_bytes())}  {p.name}\n" for p in (*paths, *extra))
        + f"{metadata['analysis_manifest_sha256']}  review_manifest.json\n")


def snapshot(evidence, condition="JFINAL", split="pilot"):
    root, inputs, config, prompts, frozen = evidence
    cfg = {"protocol_version": "3", "code_version": "0.3.0", "models": frozen["models"],
           "code_sha256": {"v3/runner.py": "a" * 64},
           "profile": frozen["implementation_profile"], "jevk5_runtime": frozen["jevk5_runtime"],
           "experiment_config_sha256": verify.sha256(config.read_bytes()),
             "data_sha256": {p.name: verify.sha256(p.read_bytes()) for p in
                (root / "pilot_inputs.jsonl", root / "test_inputs.jsonl", root / "dataset_manifest.json",
                 root / "latency_inputs.jsonl", root / "latency_plan.json", root / "SHA256SUMS") if p.is_file()},
           "prompt_sha256": {n: verify.sha256(n.encode()) for n in ("generador", "criterio_paso", "criterio_final")},
            "params": {"CONDITION": condition, "SPLIT": split, "RUN_TAG": "mock", "N_CANDIDATES": 4,
                       "MAX_OUTPUT_TOKENS": 1024, "CANDIDATE_BUDGET": 4096, "GB_CONTEXT": 4096,
                       "TEMPERATURE": .7, "TOP_P": .9, "TOP_K": 0, "REPETITION_PENALTY": 1.,
                       "PRESENCE_PENALTY": 0., "FREQUENCY_PENALTY": 0.,
                       "SEEDS": "17,29,43" if condition in verify.SAMPLED or condition == "LATENCY" else "17"}}
    digest = verify.canonical_hash(cfg)
    run = f"{condition}_mock_{digest[:8]}"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    predictions, proposals = [], []
    for row in verify.jsonl(inputs.read_text()):
        for seed in verify.SEEDS if condition in verify.SAMPLED else (17,):
            i = len(predictions)
            key = verify.sha256(f"{digest}|{row['id']}|{seed}|{condition}|{cfg['profile']}".encode())
            common = {"problem_id": row["id"], "condition": condition, "seed": seed,
                      "profile": cfg["profile"], "accepted_tokens": 20, "public_output": "Ungraded synthetic output",
                      "effective_seed": seed if condition in verify.SAMPLED else None,
                      "config_hash": digest, "resume_key": key, "status": "final",
                      "t_start_utc": (start + timedelta(seconds=i * 5)).isoformat(),
                      "t_end_utc": (start + timedelta(seconds=i * 5 + 4)).isoformat(),
                      "T_total": 4, "T_first_accepted": 4, "T_proposals": 3, "T_selector": 1,
                      "gpu_uuid": "GPU-mock", "processed_prompt_tokens": 10,
                      "candidate_tokens_total": 20, "selector_input_tokens": 5}
            predictions.append(common)
            if condition == "JFINAL":
                common["candidate_set_id"] = key
                common["winner_branch"] = 0
                proposals.append({**common, "candidate_set_id": key, "status": "complete", "candidates": [
                    {"public_output": common["public_output"], "branch": b, "seed": seed, "seed_branch": seed * 100 + b, "status": "final"}
                    for b in range(4)]})
    data = {name: [] for name in verify.LEDGERS}
    data.update(predictions=predictions, metrics=copy.deepcopy(predictions), proposals=proposals,
                progress={"done": len(predictions), "total": len(predictions)}, latency_plan=None)
    data["manifest"] = {"condition": condition, "config": cfg, "config_hash": digest, "run_name": run,
                        "created_utc": start.isoformat(), "finished_utc": (start + timedelta(days=1)).isoformat(),
                        "counts": {n: len(data[n]) for n in verify.LEDGERS}, "stages": {"run": {"ok": True}},
                        "preflight": {n: {"blocking": True, "ok": True} for n in
                                      verify.BLOCKING | {"selector_native_reference", "selector_equivalence"}},
                        "model_lock": {r: {"ok": True, **frozen["models"][r]} for r in "GJBO"},
                        "gpu_uuid": "GPU-mock", "hw_probe": {"cuda": "mock", "device": "A100",
                            "bf16_gemm_tflops": 1, "d2d_copy_gbps": 1, "small_kernel_us": 1, "python_loop_ms": 1},
                        "environment": {"cuda_available": True, "gpus": [{"name": "NVIDIA A100", "memory.total": 40960, "uuid": "GPU-mock"}]}}
    complete_fixture(data, evidence)
    return data


def postflight_fixture(cfg, phase):
    roles = {"G", "J"} if phase == "H" else {"B"} if phase == "B" else {"O"} if phase == "O" else {"G"}
    role = "G" if phase == "H" else next(iter(roles))
    logits, probs = [0.] * 4, [.25] * 4
    native_inputs = [{"question": {"criteria": ["fixture"] * 4}} for _ in range(8)]
    identity = {"config_hash": verify.canonical_hash(cfg), "model": cfg["models"]["J"], "runtime": cfg["jevk5_runtime"],
                "snapshot": "/synthetic/J", "prompt_hash": verify.canonical_hash(native_inputs)}
    native_exec = str(uuid.UUID(str(uuid.uuid5(uuid.NAMESPACE_DNS, identity["config_hash"])), version=4))
    ids = list(range(10))
    native = {"temperature": 1., "identity": identity, "reference_id": verify.canonical_hash(identity),
              "native_execution_id": native_exec, "runtime_commit": cfg["jevk5_runtime"]["commit"],
              "runtime_version": "synthetic", "snapshot": identity["snapshot"], "model_device": "cuda:0",
              "model_dtype": "torch.bfloat16", "cuda_hardware": {"returncode": 0, "stdout": GPU_UUID},
              "items": [{"n_ids": 10, "ids": ids, "ids_sha256": verify.sha256(json.dumps(ids).encode()),
              "ids_equal_prompt_text": True, "raw_logits": logits, "probabilities": probs,
              "index": index, "reference_id": verify.canonical_hash(identity), "native_execution_id": native_exec} for index in range(8)]}
    return {"eager_vs_graph": {"ok": True, "blocking": False, "skipped": phase in {"B", "O"}},
            "cache": {"ok": True, "blocking": True, "roles": {r: True for r in roles}},
            "continuity": {"ok": True, "blocking": False, "rows": [
                {"prefix": k, "agree_16": 16} for k in (1, 63, 64, 65, 127, 128)]},
            "stress": {"ok": True, "blocking": True, "synthetic": True, "outside_T_total": True,
                       "generator_role": role, "rendered_prompt": "synthetic stress",
                       "rendered_prompt_sha256": verify.sha256(b"synthetic stress"),
                       "generator": {"n": 4 if phase == "H" else 1,
                            "tokens": [cfg["params"]["MAX_OUTPUT_TOKENS"]] * (4 if phase == "H" else 1)}},
            "selector_native_reference": {"ok": True, "blocking": True, "native": native, "returncode": 0,
                "identity": identity, "reference_id": native["reference_id"], "native_execution_id": native_exec},
            "selector_equivalence": {"ok": True, "blocking": True, "identity": identity,
                "reference_id": native["reference_id"], "native_execution_id": native_exec, "rows": [
                {"n_ids": 10, "ids": ids, "ids_sha256": native["items"][0]["ids_sha256"], "same_ids": True, "vllm_logits": logits, "native_logits": logits,
                 "native_ids": ids, "native_ids_sha256": native["items"][0]["ids_sha256"], "index": index,
                 "reference_id": native["reference_id"], "native_execution_id": native_exec,
                  "vllm_probs": probs, "native_probs": probs, "argmax_vllm": 0, "argmax_native": 0} for index in range(8)]},
            "batching_audit": {"ok": True, "blocking": False},
            "detailed_trace_audit": {"ok": True, "blocking": True}}


def refresh_traces(data):
    """Synthetic persistence mirrors the runtime; corruption tests modify it afterwards."""
    data["candidates"], data["decisions"], data["failures"] = [], [], []
    by_resume = {r["resume_key"]: r for r in data["predictions"]}
    for pool in data["proposals"]:
        pred = by_resume[pool["resume_key"]]
        for candidate in pool["candidates"]:
            data["candidates"].append({**pred, **candidate})
        if pool["status"] == "complete":
            raw = f"{pred['seed']}|{pred['problem_id']}|JFINAL|0|0|option_order".encode()
            import hashlib
            seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
            perm = list(range(4))
            random.Random(seed).shuffle(perm)
            winner = pred.get("winner_branch", 0)
            data["decisions"].append({**pred, "round": 0, "perm_seed": seed, "permutation": perm,
                "status": "ok" if pred["status"] not in verify.ERROR_STATUSES else pred["status"],
                "winner_branch": winner, "winner_position": perm.index(winner)})
    data["failures"] = [copy.deepcopy(r) for r in data["predictions"] if r["status"] in verify.ERROR_STATUSES]
    data["manifest"]["counts"] = {name: len(data[name]) for name in verify.LEDGERS}


def complete_fixture(data, evidence):
    root, inputs, _, _, _ = evidence
    manifest, cfg = data["manifest"], data["manifest"]["config"]
    digest = verify.canonical_hash(cfg)
    manifest.update(config_hash=digest, gpu_uuid=GPU_UUID, preflight_full="preflight/summary.json")
    manifest.update(expected_coverage={"n_problems": 50 if cfg["params"]["SPLIT"] == "pilot" else 500,
                    "n_predictions": len(data["predictions"])}, coverage_complete=True)
    manifest["environment"]["gpus"][0]["uuid"] = GPU_UUID
    manifest["created_utc"] = "2025-12-31T23:59:59+00:00"
    texts = {r["id"]: r["problem"] for r in verify.jsonl(inputs.read_text())}
    measured = data["latency_metrics"] if manifest["condition"] == "LATENCY" else data["predictions"]
    primary_phase = "H" if manifest["condition"] == "JFINAL" else "B" if manifest["condition"].startswith("B13") else "O" if manifest["condition"] == "Q9_GREEDY" else "G"
    primary_meta = {"phase_epoch": 1, "phase_id": str(uuid.UUID(str(uuid.uuid5(uuid.NAMESPACE_DNS, digest)), version=4)),
                    "gpu_uuid": GPU_UUID, "resident_roles": ["G", "J"] if primary_phase == "H" else [primary_phase], "cleanup_confirmed": True}
    if manifest["condition"] != "LATENCY":
        manifest["active_phase"] = primary_meta
    for row in measured:
        row.update(gpu_uuid=GPU_UUID, config_hash=digest, generation=0,
                   provenance_hash=verify.canonical_hash(verify.case_provenance(cfg, row["condition"])))
        if manifest["condition"] != "LATENCY":
            row.update(primary_meta)
            row["resume_key"] = verify.sha256(f"{digest}|{row['problem_id']}|{row['seed']}|{row['condition']}|{cfg['profile']}".encode())
        if row["condition"] == "JFINAL":
            row["candidate_set_id"] = verify.canonical_hash({"problem_id": row["problem_id"],
                "problem": texts[row["problem_id"]], "provenance_hash": row["provenance_hash"],
                "proposal_seeds": verify.branch_seeds(row["problem_id"], row["seed"]), "proposal_condition": "G4_FINAL4_V3"})
    if manifest["condition"] == "LATENCY":
        previous = None
        for row in measured:
            phase = "H" if row["condition"] == "JFINAL" else "B"
            if phase != previous:
                epoch = len(data["swaps"]) + 1
                detail = postflight_fixture(cfg, phase)
                detail.pop("selector_native_reference")
                if phase == "B":
                    detail.pop("selector_equivalence")
                    detail.pop("batching_audit")
                phase_id = str(uuid.UUID(str(uuid.uuid5(uuid.NAMESPACE_DNS, digest + str(epoch))), version=4))
                metadata = {"phase_id": phase_id, "phase_epoch": epoch, "gpu_uuid": GPU_UUID,
                            "resident_roles": ["B"] if phase == "B" else ["G", "J"], "cleanup_confirmed": True}
                for result in detail.values():
                    result.update(metadata)
                    for equivalent in result.get("rows", []) if result is detail.get("selector_equivalence") else []:
                        equivalent.update(metadata)
                at = datetime.fromisoformat(row["t_start_utc"])
                closed_roles = [] if previous is None else ["B"] if previous == "B" else ["G", "J"]
                data["swaps"].append({"from": previous, "to": phase, **metadata, "method": "reload",
                    "gpu_uuid": GPU_UUID, "ok": True, "error": None, "cleanup_confirmed": True,
                    "outside_T_total": True, "load_time_measured_as_latency": False,
                    "resident_roles": ["B"] if phase == "B" else ["G", "J"], "seconds": .25, "load_seconds": .1,
                    "t_start_utc": (at - timedelta(seconds=.5)).isoformat(),
                    "t_end_utc": (at - timedelta(seconds=.25)).isoformat(), "postflight": detail,
                    "closed": {r: {"pids": [1000 + epoch], "exitcodes": [0], "cleanup_confirmed": True} for r in closed_roles}})
                previous = phase
            row.update(phase=phase, **metadata,
                       execution_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "synthetic-" + str(len(texts)) + row["resume_key"])))
            # Runtime emits UUID4, keep deterministic fixture bytes with valid version/variant.
            row["execution_id"] = str(uuid.UUID(row["execution_id"], version=4))
        data["predictions"], data["metrics"] = copy.deepcopy(measured), copy.deepcopy(measured)
    else:
        data["metrics"] = copy.deepcopy(measured)
    indexed = {(r["problem_id"], r["condition"], r["seed"], r.get("repetition")): r for r in measured}
    for pool in data["proposals"]:
        pred = indexed[(pool["problem_id"], pool["condition"], pool["seed"], pool.get("repetition"))]
        status, cs = pool["status"], pool["candidates"]
        pool.update(pred, status=status, candidates=cs, proposal_namespace="G4_FINAL4_V3", planned_candidate_slots=4,
                    seed_branches=verify.branch_seeds(pred["problem_id"], pred["seed"]))
        for c in cs:
            c.update({k: pred[k] for k in ("problem_id", "condition", "seed", "generation", "candidate_set_id", "provenance_hash")})
            c.update(round=0, seed_branch=pool["seed_branches"][c["branch"]])
    phase = "H" if manifest["condition"] in {"JFINAL", "LATENCY"} else "B" if manifest["condition"].startswith("B13") else "O" if manifest["condition"] == "Q9_GREEDY" else "G"
    full = postflight_fixture(cfg, phase)
    full.update(parser_selftest={"ok": True, "blocking": True, "failures": []},
                lock={"ok": True, "blocking": True, "params": {"fixture": 1}},
                chat_template={"ok": True, "blocking": True, "roles": {"fixture": {"rendered": "fixture"}}},
                tokenizer={"ok": True, "blocking": True, "roles": {"fixture": {"ok": True}}},
                prompt_length={"ok": True, "blocking": True, "over": {}, "n_prompts": len(texts)},
                sampling_semantics={"ok": True, "blocking": True})
    for result in full.values():
        result.update(phase_epoch=0, phase_id=None)
        if result is full["selector_equivalence"]:
            for equivalent in result["rows"]:
                equivalent.update(phase_epoch=0, phase_id=None)
    post_names = {"eager_vs_graph", "cache", "continuity", "stress", "detailed_trace_audit"}
    if primary_phase == "H":
        post_names |= {"selector_equivalence", "batching_audit"}
    if manifest["condition"] != "LATENCY":
        for name in post_names:
            full[name].update(primary_meta)
            if name == "selector_equivalence":
                for equivalent in full[name]["rows"]:
                    equivalent.update(primary_meta)
    manifest["preflight"] = {k: {field: v.get(field) for field in ("ok", "blocking", "phase_epoch", "phase_id",
                              "gpu_uuid", "resident_roles", "cleanup_confirmed")} for k, v in full.items()}
    native_ref = full["selector_native_reference"]
    native_inputs = [{"question": {"criteria": ["fixture"] * 4}} for _ in range(8)]
    data["artifacts"] = {"preflight/summary.json": full,
        "preflight/phases/pre_engines/native_ref_out.json": native_ref["native"],
        "preflight/phases/pre_engines/native_ref_in.json": {"items": native_inputs,
            "identity": native_ref["identity"], "native_execution_id": native_ref["native_execution_id"]}}
    for name, result in full.items():
        data["artifacts"][verify.phase_folder(result) + "/" + name + ".json"] = result
    if manifest["condition"] != "LATENCY":
        add_phase_artifacts(data, primary_meta, {k: full[k] for k in post_names}, native_ref, native_inputs)
    for swap in data["swaps"]:
        add_phase_artifacts(data, swap, swap["postflight"], native_ref, native_inputs)
    roles = data["swaps"][-1]["resident_roles"] if data["swaps"] else primary_meta["resident_roles"]
    manifest["final_cleanup"] = {r: {"pids": [9000 + i], "exitcodes": [0], "cleanup_confirmed": True, "seconds": .1}
                                 for i, r in enumerate(roles)}
    if cfg["params"]["SPLIT"] == "test" and (root / "budget_approval.json").exists():
        bind_approval(data, root / "budget_approval.json")
    refresh_traces(data)


def add_phase_artifacts(data, metadata, detail, native_ref, native_inputs):
    folder = verify.phase_folder(metadata)
    raw = b"synthetic current phase engine log\n"
    detail["detailed_trace_audit"]["engine_log"] = {"start_byte": 0, "end_byte": len(raw),
                    "sha256": verify.sha256(raw), "current_phase_only": True}
    data["artifacts"][folder + "/vllm_phase.log"] = raw
    data["artifacts"][folder + "/summary.json"] = detail
    data["artifacts"][folder + "/detailed_checks.jsonl"] = [{"check": name, "result": result} for name, result in detail.items()]
    for name, result in detail.items():
        data["artifacts"][folder + "/" + name + ".json"] = result
    if "J" in metadata["resident_roles"]:
        data["artifacts"][folder + "/native_reference_trace.json"] = {**{k: metadata[k] for k in (
            "phase_epoch", "phase_id", "gpu_uuid", "resident_roles")}, "output": native_ref["native"],
            "input": {"items": native_inputs}, "identity": native_ref["identity"], "reference_id": native_ref["reference_id"]}


def bind_approval(data, sentinel):
    value = json.loads(sentinel.read_bytes())
    data["manifest"]["authorization"] = {"required": True, "checked": True,
        "direct_api_gate": True, "gpu_hour_meter": "operator-owned",
        **{k: value[k] for k in ("approved_by", "approved_utc", "max_gpu_hours", "pilot_report_sha256")},
        "scope": "latency" if data["manifest"]["condition"] == "LATENCY" else "test",
        "pilot_archive_sha256": {arm: entry["verification"]["archive_sha256"] for arm, entry in
            json.loads(sentinel.with_name("pilot_go.json").read_bytes())["conditions"].items()},
        "approval_sha256": verify.sha256(sentinel.read_bytes())}


def options(evidence, condition="JFINAL", split="pilot"):
    _, inputs, config, prompts, _ = evidence
    return dict(inputs=inputs, config=config, prompts=prompts, expected_count=50 if split == "pilot" else 500,
                split=split, run_tag="mock", condition=condition,
                **({"pilot_report": evidence[0] / "pilot_go.json"} if split == "test" else {}))


def write_archive(directory, data):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (data["manifest"]["run_name"] + "_final.zip")
    with zipfile.ZipFile(path, "w") as z:
        for name in verify.LEDGERS:
            z.writestr("run/" + name + ".jsonl", lines(data[name]))
        for name in ("manifest", "progress"):
            z.writestr("run/" + name + ".json", json.dumps(data[name]))
        if data.get("latency_plan"):
            z.writestr("run/latency_plan.json", json.dumps(data["latency_plan"]))
        for name, artifact in data.get("artifacts", {}).items():
            z.writestr("run/" + name, artifact if isinstance(artifact, bytes) else lines(artifact) if name.endswith(".jsonl") else json.dumps(artifact))
    nb = directory / (data["manifest"]["condition"] + "_mock.out.timestamp.ipynb")
    nb.write_text(json.dumps({"metadata": {"papermill": {"end_time": "end", "exception": False}},
                             "cells": [{"cell_type": "code", "source": [
                                 "\n".join(f"{k} = {v!r}" for k, v in data["manifest"]["config"]["params"].items())],
                                 "outputs": [{"output_type": "stream", "text": [str(path)]}],
                                 "metadata": {"tags": ["parameters"], "papermill": {"status": "completed"}}}]}))
    return path


@pytest.mark.parametrize("condition", verify.CONDITIONS)
def test_physical_coverage(evidence, condition):
    data = snapshot(evidence, condition)
    info = verify.verify_records(data, **options(evidence, condition))
    assert info["records"] == (150 if condition in verify.SAMPLED else 50)


def test_real_continuity_probe_advisory_label_still_requires_success(evidence):
    data = snapshot(evidence)
    data["manifest"]["preflight"]["continuity"]["blocking"] = False
    assert verify.verify_records(data, **options(evidence))["ok"]
    data["manifest"]["preflight"]["continuity"]["ok"] = False
    with pytest.raises(ValueError, match="cache continuity"):
        verify.verify_records(data, **options(evidence))


def test_a10040_visible_memory_matches_runtime_range_not_nominal_byte_equality(evidence):
    data = snapshot(evidence)
    data["manifest"]["environment"]["gpus"][0]["memory.total"] = 40536
    assert verify.verify_records(data, **options(evidence))["ok"]


@pytest.mark.parametrize("fault", ["unlogged_answer", "foreign_seed", "no_winner"])
def test_jfinal_prediction_must_belong_to_its_logged_selected_pool(evidence, fault):
    data = snapshot(evidence)
    if fault == "unlogged_answer":
        data["predictions"][0]["public_output"] = data["metrics"][0]["public_output"] = "Invented selected answer"
    elif fault == "foreign_seed":
        data["proposals"][0]["candidates"][0]["seed"] = 29
    else:
        data["predictions"][0].pop("winner_branch")
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "seed", "effective", "hash", "preflight", "cache",
                                   "cuda", "gpu", "lock", "metrics", "proposals", "branches", "clock"])
def test_fail_closed(evidence, fault):
    data = snapshot(evidence)
    if fault == "missing":
        data["predictions"].pop()
    elif fault == "duplicate":
        data["predictions"][-1] = data["predictions"][0]
    elif fault == "seed":
        data["predictions"][0]["seed"] = 19
    elif fault == "effective":
        data["predictions"][0]["effective_seed"] = None
    elif fault == "hash":
        data["predictions"][0]["config_hash"] = "bad"
    elif fault == "preflight":
        data["manifest"]["preflight"]["lock"]["ok"] = False
    elif fault == "cache":
        del data["manifest"]["preflight"]["continuity"]
    elif fault == "cuda":
        data["manifest"]["environment"]["cuda_available"] = False
    elif fault == "gpu":
        data["manifest"]["environment"]["gpus"][0]["memory.total"] = 81920
    elif fault == "lock":
        del data["manifest"]["model_lock"]["J"]
    elif fault == "metrics":
        data["metrics"][0]["problem_id"] = "unknown"
    elif fault == "proposals":
        data["proposals"].pop()
    elif fault == "branches":
        data["proposals"][0]["candidates"][1]["branch"] = 0
    else:
        data["predictions"][0]["t_end_utc"] = data["predictions"][0]["t_start_utc"]
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence))


def test_failed_cases_retained_and_no_candidates(evidence):
    data = snapshot(evidence)
    data["predictions"][0]["status"] = "engine_error"
    data["metrics"][0]["status"] = "engine_error"
    data["predictions"][0].pop("winner_branch")
    data["metrics"][0].pop("winner_branch")
    data["proposals"][0].update(status="no_candidates", candidates=[])
    refresh_traces(data)
    assert verify.verify_records(data, **options(evidence))["records"] == 150
    assert pilot.failure_rates(data, 150)["infra_count"] == 1


def test_no_gold_access_even_hash_map(evidence, monkeypatch):
    data = snapshot(evidence)
    original = Path.read_bytes
    def guarded(path):
        assert "gold" not in path.name.lower(), "Attempted gold read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert verify.verify_records(data, **options(evidence))["ok"]
    data["manifest"]["config"]["data_sha256"]["pilot_gold.jsonl"] = "never-read"
    data["manifest"]["config_hash"] = verify.canonical_hash(data["manifest"]["config"])
    with pytest.raises(ValueError, match="gold forbidden"):
        verify.verify_records(data, **options(evidence))


def test_archive_and_notebook(evidence):
    root = evidence[0]
    data = snapshot(evidence)
    path = write_archive(root, data)
    opts = options(evidence)
    opts.pop("condition")
    report = verify.verify_run(root, conditions=["JFINAL"], **opts)
    assert report["ok"], report
    assert verify.read_archive(path)["archive_sha256"] == verify.sha256(path.read_bytes())
    next(root.glob("*.ipynb")).write_text("{}")
    assert not verify.verify_run(root, conditions=["JFINAL"], **opts)["ok"]


@pytest.mark.parametrize("fault", ["wrong_tag", "wrong_archive", "no_parameters", "wrong_output_type"])
def test_completed_unrelated_notebook_and_missing_output_do_not_pass(evidence, fault):
    root = evidence[0]
    data = snapshot(evidence)
    write_archive(root, data)
    notebook = next(root.glob("*.ipynb"))
    nb = json.loads(notebook.read_text())
    if fault == "wrong_tag":
        nb["cells"][0]["source"][0] = nb["cells"][0]["source"][0].replace("RUN_TAG = 'mock'", "RUN_TAG = 'unrelated'")
    elif fault == "wrong_archive":
        nb["cells"][0]["outputs"][0]["text"] = ["another_run_final.zip"]
    elif fault == "no_parameters":
        nb["cells"][0]["source"] = ["pass"]
    else:
        data["predictions"][0]["public_output"] = None
        data["metrics"][0]["public_output"] = None
        with pytest.raises(ValueError, match="public_output"):
            verify.verify_records(data, **options(evidence))
        return
    notebook.write_text(json.dumps(nb))
    opts = options(evidence)
    opts.pop("condition")
    assert not verify.verify_run(root, conditions=["JFINAL"], **opts)["ok"]


@pytest.mark.parametrize("fault", ["unsafe_zip", "duplicate_zip", "gold_zip", "unsealed", "human", "policy", "input_changed"])
def test_provenance_boundaries(evidence, fault):
    root = evidence[0]
    data = snapshot(evidence)
    if fault.endswith("zip"):
        path = write_archive(root, data)
        name = {"unsafe_zip": "../escape.json", "duplicate_zip": "run/manifest.json",
                 "gold_zip": "run/test_gold.jsonl"}[fault]
        with pytest.warns(UserWarning, match="Duplicate name") if fault == "duplicate_zip" else nullcontext():
            with zipfile.ZipFile(path, "a") as z:
                z.writestr(name, "synthetic-not-real-data")
        with pytest.raises(ValueError):
            verify.read_archive(path)
    else:
        if fault == "input_changed":
            evidence[1].write_text(evidence[1].read_text() + "\n")
        else:
            metadata = root / "dataset_manifest.json"
            obj = json.loads(metadata.read_text())
            if fault == "unsealed":
                obj["sealed"] = False
            elif fault == "human":
                obj["review"]["human_reviewed"] = True
            else:
                del obj["review"]["policy_sha256"]
            metadata.write_text(json.dumps(obj))
            seal_public_fixture(root, (evidence[1], metadata, root / "REVIEW_CONTRACT.md"))
        with pytest.raises(ValueError):
            verify.verify_records(data, **options(evidence))


def test_missing_pilot_arm_stops(evidence):
    report = pilot.decide(evidence[0], evidence[1], config=evidence[2], prompts=evidence[3])
    assert not report["go"] and report["aggregate"]["denominator"] == 600
    assert len(report["errors"]) == 6


@pytest.mark.parametrize("failures,go", [(2, True), (3, False)])
def test_pilot_technical_only_all_six(evidence, failures, go):
    results = evidence[0] / "results"
    for condition in verify.CONDITIONS:
        data = snapshot(evidence, condition)
        if condition == "G_GREEDY":
            for row in data["predictions"][:failures]:
                row["status"] = "timeout"
            for row in data["metrics"][:failures]:
                row["status"] = "timeout"
            refresh_traces(data)
        directory = results / condition
        write_archive(directory, data)
        opts = options(evidence, condition)
        opts.pop("condition")
        record = verify.verify_run(directory, conditions=[condition], **opts)
        assert record["ok"], record
        (directory / "verification.json").write_text(json.dumps(record))
    report = pilot.decide(results, evidence[1], config=evidence[2], prompts=evidence[3])
    assert report["go"] is go, report
    assert report["aggregate"]["denominator"] == report["aggregate"]["observed_cases"] == 600
    assert report["actual_cu"] is None and report["budget_approval"] is False
    assert report["estimated_test_walltime_s"] == 24000
    assert report["estimated_latency_walltime_s"] == 7200
    assert "accuracy" not in json.dumps(report)


def test_union_not_sum():
    data = {"predictions": [{"problem_id": "a", "seed": 17, "status": "timeout"}],
            "failures": [{"problem_id": "a", "seed": 17, "kind": "infraerror"}] * 3}
    assert pilot.failure_rates(data, 50)["timeout_infra_count"] == 1


def full_evidence(evidence):
    root, _, config, prompts, frozen = evidence
    inputs = root / "test_inputs.jsonl"
    inputs.write_text(lines([{"id": f"t{i}", "problem": f"Test problem {i}", "language": "en"} for i in range(500)]))
    metadata = root / "dataset_manifest.json"
    subset = root / "latency_inputs.jsonl"
    subset.write_text(lines(verify.jsonl(inputs.read_text())[:100]))
    plan = root / "latency_plan.json"
    plan.write_text(json.dumps({"quota_axis": verify.QUOTA_AXIS, "quota_per_domain": verify.SAMPLING_QUOTAS["latency"],
        "intrinsic_difficulty_counts": {"easy": 100, "medium": 0, "hard": 0}, "items": [
        {"item_id": f"t{i}", "problem": f"Test problem {i}", "domain": f"domain{i // 20}",
         "difficulty": "easy", "sampling_tier": "easy" if i % 20 < 6 else "medium" if i % 20 < 14 else "hard"} for i in range(100)]}))
    seal_public_fixture(root, (inputs, root / "pilot_inputs.jsonl", metadata, root / "REVIEW_CONTRACT.md", subset, plan))
    full = root, inputs, config, prompts, frozen
    approved(full)
    return full


def approved(evidence):
    if (evidence[0] / "budget_approval.json").exists():
        return evidence[0] / "budget_approval.json"
    report_path = evidence[0] / "pilot_go.json"
    report = {"protocol_version": "3", "code_version": "0.3.0", "go": True, "decision": "go", "errors": [],
        "budget_approval": False, "conditions": {}, "aggregate": {"complete": True, "denominator": 600,
        "observed_cases": 600, "timeout_infra_count": 0, "timeout_infra_rate": 0.}}
    for arm in verify.CONDITIONS:
        count = 150 if arm in verify.SAMPLED else 50
        pilot_evidence = (evidence[0], evidence[0] / "pilot_inputs.jsonl", *evidence[2:])
        snap = snapshot(pilot_evidence, arm)
        path = write_archive(evidence[0] / "public_pilot" / arm, snap)
        report["conditions"][arm] = {"verified": True, "denominator": count,
            "verification": {"ok": True, "condition": arm, "records": count,
                             "archive": str(path), "archive_sha256": verify.sha256(path.read_bytes()),
                             "notebook_sha256": verify.sha256(next(path.parent.glob("*.ipynb")).read_bytes()),
                             "config_hash": snap["manifest"]["config_hash"]},
            "rates": verify.failure_rates(snap, count)}
    report_path.write_text(json.dumps(report))
    path = evidence[0] / "budget_approval.json"
    path.write_text(json.dumps({"approved": True, "config_sha256": verify.sha256(evidence[2].read_bytes()),
             "scope": "test_and_latency", "approved_by": "synthetic-only",
             "max_gpu_hours": 25, "pilot_report_sha256": verify.sha256(report_path.read_bytes()),
             "approved_utc": "2025-12-31T00:00:00+00:00"}))
    return path


def latency_snapshot(evidence):
    data = snapshot(evidence, "LATENCY", "test")
    subset = evidence[0] / "latency_inputs.jsonl"
    data["manifest"]["config"]["data_sha256"][subset.name] = verify.sha256(subset.read_bytes())
    plan = evidence[0] / "latency_plan.json"
    data["manifest"]["config"]["data_sha256"][plan.name] = verify.sha256(plan.read_bytes())
    data["manifest"]["config_hash"] = verify.canonical_hash(data["manifest"]["config"])
    digest = data["manifest"]["config_hash"]
    items = []
    for domain in range(5):
        for slot in range(20):
            i = domain * 20 + slot
            items.append({"item_id": f"t{i}", "problem": f"Test problem {i}", "domain": f"domain{domain}",
                          "difficulty": "easy", "sampling_tier": "easy" if slot < 6 else "medium" if slot < 14 else "hard"})
    data["latency_plan"] = {"items": items}
    rows, proposals = [], []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for item in items:
        for arm in ("JFINAL", "B13_GREEDY"):
            for seed in verify.SEEDS:
                for repetition in range(3):
                    i = len(rows)
                    common = {"item_id": item["item_id"], "problem_id": item["item_id"], "condition": arm,
                              **{k: item[k] for k in ("domain", "difficulty", "sampling_tier")},
                              "profile": data["manifest"]["config"]["profile"], "public_output": "Ungraded synthetic output",
                              "accepted_tokens": 20, "candidate_tokens_total": 20,
                              "seed": seed, "effective_seed": seed if arm == "JFINAL" else None,
                               "repetition": repetition, "status": "final", "config_hash": digest,
                              "resume_key": verify.sha256(f"{digest}|{item['item_id']}|{arm}|{seed}|{repetition}".encode()),
                              "clock_scope": "candidate_generation_inclusive", "execution_id": f"exec-{i}",
                              "gpu_uuid": "GPU-mock", "candidate_set_id": f"set-{i}",
                              "T_total": 4, "T_first_accepted": 4, "T_proposals": 3, "T_selector": 1,
                              "t_start_perf": i * 5., "t_end_perf": i * 5. + 4.,
                              "t_start_utc": (start + timedelta(seconds=i * 5)).isoformat(),
                              "t_end_utc": (start + timedelta(seconds=i * 5 + 4)).isoformat()}
                    rows.append(common)
                    if arm == "JFINAL":
                        common["winner_branch"] = 0
                        proposals.append({**common, "status": "complete", "candidates": [{"branch": b, "seed": seed,
                            "seed_branch": seed + b, "status": "final", "public_output": common["public_output"]} for b in range(4)]})
    data.update(latency_metrics=rows, predictions=copy.deepcopy(rows), metrics=copy.deepcopy(rows),
                proposals=proposals, progress={"done": 1800, "total": 1800})
    data["manifest"]["counts"] = {name: len(data[name]) for name in verify.LEDGERS}
    complete_fixture(data, evidence)
    return data


@pytest.mark.parametrize("condition", verify.CONDITIONS)
def test_full_test_requires_separate_preexisting_approval(evidence, condition):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, condition, "test")
    opts = options(evidence, condition, "test")
    with pytest.raises(ValueError, match="Separate budget approval"):
        verify.verify_records(data, **opts)
    info = verify.verify_records(data, **opts, approval=approved(evidence))
    assert info["records"] == (1500 if condition in verify.SAMPLED else 500)
    sentinel = approved(evidence)
    value = json.loads(sentinel.read_text())
    value["approved_utc"] = "2026-01-02T00:00:00+00:00"
    sentinel.write_text(json.dumps(value))
    bind_approval(data, sentinel)
    with pytest.raises(ValueError, match="precede"):
        verify.verify_records(data, **opts, approval=sentinel)


@pytest.mark.parametrize("fault", [None, "cloned", "execution_id", "strata", "effective", "scope", "missing", "proposals"])
def test_latency_actual_1800(evidence, fault):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    if fault == "cloned":
        for field in ("t_start_utc", "t_end_utc"):
            data["latency_metrics"][1][field] = data["latency_metrics"][0][field]
    elif fault == "execution_id":
        data["latency_metrics"][1]["execution_id"] = data["latency_metrics"][0]["execution_id"]
    elif fault == "strata":
        data["latency_plan"]["items"][0]["difficulty"] = "hard"
    elif fault == "effective":
        next(r for r in data["latency_metrics"] if r["condition"] == "B13_GREEDY")["effective_seed"] = 17
    elif fault == "scope":
        data["latency_metrics"][0]["clock_scope"] = "selector_only"
    elif fault == "missing":
        data["latency_metrics"].pop()
    elif fault == "proposals":
        data["proposals"].pop()
    opts = options(evidence, "LATENCY", "test")
    if fault is None:
        assert verify.verify_records(data, **opts, approval=approved(evidence))["records"] == 1800
    else:
        with pytest.raises(ValueError):
            verify.verify_records(data, **opts, approval=approved(evidence))


def test_latency_companion_persistence_can_end_after_measurement(evidence):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    for name in ("predictions", "metrics"):
        data[name][0]["t_end_utc"] = "2026-01-01T00:00:04.050000+00:00"
    assert verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))["records"] == 1800


def test_latency_cannot_replace_frozen_ids_while_preserving_all_quotas(evidence):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    data["latency_plan"]["items"][0].update(item_id="t200", problem="Test problem 200")
    with pytest.raises(ValueError, match="frozen test subset"):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


@pytest.mark.parametrize("fault", ["cloned_perf", "perf_total", "first", "first_none", "wrong_profile", "fractional_seed", "fractional_rep"])
def test_latency_repetition_clocks_and_timepoints_are_strict(evidence, fault):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    row = data["latency_metrics"][0]
    if fault == "cloned_perf":
        for field in ("t_start_perf", "t_end_perf"):
            data["latency_metrics"][1][field] = row[field]
    elif fault == "perf_total":
        row["t_end_perf"] += .1
    elif fault == "first":
        row["T_first_accepted"] = 3.9
    elif fault == "first_none":
        row["T_first_accepted"] = None
    elif fault == "wrong_profile":
        row["profile"] = "historical"
    elif fault == "fractional_seed":
        row["seed"] = 17.
    else:
        row["repetition"] = 0.
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


@pytest.mark.parametrize("field,value", [("accepted_tokens", 1025), ("accepted_tokens", True),
                                        ("candidate_tokens_total", 1025), ("T_first_accepted", -1)])
def test_measured_case_budget_counters_are_validated_not_only_frozen_settings(evidence, field, value):
    data = snapshot(evidence, "B13_GREEDY")
    data["predictions"][0][field] = data["metrics"][0][field] = value
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence, "B13_GREEDY"))


@pytest.mark.parametrize("fault", ["hash", "scope", "not_file"])
def test_budget_approval_is_file_bound_to_exact_config_and_phase(evidence, fault):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, "B13_GREEDY", "test")
    sentinel = approved(evidence)
    value = json.loads(sentinel.read_text())
    if fault == "hash":
        value["config_sha256"] = "b" * 64
    elif fault == "scope":
        value["scope"] = "pilot"
    else:
        sentinel = value
    if isinstance(sentinel, Path):
        sentinel.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="approval|Approval"):
        verify.verify_records(data, **options(evidence, "B13_GREEDY", "test"), approval=sentinel)


@pytest.mark.parametrize("tool", ["verify_run_v3", "pilot_decision_v3"])
def test_cli_help(tool):
    result = subprocess.run([sys.executable, str(ROOT / "tools" / (tool + ".py")), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--gold" not in result.stdout


@pytest.mark.parametrize("field,value", [("approved", 1), ("approved_by", "  "), ("approved_by", 123),
    ("approved_utc", "2025-12-31T00:00:00"), ("approved_utc", "2025-12-31T01:00:00+01:00"),
    ("pilot_report_sha256", None), ("max_gpu_hours", None), ("max_gpu_hours", True),
    ("max_gpu_hours", 25.01), ("max_gpu_hours", 0), ("max_gpu_hours", float("inf"))])
def test_actual_operator_consent_schema_fails_closed(evidence, field, value):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, "B13_GREEDY", "test")
    sentinel = approved(evidence)
    approval = json.loads(sentinel.read_bytes())
    if value is None:
        approval.pop(field)
    else:
        approval[field] = value
    sentinel.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence, "B13_GREEDY", "test"), approval=sentinel)


@pytest.mark.parametrize("fault", ["checked", "sha", "reformatted", "pilot_bytes", "missing_pilot", "pilot_archive"])
def test_consent_original_bytes_and_checked_manifest_not_just_json_equivalence(evidence, fault):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, "B13_GREEDY", "test")
    sentinel = approved(evidence)
    opts = options(evidence, "B13_GREEDY", "test")
    if fault == "checked":
        data["manifest"]["authorization"]["checked"] = False
    elif fault == "sha":
        data["manifest"]["authorization"]["approval_sha256"] = "a" * 64
    elif fault == "reformatted":
        sentinel.write_text(json.dumps(json.loads(sentinel.read_bytes()), indent=2))
    elif fault == "pilot_bytes":
        opts["pilot_report"].write_bytes(opts["pilot_report"].read_bytes() + b"\n")
    elif fault == "missing_pilot":
        opts.pop("pilot_report")
    else:
        report = json.loads(opts["pilot_report"].read_bytes())
        Path(report["conditions"]["G_SINGLE"]["verification"]["archive"]).write_bytes(b"corrupt")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        verify.verify_records(data, **opts, approval=sentinel)


def test_legacy_named_consent_uses_passed_original_file_bytes(evidence):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, "B13_GREEDY", "test")
    sentinel = approved(evidence)
    legacy = evidence[0] / "legacy_user_consent.json"
    legacy.write_bytes(sentinel.read_bytes())
    assert verify.verify_records(data, **options(evidence, "B13_GREEDY", "test"), approval=legacy)["ok"]


@pytest.mark.parametrize("fault", ["no_swaps", "failed_swap", "other_gpu", "zero_epoch", "missing_epoch",
    "missing_details", "missing_check", "missing_log", "native_trace", "native_vectors", "native_ids",
    "role_overlap", "open_process", "no_pids", "unclosed_role", "stale_phase", "resume", "proposal_seed",
    "proposal_stage", "foreign_candidate_hash", "foreign_proposal_condition", "no_decision"])
def test_latency_certificate_requires_real_lifecycle_and_bound_traces(evidence, fault):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    swap, row, pool = data["swaps"][0], data["latency_metrics"][0], data["proposals"][0]
    folder = verify.phase_folder(swap)
    if fault == "no_swaps":
        data["swaps"] = []
    elif fault == "failed_swap":
        swap["ok"] = False
    elif fault == "other_gpu":
        swap["gpu_uuid"] = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    elif fault == "zero_epoch":
        swap["phase_epoch"] = 0
    elif fault == "missing_epoch":
        row.pop("phase_epoch")
    elif fault == "missing_details":
        data["artifacts"].pop(folder + "/summary.json")
    elif fault == "missing_check":
        data["artifacts"].pop(folder + "/stress.json")
    elif fault == "missing_log":
        data["artifacts"].pop(folder + "/detailed_checks.jsonl")
    elif fault == "native_trace":
        data["artifacts"][folder + "/native_reference_trace.json"]["output"] = {}
    elif fault == "native_vectors":
        swap["postflight"]["selector_equivalence"]["rows"][0]["vllm_logits"] = [True] * 4
    elif fault == "native_ids":
        swap["postflight"]["selector_equivalence"]["rows"][0]["ids"] = [99] * 10
    elif fault == "role_overlap":
        swap["resident_roles"].append("B")
    elif fault == "open_process":
        data["swaps"][1]["closed"]["G"]["exitcodes"] = [None]
    elif fault == "no_pids":
        data["swaps"][1]["closed"]["G"]["pids"] = []
    elif fault == "unclosed_role":
        data["swaps"][1]["closed"].pop("J")
    elif fault == "stale_phase":
        row["phase_id"] = data["swaps"][1]["phase_id"]
    elif fault == "resume":
        row["resume_key"] = "a" * 64
    elif fault == "proposal_seed":
        pool["candidates"][0]["seed_branch"] += 1
    elif fault == "proposal_stage":
        pool["generation"] = 1
    elif fault == "foreign_candidate_hash":
        pool["candidates"][0]["config_hash"] = "a" * 64
    elif fault == "foreign_proposal_condition":
        pool["condition"] = "B13_GREEDY"
    else:
        data["decisions"].pop(0)
    data["manifest"]["counts"] = {name: len(data[name]) for name in verify.LEDGERS}
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


def test_recognized_soft_timeout_partial_candidates_preserve_all1800_cases(evidence):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    for name in ("latency_metrics", "predictions", "metrics"):
        data[name][0].update(status="timeout", T_first_accepted=None)
        data[name][0].pop("winner_branch")
    pool = data["proposals"][0]
    pool.update(status="partial", candidates=pool["candidates"][:2])
    pool["candidates"][1]["status"] = "timeout"
    refresh_traces(data)
    result = verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))
    assert result["records"] == 1800 and result["statuses"]["timeout"] == 1
    data["failures"] = []
    data["manifest"]["counts"]["failures"] = 0
    with pytest.raises(ValueError, match="failure trace"):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


def test_four_returned_timeout_candidates_are_not_fake_complete(evidence):
    data = snapshot(evidence)
    data["proposals"][0]["candidates"][3]["status"] = "timeout"
    refresh_traces(data)
    with pytest.raises(ValueError, match="complete pool"):
        verify.verify_records(data, **options(evidence))


@pytest.mark.parametrize("name", ["run//alias.json", "run/./alias.json", "run/alias.json/../escape", "run/alias.json ",
                                "RUN/MANIFEST.JSON", "run/CON.json", "run/evil?.json"])
def test_zip_canonical_aliases_rejected_before_member_decompression(evidence, name, monkeypatch):
    path = write_archive(evidence[0], snapshot(evidence))
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(name, "unknown")
    monkeypatch.setattr(zipfile.ZipFile, "open", lambda *a, **k: pytest.fail("Decompressed before canonical path validation"))
    with pytest.raises(ValueError):
        verify.read_archive(path)


def test_zip_reads_only_minimum_known_members_and_enforces_member_bound(evidence, monkeypatch):
    path = write_archive(evidence[0], snapshot(evidence))
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run/ignored.log", "x" * 1000000)
    original = zipfile.ZipFile.open
    def guarded(archive, entry, *args, **kwargs):
        assert (entry.filename if isinstance(entry, zipfile.ZipInfo) else entry) != "run/ignored.log"
        return original(archive, entry, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda *a: pytest.fail("testzip decompresses unknown members"))
    assert verify.read_archive(path)["manifest"]
    monkeypatch.setattr(verify, "MAX_MEMBER_BYTES", 16)
    with pytest.raises(ValueError, match="decompression byte limit"):
        verify.read_archive(path)


def test_zip_known_member_crc_corruption_is_not_ignored(evidence):
    path = write_archive(evidence[0], snapshot(evidence))
    content = bytearray(path.read_bytes())
    index = content.index(b"Ungraded synthetic output")
    content[index] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="integrity failure"):
        verify.read_archive(path)


def test_pilot_report_pinned_bytes_timestamp_and_atomic_exclusive_publish(evidence, monkeypatch):
    report = pilot.decide(evidence[0], evidence[1], config=evidence[2], prompts=evidence[3])
    output = evidence[0] / "decision.json"
    pilot.write_report(copy.deepcopy(report), output)
    pinned = output.read_bytes(), output.with_suffix(".md").read_bytes()
    pilot.write_report(copy.deepcopy(report), output)
    assert pinned == (output.read_bytes(), output.with_suffix(".md").read_bytes())
    changed = copy.deepcopy(report)
    changed["errors"].append("New evidence requires a new report")
    with pytest.raises(ValueError, match="new filename"):
        pilot.write_report(changed, output)
    assert pinned == (output.read_bytes(), output.with_suffix(".md").read_bytes())
    new = evidence[0] / "race.json"
    original = pilot.os.link
    def raced(source, target):
        Path(target).write_bytes(b"concurrent pinned evidence")
        return original(source, target)
    monkeypatch.setattr(pilot.os, "link", raced)
    with pytest.raises(ValueError, match="new filename"):
        pilot.write_report(copy.deepcopy(report), new)
    assert new.read_bytes() == b"concurrent pinned evidence"


@pytest.mark.parametrize("fault", ["no_phase", "foreign_gpu", "stale_row", "missing_log", "corrupt_log", "pending_only", "final_cleanup"])
def test_physical_pilot_requires_actual_successful_resident_phase(evidence, fault):
    data = snapshot(evidence)
    phase = data["manifest"]["active_phase"]
    folder = verify.phase_folder(phase)
    if fault == "no_phase":
        phase["phase_id"] = None
    elif fault == "foreign_gpu":
        phase["gpu_uuid"] = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    elif fault == "stale_row":
        data["predictions"][0]["phase_epoch"] = 0
    elif fault == "missing_log":
        data["artifacts"].pop(folder + "/vllm_phase.log")
    elif fault == "corrupt_log":
        data["artifacts"][folder + "/vllm_phase.log"] += b"corruption"
    elif fault == "pending_only":
        log = data["artifacts"][folder + "/detailed_checks.jsonl"]
        log[-1] = {**log[-1], "result": {**log[-1]["result"], "ok": False}}
    else:
        data["manifest"]["final_cleanup"]["G"]["exitcodes"] = [None]
    with pytest.raises(ValueError):
        verify.verify_records(data, **options(evidence))


def test_soft_timeout_allows_bounded_cleanup_not_faked_fast_completion(evidence):
    data = snapshot(evidence, "B13_GREEDY")
    data["manifest"]["config"]["params"]["TIMEOUT_S"] = 2
    data["predictions"][0].update(status="timeout", T_first_accepted=None)
    for row in data["predictions"][1:]:
        row.update(T_total=2, T_first_accepted=2, T_proposals=2,
                   t_end_utc=(datetime.fromisoformat(row["t_start_utc"]) + timedelta(seconds=2)).isoformat())
    complete_fixture(data, evidence)
    assert verify.verify_records(data, **options(evidence, "B13_GREEDY"))["records"] == 50
    data["predictions"][0]["status"] = data["metrics"][0]["status"] = "final"
    data["predictions"][0]["T_first_accepted"] = data["metrics"][0]["T_first_accepted"] = 4
    refresh_traces(data)
    with pytest.raises(ValueError, match="timeout/confirmed cleanup budget"):
        verify.verify_records(data, **options(evidence, "B13_GREEDY"))


def test_recovered_engine_exception_binds_pending_and_confirmed_terminal_traces(evidence):
    data = snapshot(evidence, "B13_GREEDY")
    for name in ("predictions", "metrics"):
        data[name][0].update(status="engine_error", T_first_accepted=None,
                             engine_cleanup={"cleanup_confirmed": True}, engine_usable=True)
    refresh_traces(data)
    case = data["predictions"][0]
    failure = {"kind": "engine_exception", "engine_role": "B", "attempted_ids": ["synthetic-rid"],
               "error": "Synthetic recovered error", "traceback": "Synthetic traceback only",
               "config_hash": case["config_hash"], "ts": "2026-01-01T00:00:01+00:00",
               **{k: case[k] for k in ("phase_epoch", "phase_id", "gpu_uuid")}}
    data["failures"].extend([{**failure, "cleanup_pending": True}, {**failure, "cleanup_pending": False,
        "engine_cleanup": {"cleanup_confirmed": True}, "engine_usable": True}])
    data["manifest"]["counts"]["failures"] = len(data["failures"])
    assert verify.verify_records(data, **options(evidence, "B13_GREEDY"))["records"] == 50
    data["failures"][-1]["engine_cleanup"]["cleanup_confirmed"] = False
    with pytest.raises(ValueError, match="terminal engine failure cleanup"):
        verify.verify_records(data, **options(evidence, "B13_GREEDY"))


@pytest.mark.parametrize("linked", [False, True])
def test_engine_failure_request_metadata_preserves_case_failure_and_rejects_foreign_case(evidence, linked):
    data = snapshot(evidence, "B13_GREEDY")
    for name in ("predictions", "metrics"):
        data[name][0].update(status="engine_error", T_first_accepted=None,
            engine_cleanup={"cleanup_confirmed": True, "attempted_ids": ["actual-rid"]}, engine_usable=True)
    refresh_traces(data)
    case = data["predictions"][0]
    base = {"kind": "engine_exception", "engine_role": "B", "attempted_ids": ["actual-rid"],
            "config_hash": case["config_hash"], "error": "Synthetic recovery", "traceback": "Synthetic traceback",
            "ts": "2026-01-01T00:00:01+00:00", **{k: case[k] for k in ("phase_epoch", "phase_id", "gpu_uuid")}}
    if linked:
        base.update({k: case[k] for k in ("resume_key", "problem_id", "seed", "condition")})
    data["failures"].extend([{**base, "cleanup_pending": True}, {**base, "cleanup_pending": False,
        "engine_cleanup": {"cleanup_confirmed": True}, "engine_usable": True}])
    data["manifest"]["counts"]["failures"] = len(data["failures"])
    assert verify.verify_records(data, **options(evidence, "B13_GREEDY"))["records"] == 50
    data["failures"][-1]["resume_key"] = data["predictions"][1]["resume_key"]
    with pytest.raises(ValueError, match="Unbound engine failure"):
        verify.verify_records(data, **options(evidence, "B13_GREEDY"))


def test_confirmed_warmup_engine_cleanup_is_not_a_missing_planned_case(evidence):
    data = snapshot(evidence, "B13_GREEDY")
    case = data["predictions"][0]
    warmup = {**case, "problem_id": "development-warmup", "generation": 900, "outside_T_total": True,
              "status": "engine_error", "engine_cleanup": {"attempted_ids": ["warmup-rid"], "cleanup_confirmed": True}}
    data["warmup"] = [warmup]
    trace = {"kind": "engine_exception", "engine_role": "B", "attempted_ids": ["warmup-rid"],
        "config_hash": case["config_hash"], "error": "Synthetic warmup recovery", "traceback": "Synthetic trace",
        "ts": "2025-12-31T23:59:59.500000+00:00", **{k: case[k] for k in ("phase_epoch", "phase_id", "gpu_uuid")}}
    data["failures"] = [{**trace, "cleanup_pending": True}, {**trace, "cleanup_pending": False,
        "engine_usable": True, "engine_cleanup": {"cleanup_confirmed": True}}]
    data["manifest"]["counts"] = {n: len(data[n]) for n in verify.LEDGERS}
    assert verify.verify_records(data, **options(evidence, "B13_GREEDY"))["records"] == 50
    assert pilot.failure_rates(data, 50)["timeout_infra_count"] == 0


@pytest.mark.parametrize("split", ["test", "pilot", "latency"])
def test_source_sampling_tier_quotas_allow_no_intrinsic_hard_items(split):
    rows = [{"domain": f"d{d}", "sampling_tier": tier, "difficulty": "easy"}
            for d in range(5) for tier, count in verify.SAMPLING_QUOTAS[split].items() for _ in range(count)]
    profile = verify.sampling_profile(rows, split)
    assert profile["source_sampling_tier_counts"]["hard"] == 5 * verify.SAMPLING_QUOTAS[split]["hard"]
    assert profile["intrinsic_difficulty_counts"] == {"easy": len(rows), "medium": 0, "hard": 0}
    rows[0]["sampling_tier"] = "hard"
    with pytest.raises(ValueError, match="sampling-tier quota"):
        verify.sampling_profile(rows, split)


@pytest.mark.parametrize("fault", ["axis", "missing_axis", "quota", "intrinsic_claim", "intrinsic_boolean", "intrinsic_sum", "policy_axis"])
def test_public_profile_rejects_intrinsic_quota_and_false_source_policy(evidence, fault):
    root, inputs, config, prompts, frozen = evidence
    path = root / "dataset_manifest.json"
    metadata = json.loads(path.read_bytes())
    if fault == "axis":
        metadata["quota_axis"] = "intrinsic_difficulty"
    elif fault == "missing_axis":
        metadata.pop("quota_axis")
    elif fault == "quota":
        metadata["quota"]["test"]["hard"] = 0
    elif fault == "intrinsic_claim":
        metadata["review"]["source_sampling_tier_is_intrinsic_difficulty"] = True
    elif fault == "intrinsic_boolean":
        metadata["review"]["intrinsic_difficulty_counts"]["test"]["hard"] = False
    elif fault == "intrinsic_sum":
        metadata["review"]["intrinsic_difficulty_counts"]["test"]["easy"] = 499
    else:
        policy = root / "review_policy.json"
        obj = json.loads(policy.read_bytes())
        obj["quota_axis"] = "intrinsic_difficulty"
        policy.write_text(json.dumps(obj))
        metadata["policy_artifact_sha256"][policy.name] = verify.sha256(policy.read_bytes())
    path.write_text(json.dumps(metadata))
    seal_public_fixture(root, (inputs, path, root / "REVIEW_CONTRACT.md"))
    data = snapshot(evidence)
    with pytest.raises(ValueError, match="source_sampling_tier|intrinsic|quota"):
        verify.verify_records(data, **options(evidence))


def test_latency_quotas_use_bound_source_tiers_not_actual_difficulty(evidence):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    result = verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))
    profile = result["dataset_provenance"]["latency_profile"]
    assert profile["source_sampling_tier_counts"] == {"easy": 30, "medium": 40, "hard": 30}
    assert profile["intrinsic_difficulty_counts"] == {"easy": 100, "medium": 0, "hard": 0}
    # Keep all6/8/6 totals but move a tier onto a different immutable selected ID.
    first, second = data["latency_plan"]["items"][0], data["latency_plan"]["items"][6]
    first["sampling_tier"], second["sampling_tier"] = second["sampling_tier"], first["sampling_tier"]
    with pytest.raises(ValueError, match="frozen latency plan"):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


@pytest.mark.parametrize("field,value", [("domain", "foreign-domain"), ("difficulty", "hard"), ("sampling_tier", "hard")])
def test_latency_case_and_companions_cannot_change_bound_reviewed_source_metadata(evidence, field, value):
    evidence = full_evidence(evidence)
    data = latency_snapshot(evidence)
    for name in ("latency_metrics", "predictions", "metrics"):
        data[name][0][field] = value
    with pytest.raises(ValueError, match="Latency case reviewed-domain"):
        verify.verify_records(data, **options(evidence, "LATENCY", "test"), approval=approved(evidence))


def test_selector_context_limit_requires_error_and_defined_selection_traces(evidence):
    data = snapshot(evidence)
    for name in ("predictions", "metrics"):
        data[name][0].update(status="selector_context_limit", T_first_accepted=None)
        data[name][0].pop("winner_branch")
    refresh_traces(data)
    decision = data["decisions"][0]
    decision.pop("winner_branch")
    decision.pop("winner_position")
    assert verify.verify_records(data, **options(evidence))["records"] == 150
    data["failures"].pop()
    data["manifest"]["counts"]["failures"] -= 1
    with pytest.raises(ValueError, match="failure trace"):
        verify.verify_records(data, **options(evidence))


def test_approved_report_cannot_certify_empty_pilot_ledgers_with_true_manifest_claims(evidence):
    evidence = full_evidence(evidence)
    data = snapshot(evidence, "B13_GREEDY", "test")
    sentinel = approved(evidence)
    report_path = evidence[0] / "pilot_go.json"
    report = json.loads(report_path.read_bytes())
    entry = report["conditions"]["JFINAL"]
    path = Path(entry["verification"]["archive"])
    pilot_data = verify.read_archive(path)
    for name in verify.LEDGERS:
        pilot_data[name] = []
        pilot_data["manifest"]["counts"][name] = 0
    write_archive(path.parent, pilot_data)
    entry["verification"]["archive_sha256"] = verify.sha256(path.read_bytes())
    report_path.write_text(json.dumps(report))
    approval = json.loads(sentinel.read_bytes())
    approval["pilot_report_sha256"] = verify.sha256(report_path.read_bytes())
    sentinel.write_text(json.dumps(approval))
    bind_approval(data, sentinel)
    with pytest.raises(ValueError, match="coverage"):
        verify.verify_records(data, **options(evidence, "B13_GREEDY", "test"), approval=sentinel)
