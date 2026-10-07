"""Mandatory v3 validation, including when an old notebook passes light/off.

The implementations below reuse only version-neutral checks; orchestration and
prompt-role dispatch are v3-owned. No cached success can bypass a fresh check.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

from .. import preflight as legacy
from ..common import append_jsonl, config_hash, sha256_bytes, sha256_text, subprocess_env, write_json
from ..prompts import generator_prompt_ids

parser_selftest = legacy.parser_selftest
lock_check = legacy.lock_check
template_check = legacy.template_check
tokenizer_check = legacy.tokenizer_check
sampling_semantics_check = legacy.sampling_semantics_check
batching_audit = legacy.batching_audit


def _metadata(exp):
    hook = getattr(exp, "_phase_metadata", None)
    return dict(hook()) if hook else {
        "phase_epoch": getattr(exp, "phase_epoch", 0),
        "phase_id": getattr(exp, "phase_id", None),
        "gpu_uuid": getattr(exp, "gpu_uuid", None),
        "resident_roles": list(getattr(exp, "engines", {})),
        "cleanup_confirmed": getattr(exp, "cleanup_confirmed", False),
        "outside_T_total": True}


def _phase_dir(exp):
    meta = _metadata(exp)
    d = Path(exp.dir) / "preflight" / "phases" / (meta["phase_id"] or "pre_engines")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist(exp, name, result):
    result.update(_metadata(exp))
    if not hasattr(exp, "preflight_results"):
        exp.preflight_results = {}
    exp.preflight_results[name] = result
    d = _phase_dir(exp)
    write_json(d / f"{name}.json", result)
    append_jsonl(d / "detailed_checks.jsonl", {"check": name, "result": result})
    # Rebuild this phase's summary only, never mix a previous engine attempt.
    current = {k: v for k, v in exp.preflight_results.items()
               if v.get("phase_epoch") == result["phase_epoch"] and v.get("phase_id") == result["phase_id"]}
    write_json(d / "summary.json", current)
    hook = getattr(exp, "_persist_preflight", None)
    if hook:
        hook(name, result)
    else:
        write_json(Path(exp.dir) / "preflight" / "summary.json", exp.preflight_results)
        manifest = getattr(exp, "manifest", {})
        manifest["preflight"] = {k: {field: v.get(field) for field in
                                 ("ok", "blocking", "summary", "skipped", "phase_epoch", "phase_id",
                                  "gpu_uuid", "resident_roles", "cleanup_confirmed", "outside_T_total")}
                                 for k, v in exp.preflight_results.items()}
        exp.manifest = manifest
        write_json(Path(exp.dir) / "manifest.json", manifest)
    return result


def _run_check(exp, name, fn):
    try:
        # Legacy artifact writers are redirected to this attempt, without globals.
        result = fn()
    except Exception as exc:
        result = dict(getattr(exp, "preflight_results", {}).get(name, {}))
        if any(result.get(k) != _metadata(exp)[k] for k in ("phase_epoch", "phase_id")):
            result = {}
        result.update(ok=False, blocking=True, summary=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        _persist(exp, name, result)
        raise
    return _persist(exp, name, result)


def _view(exp):
    view = SimpleNamespace(**vars(exp))
    view.dir = _phase_dir(exp)
    (view.dir / "preflight").mkdir(exist_ok=True)
    # Preserve bound methods absent from vars(exp).
    view.dev_problems = exp.dev_problems
    view.roles = exp.roles
    return view


def _identity(exp, items):
    return {"config_hash": getattr(exp, "chash", config_hash(getattr(exp, "config", exp.cfg))),
            "model": exp.cfg["models"]["J"], "runtime": exp.cfg["jevk5_runtime"],
            "snapshot": str(Path(exp.paths["J"]).resolve()),
            "prompt_hash": config_hash(items)}


def _numeric(values, n):
    return (isinstance(values, (list, tuple)) and len(values) == n
            and all(type(x) in (int, float) and math.isfinite(x) for x in values))


def native_reference(exp):
    return _run_check(exp, "selector_native_reference", lambda: _native_reference(exp))


def _native_reference(exp):
    items = legacy.selector_items(exp)
    identity = _identity(exp, items)
    d = _phase_dir(exp)
    inp, out = d / "native_ref_in.json", d / "native_ref_out.json"
    execution_id = str(uuid.uuid4())
    if out.exists():
        out.unlink()
    write_json(inp, {"items": items, "identity": identity, "native_execution_id": execution_id, **_metadata(exp)})
    result = {"ok": False, "blocking": True, "summary": "native reference pending",
               "identity": identity, "reference_id": config_hash(identity),
               "native_execution_id": execution_id,
              "native_input": str(inp), "native_output": str(out)}
    _persist(exp, "selector_native_reference", result)
    cmd = [sys.executable, "-m", "jevlab.v3.native_ref", str(inp), str(out), exp.paths["J"]]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=subprocess_env())
    except subprocess.TimeoutExpired as exc:
        for name, text in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            (d / f"native_ref_{name}.txt").write_text(
                text.decode(errors="replace") if isinstance(text, bytes) else text or "", encoding="utf-8")
        if out.exists():
            result["native"] = json.loads(out.read_text(encoding="utf-8"))
        _persist(exp, "selector_native_reference", result)
        raise
    for name, text in (("stdout", process.stdout), ("stderr", process.stderr)):
        (d / f"native_ref_{name}.txt").write_text(text, encoding="utf-8")
    result.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    _persist(exp, "selector_native_reference", result)
    if out.exists():
        result["native"] = json.loads(out.read_text(encoding="utf-8"))
    reference = result.get("native", {})
    rows = reference.get("items", [])
    temperature = reference.get("temperature")
    ok = (process.returncode == 0 and reference.get("identity") == identity
          and reference.get("native_execution_id") == execution_id
          and reference.get("reference_id") == result["reference_id"]
          and reference.get("runtime_commit") == identity["runtime"]["commit"]
          and bool(reference.get("runtime_version"))
          and reference.get("snapshot") == identity["snapshot"]
          and isinstance(reference.get("cuda_hardware"), dict)
          and str(reference.get("model_device", "")).startswith("cuda")
          and reference.get("model_dtype") == "torch.bfloat16"
          and reference.get("cuda_hardware", {}).get("returncode") == 0
          and (not exp.gpu_uuid or exp.gpu_uuid in reference.get("cuda_hardware", {}).get("stdout", ""))
          and _numeric([temperature], 1)
          and abs(temperature - exp.cfg["selector"]["calibration_temperature_expected"]) <= 1e-9
          and len(rows) == len(items) and bool(items))
    for item, row in zip(items, rows):
        n = len(item["question"]["criteria"])
        ids = row.get("ids", [])
        ok &= (row.get("ids_equal_prompt_text") is True and bool(ids)
               and all(type(x) is int for x in ids)
               and row.get("ids_sha256") == sha256_text(json.dumps(ids))
               and _numeric(row.get("raw_logits"), n) and _numeric(row.get("probabilities"), n))
        probs = row.get("probabilities")
        if _numeric(probs, n):
            ok &= all(0 <= x <= 1 for x in probs) and abs(sum(probs) - 1) <= 1e-5
    result.update(ok=bool(ok), summary=f"native reference validated={bool(ok)}; rc={process.returncode}")
    return result


def selector_equivalence(exp, native):
    return _run_check(exp, "selector_equivalence", lambda: _selector_equivalence(exp, native))


def _selector_equivalence(exp, native):
    from ..prompts import selector_prompt_ids

    items = legacy.selector_items(exp)
    identity = _identity(exp, items)
    result = {"ok": False, "blocking": True, "summary": "selector equivalence pending", "rows": [],
              "identity": identity, "reference_id": config_hash(identity),
              "reference_phase": {k: (native or {}).get(k) for k in
                                  ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles")}}
    _persist(exp, "selector_equivalence", result)
    reference = (native or {}).get("native", {})
    result["native_execution_id"] = (native or {}).get("native_execution_id")
    temperature = reference.get("temperature")
    if (not native or native.get("ok") is not True or native.get("skipped")
            or native.get("identity") != identity or reference.get("identity") != identity
            or native.get("reference_id") != result["reference_id"]
            or reference.get("reference_id") != result["reference_id"]
            or reference.get("runtime_commit") != identity["runtime"]["commit"]
            or not isinstance(result["native_execution_id"], str) or not result["native_execution_id"]
            or reference.get("native_execution_id") != result["native_execution_id"]
            or not _numeric([temperature], 1)
            or abs(temperature - exp.cfg["selector"]["calibration_temperature_expected"]) > 1e-9
            or len(reference.get("items", [])) != len(items) or not items):
        result["summary"] = "missing, stale, or invalid native reference identity/calibration"
        return result
    selector = exp.ctx.selector
    write_json(_phase_dir(exp) / "native_reference_trace.json",
               {**_metadata(exp), "reference_phase": result["reference_phase"],
                 "identity": identity, "reference_id": result["reference_id"],
                 "native_execution_id": result["native_execution_id"],
                "input": {"items": items, "identity": identity}, "output": reference})
    for index, (item, nat) in enumerate(zip(items, reference["items"])):
        ids, text = selector_prompt_ids(exp.toks["J"], item["state"], item["question"])
        n = len(item["question"]["criteria"])
        logits, dt = selector.letter_logits_from_ids(ids, n)
        # Never substitute booleans indicating finiteness for the actual values.
        probs = selector.probabilities(logits) if _numeric(logits, n) else None
        nl, npb = nat.get("raw_logits"), nat.get("probabilities")
        row = {"index": index, "ids": ids, "n_ids": len(ids),
               "ids_sha256": sha256_text(json.dumps(ids)), "prompt_sha256": sha256_text(text),
               "native_ids": nat.get("ids"), "native_ids_sha256": nat.get("ids_sha256"),
               "vllm_logits": logits, "native_logits": nl, "vllm_probs": probs, "native_probs": npb,
               "vllm_seconds": dt, "native_seconds": nat.get("seconds"),
                "reference_id": result["reference_id"], **_metadata(exp)}
        row["native_execution_id"] = result["native_execution_id"]
        row["same_ids"] = (ids == nat.get("ids") and row["ids_sha256"] == nat.get("ids_sha256")
                           and nat.get("ids_equal_prompt_text") is True)
        valid = (all(_numeric(v, n) for v in (logits, probs, nl, npb))
                 and all(0 <= x <= 1 for v in (probs, npb) for x in v)
                 and all(abs(sum(v) - 1) <= 1e-5 for v in (probs, npb)))
        row["ok"] = False
        if valid:
            av, an = probs.index(max(probs)), npb.index(max(npb))
            ordered = sorted(nl, reverse=True)
            margin = ordered[0] - ordered[1] if n > 1 else 99
            ulp = 2.0 ** (math.floor(math.log2(max(abs(ordered[0]), 1e-6))) - 7)
            tie = margin <= 2 * ulp and max(nl) - nl[av] <= 2 * ulp
            row.update(ok=row["same_ids"] and (av == an or tie), argmax_vllm=av, argmax_native=an,
                       native_top2_margin=margin, bf16_ulp=ulp, tie_tolerated=tie,
                       max_abs_prob_diff=max(abs(a - b) for a, b in zip(probs, npb)))
        else:
            row["error"] = "invalid non-bool numeric logits/probabilities"
        result["rows"].append(row)
        result["summary"] = f"selector equivalence: {len(result['rows'])}/{len(items)} rows completed"
        _persist(exp, "selector_equivalence", result)
    result.update(ok=all(r["ok"] for r in result["rows"]),
                  summary=f"selector equivalence: {len(result['rows'])} decisions; "
                          f"failures={sum(not r['ok'] for r in result['rows'])}")
    return result


def continuity_test(exp):
    result = legacy.continuity_test(exp)
    result["blocking"] = True
    return result


def trace_audit(exp):
    """Require current-attempt detailed evidence, not files from an earlier phase."""
    d, meta = _phase_dir(exp), _metadata(exp)
    log_path = d / "detailed_checks.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
    logs_ok = bool(records) and all(
        all(record["result"].get(k) == meta[k] for k in
            ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles"))
        for record in records)
    engine_log = Path(exp.dir) / "logs" / "vllm.log"
    start = getattr(exp, "_phase_log_start", None)
    raw = b""
    if type(start) is int and start >= 0 and engine_log.is_file():
        with engine_log.open("rb") as stream:
            stream.seek(start)
            raw = stream.read()
    engine_logs_ok = bool(raw.strip())
    if engine_logs_ok:
        (d / "vllm_phase.log").write_bytes(raw)
    trace_ok, trace = True, None
    if exp.ctx.selector is not None:
        path = d / "native_reference_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        trace_ok = bool(trace) and all(trace.get(k) == meta[k] for k in
                                      ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles"))
        equivalence = exp.preflight_results.get("selector_equivalence", {})
        trace_ok = (trace_ok and trace["reference_id"] == equivalence.get("reference_id")
                    and trace["output"].get("reference_id") == trace["reference_id"]
                    and trace["identity"] == equivalence.get("identity")
                    and trace["output"].get("identity") == trace["identity"]
                    and bool(equivalence.get("rows"))
                    and all(all(row.get(k) == meta[k] for k in
                                ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles"))
                            for row in equivalence["rows"]))
        if trace_ok:
            native_items = trace["output"].get("items", [])
            rows = equivalence["rows"]
            trace_ok = (trace.get("native_execution_id") == equivalence.get("native_execution_id")
                        == trace["output"].get("native_execution_id")
                        and len(native_items) == len(rows))
            for index, (item, row) in enumerate(zip(native_items, rows)):
                values = row.get("native_logits")
                n = len(values) if isinstance(values, (list, tuple)) else 0
                trace_ok &= (item.get("index") == row.get("index") == index
                             and item.get("reference_id") == row.get("reference_id") == trace["reference_id"]
                             and row.get("native_execution_id") == trace.get("native_execution_id")
                             and item.get("ids") == row.get("native_ids") == row.get("ids")
                             and item.get("ids_sha256") == row.get("native_ids_sha256") == row.get("ids_sha256")
                             and n > 0 and _numeric(item.get("raw_logits"), n)
                             and _numeric(item.get("probabilities"), n)
                             and item["raw_logits"] == row["native_logits"]
                             and item["probabilities"] == row["native_probs"]
                             and all(0 <= x <= 1 for x in item["probabilities"])
                             and abs(sum(item["probabilities"]) - 1) <= 1e-5)
    return {"ok": bool(logs_ok and engine_logs_ok and trace_ok), "blocking": True,
            "summary": f"current-phase detailed logs={logs_ok}; engine logs={engine_logs_ok}; native JSON trace={bool(trace_ok)}",
            "detailed_log": str(log_path), "n_records": len(records),
            "engine_log": {"path": str(engine_log), "start_byte": start,
                           "end_byte": start + len(raw) if start is not None else None,
                           "sha256": sha256_bytes(raw), "current_phase_only": True},
            "native_trace": str(d / "native_reference_trace.json") if trace is not None else None}


def prompt_length_check(exp):
    # Validate every frozen input, not merely N_PROBLEMS of a partial job.
    from .runner import _input_rows

    if exp.p["SPLIT"] == "dev":
        items = exp.dev_problems()
    elif exp.cond == "LATENCY":
        items = exp.latency_items()
    else:
        split = exp.p["SPLIT"]
        items = _input_rows(exp.data_dir / f"{split}_inputs.jsonl", 500 if split == "test" else 50)
    roles = [r for r in exp.toks if r != "J"]
    lengths = {f"{role}:{it.get('id', it.get('item_id'))}":
               len(generator_prompt_ids(exp.toks[role], exp.prompts["generador"], it["problem"])[0])
               for role in roles for it in items}
    over = {k: n for k, n in lengths.items() if n > exp.p["MAX_PROMPT_TOKENS"]}
    return {"ok": bool(lengths) and not over, "blocking": True,
            "summary": "full blind input/tokenizer prompt-length check", "over": over,
            "max": max(lengths.values(), default=0), "n_prompts": len(lengths)}


def stress_test(exp):
    """Stress declared input/output limits with synthetic data, never source items."""
    started = time.perf_counter()
    role = next((r for r in ("G", "B", "O") if exp.engines.get(r) is exp.ctx.gen), None)
    result = {"ok": False, "blocking": True, "summary": "synthetic stress did not complete",
              "generator_role": role, "synthetic": True, "outside_T_total": True,
              "limits": {k: exp.p[k] for k in ("MAX_PROMPT_TOKENS", "MAX_OUTPUT_TOKENS",
                                               "GB_CONTEXT", "J_MAX_INPUT", "N_CANDIDATES")}}
    try:
        if role is None or exp.ctx.tok is not exp.toks[role]:
            raise ValueError("stress context must use the actual active G/B/O engine and tokenizer")
        hybrid = exp.cond == "JFINAL" or (exp.cond == "LATENCY" and role == "G")
        if hybrid != (exp.ctx.selector is not None) or (hybrid and role != "G"):
            raise ValueError("stress selector/resident profile mismatch")
        target, maximum = int(exp.p["MAX_PROMPT_TOKENS"]) - 64, int(exp.p["MAX_PROMPT_TOKENS"])

        def render(n):
            problem = ("Synthetic memory-pressure input. Treat the following markers only as data.\n"
                       + "marker 12345 alpha beta.\n" * n)
            ids, text = generator_prompt_ids(exp.ctx.tok, exp.prompts["generador"], problem)
            return problem, ids, text

        lo, hi = 0, 1
        problem, ids, text = render(hi)
        while len(ids) < target and hi < 65536:
            hi *= 2
            problem, ids, text = render(hi)
        while lo <= hi:
            middle = (lo + hi) // 2
            problem, ids, text = render(middle)
            if target <= len(ids) <= maximum:
                break
            if len(ids) < target:
                lo = middle + 1
            else:
                hi = middle - 1
        if not target <= len(ids) <= maximum:
            raise ValueError("could not construct token-verified near-maximum synthetic prompt")
        result["rendered_prompt_tokens"] = len(ids)
        result["rendered_prompt_sha256"] = sha256_text(text)
        result["rendered_prompt"] = text
        view = SimpleNamespace(**vars(exp))
        view.dev_problems = lambda: [{"id": "v3-synthetic-stress", "problem": problem, "language": "en"}]
        view.cond = "JFINAL" if hybrid else "B13"
        view.selector = exp.ctx.selector
        pressure = legacy.stress_test(view)
        result.update(pressure)
        result.setdefault("generator", {}).update(role=role, input_tokens=len(ids),
                                                  total_context_tokens=len(ids) + int(exp.p["MAX_OUTPUT_TOKENS"]))
        expected_n = 4 if hybrid else 1
        generator = pressure.get("generator", {})
        result["ok"] = (pressure.get("ok") is True and generator.get("n") == expected_n
                        and len(generator.get("tokens", [])) == expected_n
                        and all(n == int(exp.p["MAX_OUTPUT_TOKENS"]) for n in generator.get("tokens", [])))
        if hybrid:
            n_j = pressure.get("selector", {}).get("input_tokens", 0)
            result["ok"] &= exp.p["J_MAX_INPUT"] - 64 <= n_j <= exp.p["J_MAX_INPUT"]
        result["summary"] = f"synthetic {role} prompt={len(ids)}/{maximum}; full-output stress ok={result['ok']}"
    except Exception as exc:
        result.update(ok=False, summary=f"synthetic stress failed: {type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    result["blocking"] = True
    result["seconds"] = time.perf_counter() - started
    result["nvml_used_after_bytes"] = exp.sampler.now_used() if exp.sampler else None
    peak = exp.sampler.window_peak(started, time.perf_counter()) if exp.sampler else None
    result["nvml_peak_gib"] = peak / 2**30 if peak is not None else None
    result["memory_note"] = "NVML periodic observations, not an exact allocator peak"
    _persist(exp, "stress", result)
    return result


def eager_vs_graph(exp):
    """Legacy 3-to-20 diagnostic, preserving exact v3 resident engine arguments.

    The legacy callable hard-codes max_num_seqs=4, so it cannot restore v3 single
    controls correctly. This owned adapter uses the same algorithm, not globals.
    """
    from ..engines import Req, sampling_params
    from .engines import SafeEngine
    from ..runner import shutdown_engine_checked

    started = time.perf_counter()
    result = {"ok": False, "blocking": True, "threshold": 0.75, "outside_T_total": True,
              "diagnostic_load_seconds": []}
    cleanup, contexts = [], []
    try:
        role = next((r for r in ("G", "B", "O") if exp.engines.get(r) is exp.ctx.gen), None)
        if role in ("B", "O"):
            result.update(ok=True, blocking=False, skipped=True, generator_role=role,
                          summary=f"skipped for {role} (memory)")
            return result
        if role != "G" or exp.ctx.tok is not exp.toks["G"]:
            raise ValueError("graph diagnostic requires the actual active G context")
        graph = exp.engines["G"]
        original = dict(graph.engine_kwargs)
        result["original_engine_kwargs"] = original
        if original.get("enforce_eager"):
            result.update(ok=True, blocking=False, skipped=True, generator_role="G",
                          summary="eager profile: no resident graph engine to compare")
            return result
        dev = exp.dev_problems()
        if len(dev) != 20:
            raise ValueError("graph diagnostic requires exactly 20 copied development prompts")
        for owner in (exp, getattr(exp, "exp", exp)):
            for name in ("ctx", "ctx_H"):
                ctx = getattr(owner, name, None)
                if ctx is not None and ctx.gen is graph and all(ctx is not c for c in contexts):
                    contexts.append(ctx)

        def tokens(engine, item):
            ids, _ = generator_prompt_ids(exp.toks["G"], exp.prompts["generador"], item["problem"])
            params = sampling_params(temperature=0.0, top_p=1.0, top_k=0, max_tokens=64,
                                     seed=None, stop_token_ids=[], ignore_eos=True)
            rid = engine.new_id("eg")
            return engine.run([Req(rid, ids, params, watch_final=False)])[rid].token_ids

        def load(eager):
            engine = SafeEngine("G_eager" if eager else "G", exp.paths["G"], exp.toks["G"],
                            max_model_len=original["max_model_len"], max_num_seqs=original["max_num_seqs"],
                            kv_cache_gb=None, gpu_memory_utilization=original["gpu_memory_utilization"],
                             enforce_eager=eager, eos_ids=exp.eos["G"], newline_ids=exp.nl["G"],
                             failure_callback=getattr(exp, "_engine_failure", None),
                             extra={k: v for k, v in original.items()
                                   if k not in ("model", "tokenizer", "enforce_eager")})
            result["diagnostic_load_seconds"].append(getattr(engine, "load_seconds", None))
            return engine

        def close(engine):
            report = shutdown_engine_checked(engine, float(exp.p["ENGINE_CLOSE_TIMEOUT_S"]))
            cleanup.append(report)
            if report.get("cleanup_confirmed") is not True:
                raise RuntimeError("graph diagnostic engine cleanup was not confirmed")

        a = [tokens(graph, item) for item in dev[:3]]
        close(graph)
        del exp.engines["G"]
        for ctx in contexts:
            ctx.gen = None
        eager = None
        try:
            eager = load(True)
            exp.engines["G"] = eager
            b = [tokens(eager, item) for item in dev[:3]]
            initial = sum(sum(x == y for x, y in zip(ta, tb)) for ta, tb in zip(a, b)) / (3 * 64)
            expanded = initial < 0.75
            if expanded:
                b.extend(tokens(eager, item) for item in dev[3:])
        finally:
            if eager is not None:
                close(eager)
                del exp.engines["G"]
                restored = load(False)
                exp.engines["G"] = restored
                for ctx in contexts:
                    ctx.gen = restored
                if restored.engine_kwargs != original:
                    raise RuntimeError("restored graph engine arguments differ from the original")
        if expanded:
            a.extend(tokens(exp.engines["G"], item) for item in dev[3:])
        rows = [{"problem_id": item["id"], "graph_tokens": ta, "eager_tokens": tb,
                 "agree": sum(x == y for x, y in zip(ta, tb)), "n": 64,
                 "complete": len(ta) == len(tb) == 64} for item, ta, tb in zip(dev, a, b)]
        agreement = sum(r["agree"] for r in rows) / (64 * len(rows))
        result.update(ok=agreement >= 0.75 and all(r["complete"] for r in rows), blocking=False,
                      summary=f"graph/eager greedy agreement {agreement:.3f}; {len(rows)} dev prompts",
                      agreement=agreement, initial_agreement=initial, expanded_to_20=expanded, rows=rows,
                      restored_engine_kwargs=dict(exp.engines["G"].engine_kwargs))
    except Exception as exc:
        result.update(ok=False, blocking=True, summary=f"graph diagnostic lifecycle failure: {type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        result["process_cleanup"] = cleanup
        result["seconds"] = time.perf_counter() - started
        _persist(exp, "eager_vs_graph", result)
    return result


def cache_check(exp):
    rows = {r: e.engine_kwargs.get("enable_prefix_caching") is False for r, e in exp.engines.items()}
    return {"ok": bool(rows) and all(rows.values()), "blocking": True,
            "summary": "prefix caching disabled on every resident engine", "roles": rows}


def pre_engine_checks(exp, mode="full"):
    if mode not in ("full", "light", "off"):
        raise ValueError("unknown PREFLIGHT_MODE")
    res = {}
    for name, fn in (("parser_selftest", parser_selftest), ("lock", lambda: lock_check(exp)),
                     ("chat_template", lambda: template_check(_view(exp))),
                     ("tokenizer", lambda: tokenizer_check(exp)),
                     ("prompt_length", lambda: prompt_length_check(exp)),
                     ("sampling_semantics", sampling_semantics_check)):
        res[name] = _run_check(exp, name, fn)
        if res[name].get("blocking", True) and (res[name].get("ok") is not True or res[name].get("skipped")):
            return res
    if "J" in exp.roles():
        res["selector_native_reference"] = _run_check(exp, "selector_native_reference", lambda: native_reference(exp))
    return res


def post_engine_checks(exp, mode="full"):
    if mode not in ("full", "light", "off"):
        raise ValueError("unknown PREFLIGHT_MODE")
    if exp.ctx is None:
        raise RuntimeError("v3 postflight requires an active engine context")
    checks = [("eager_vs_graph", lambda: eager_vs_graph(exp)), ("cache", lambda: cache_check(exp))]
    if exp.ctx.selector is not None:
        checks.extend((("selector_equivalence", lambda: selector_equivalence(
            exp, exp.preflight_results.get("selector_native_reference"))),
                       ("batching_audit", lambda: batching_audit(exp))))
    checks.extend((("continuity", lambda: continuity_test(exp)), ("stress", lambda: stress_test(exp)),
                   ("detailed_trace_audit", lambda: trace_audit(exp))))
    res = {}
    for name, fn in checks:
        res[name] = _run_check(exp, name, fn)
        # The runner calls _check_blocking after return; do not execute later checks.
        if res[name].get("blocking", True) and (res[name].get("ok") is not True or res[name].get("skipped")):
            break
    return res
