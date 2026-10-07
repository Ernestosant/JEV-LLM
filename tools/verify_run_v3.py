"""Strict offline v3 execution audit. Never opens gold or grades answers."""

import argparse
import ast
import hashlib
import io
import json
import math
import random
import re
import stat
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("G_SINGLE", "JFINAL", "B13", "G_GREEDY", "B13_GREEDY", "Q9_GREEDY")
SAMPLED = {"G_SINGLE", "JFINAL", "B13"}
SEEDS = (17, 29, 43)
QUOTA_AXIS = "source_sampling_tier"
REVIEW_POLICY_ID = "jev-v3-source-sampling-tier-agent-review-20261004"
LEVELS = ("easy", "medium", "hard")
SAMPLING_QUOTAS = {"test": {"easy": 30, "medium": 40, "hard": 30},
                   "pilot": {"easy": 3, "medium": 4, "hard": 3},
                   "latency": {"easy": 6, "medium": 8, "hard": 6}}
LEDGERS = ("predictions", "metrics", "proposals", "decisions", "failures", "latency_metrics",
           "candidates", "rounds", "memory", "warmup", "swaps")
BLOCKING = {"parser_selftest", "lock", "chat_template", "prompt_length", "sampling_semantics",
            "tokenizer", "stress", "continuity", "cache"}
STATUSES = {"final", "truncated", "eos_invalid", "timeout", "infraerror", "infrastructure",
             "exception", "engine_error", "selector_error", "selector_context_limit", "partial", "no_candidates", "context_limit"}
POOL_STATUSES = {"complete", "partial", "no_candidates"}
ERROR_STATUSES = {"timeout", "infraerror", "infrastructure", "exception", "engine_error", "selector_error",
                  "selector_context_limit"}
MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_MEMBER_BYTES = 256 * 1024**2
MAX_READ_BYTES = 512 * 1024**2


def check_operational_transition(old, new, approval, manifest, config_sha, report_sha):
    """Only explicit consent permits the exact full pilot-to-execution inventory delta."""
    if "new_verifier_sha256" in approval:
        require(approval["new_verifier_sha256"] == sha256(safe_read(__file__)),
                "Approved current verifier SHA mismatch")
    if "operational_amendment" not in approval:
        require(old == new, "Pilot code inventory mismatch without operational approval")
        require("operational_amendment" not in manifest.get("authorization", {}),
                "Unapproved manifest operational amendment")
        return None
    policy = approval["operational_amendment"]
    fields = {"id", "approved", "approved_by", "approved_utc", "config_sha256", "pilot_report_sha256",
              "old_code_sha256", "new_code_sha256", "latency_source_plan_sha256", "latency_schedule_sha256",
              "calendar_sha256", "cohort_a", "cohort_c", "max_job_seconds", "max_gpu_seconds",
              "expected_test_rows", "expected_latency_rows", "seeds", "repetitions", "load_counts",
              "authorization_source"}
    require(isinstance(policy, dict) and set(policy) == fields,
            "Missing/private operational policy fields or null amendment")
    require(all(type(policy[k]) is int for k in ("max_job_seconds", "max_gpu_seconds", "expected_test_rows",
                                                "expected_latency_rows", "repetitions"))
            and isinstance(policy["seeds"], list) and all(type(v) is int for v in policy["seeds"])
            and isinstance(policy["load_counts"], dict)
            and all(type(v) is int for v in policy["load_counts"].values()), "Invalid operational policy value types")
    # Import only the pure helper, never runner/engines, including standalone CLI use.
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from jevlab.v3.operational import validate_code_transition

    validate_code_transition(old, new, approval, config_sha, report_sha)
    at = datetime.fromisoformat(policy["approved_utc"].replace("Z", "+00:00"))
    require(at.utcoffset() == timedelta(0) and at.timestamp() <= timestamp(manifest["created_utc"]),
            "Operational approval must be UTC and precede construction")
    require(manifest.get("authorization", {}).get("operational_amendment") == policy,
            "Manifest operational approval mismatch")
    package = ROOT / "src" / "jevlab"
    inventory = {p.relative_to(package).as_posix(): sha256(safe_read(p))
                 for p in sorted(package.glob("*.py")) + sorted((package / "v3").glob("*.py"))}
    require(new == inventory, "Operational executed source inventory mismatch")
    return policy


def check_operational_calendar(data, subset, frozen_plan, source_sha, schedule_raw, policy):
    """Bind the approved deterministic calendar to all actual executions, not ETA."""
    from jevlab.v3.operational import build_calendar, validate_calendar

    plan, manifest = data["latency_plan"], data["manifest"]
    order = json.loads(schedule_raw)["order"]
    metadata = {r["item_id"]: r for r in frozen_plan["items"]}
    calendar = build_calendar(subset, order, metadata)
    validate_calendar(plan.get("calendar"), subset, order, metadata, policy)
    require(policy["latency_source_plan_sha256"] == source_sha == plan.get("source_plan_sha256")
            and policy["latency_schedule_sha256"] == sha256(schedule_raw),
            "Operational source plan/schedule SHA mismatch")
    operational = {"operational_amendment": policy, "operational_policy_sha256": canonical_hash(policy),
                   "calendar_sha256": canonical_hash(calendar), "calendar": calendar,
                   "residency_phases": ["B", "H", "B"]}
    require(all(plan.get(k) == v for k, v in operational.items())
            and manifest.get("operational_metadata") == operational,
            "Archived/manifest operational metadata mismatch")
    fields = {"items", "quota_axis", "source_plan_sha256", "intrinsic_difficulty_counts", "blocks", "seeds",
              "repetitions", "conditions", "generation", "split", *operational}
    selected = [pid for pid in order if pid in metadata]
    require(set(plan) == fields and plan["items"] == frozen_plan["items"]
            and plan["blocks"] == [selected[i:i + 10] for i in range(0, 100, 10)]
            and plan["seeds"] == list(SEEDS) and type(plan["repetitions"]) is int and plan["repetitions"] == 3
            and plan["conditions"] == ["JFINAL", "B13_GREEDY"]
            and type(plan["generation"]) is int and plan["generation"] == 0 and plan["split"] == "test"
            and plan["quota_axis"] == QUOTA_AXIS
            and plan["intrinsic_difficulty_counts"] == frozen_plan["intrinsic_difficulty_counts"],
            "Operational execution plan changed frozen metadata/blocks or contains private fields")
    swaps = data.get("swaps", [])
    require(len(swaps) == 3 and [s.get("to") for s in swaps] == ["B", "H", "B"]
            and all(s.get("ok") is True for s in swaps), "Operational requires exactly B2/H1 successful loads")
    for epoch, swap in enumerate(swaps, 1):
        warmup = [r for r in data.get("warmup", []) if r.get("phase_epoch") == epoch]
        require(len(warmup) >= manifest["config"]["params"].get("WARMUP_N", 0),
                "Missing operational phase warmup")
        require(all(r.get("outside_T_total") is True and type(r.get("generation")) is int
                    and r["generation"] >= 900 and all(r.get(k) == swap.get(k) for k in
                    ("phase_id", "phase_epoch", "gpu_uuid", "resident_roles", "cleanup_confirmed"))
                    and all(type(record.get(k)) in (int, float) and math.isfinite(record[k])
                            for record in (r, swap) for k in ("t_start_perf", "t_end_perf"))
                    and swap["t_start_perf"] <= r["t_start_perf"] < r["t_end_perf"] <= swap["t_end_perf"]
                    for r in warmup),
                "Operational warmup inside case clock or outside successful residency")
    require(all(type(r.get("phase_epoch")) is int and 1 <= r["phase_epoch"] <= 3
                for r in data.get("warmup", [])), "Foreign operational warmup phase")
    for name in ("latency_metrics", "predictions", "metrics"):
        rows = data[name]
        require(len(rows) == len(calendar), "Operational actual calendar coverage mismatch")
        previous_end = None
        for index, (row, descriptor) in enumerate(zip(rows, calendar), 1):
            require(all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in descriptor.items())
                    and type(row.get("case_index")) is int and row["case_index"] == index
                    and type(row.get("phase_epoch")) is int and row["phase_epoch"] == descriptor["residency"] + 1,
                    "Operational actual order/cohort/block/residency mismatch: " + name)
            start, end = timestamp(row["t_start_utc"]), timestamp(row["t_end_utc"])
            require(start < end and (previous_end is None or previous_end <= start),
                    "Operational actual clocks do not follow calendar")
            previous_end = end
    pairs = {}
    for row in data["latency_metrics"]:
        pairs.setdefault((row["item_id"], row["seed"], row["repetition"]), []).append(row)
    require(len(pairs) == 900 and all(len(pair) == 2 for pair in pairs.values())
            and Counter(pair[0]["condition"] for pair in pairs.values()) == {"B13_GREEDY": 450, "JFINAL": 450},
            "Operational actual paired direction balance mismatch")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), default=str).encode())


def jsonl(text):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "Ledger rows must be objects")
    return rows


def safe_read(path):
    path = Path(path)
    require("gold" not in path.name.lower(), "Gold access forbidden")
    return path.read_bytes()


def read_archive(path):
    """Read a single immutable snapshot; unknown members are never opened."""
    path = Path(path)
    require("gold" not in path.name.lower(), "Gold access forbidden")
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, "ZIP snapshot exceeds byte limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARCHIVE_BYTES + 1)
    require(len(raw) <= MAX_ARCHIVE_BYTES, "ZIP snapshot exceeds byte limit")
    try:
        snapshot = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        raise ValueError("ZIP integrity failure") from error
    with snapshot as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) <= 10000 and len(names) == len({n.casefold().rstrip("/") for n in names}),
                "Duplicate/canonical ZIP members or excessive member count")
        require(all(n and not PurePosixPath(n).is_absolute() and "\\" not in n and ":" not in n
                    and all(p not in {"", ".", ".."} and p == p.rstrip(" .") for p in n.rstrip("/").split("/"))
                    and not any(ord(c) < 32 or c in '<>"|?*' for c in n)
                    and not any(p.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                *[f"LPT{i}" for i in range(1, 10)]} for p in n.rstrip("/").split("/"))
                    for n in names), "Unsafe ZIP member")
        require(all(not stat.S_ISLNK(entry.external_attr >> 16) for entry in entries), "ZIP symlink forbidden")
        require(not any("gold" in n.lower() for n in names), "Gold forbidden in archive")
        manifests = [n for n in names if n == "manifest.json" or n.endswith("/manifest.json")]
        require(len(manifests) == 1, "Expected exactly one manifest.json")
        prefix = manifests[0][:-len("manifest.json")]
        consumed = 0

        def member(name, optional=False, binary=False):
            nonlocal consumed
            require(isinstance(name, str) and name and not PurePosixPath(name).is_absolute()
                    and all(p not in {"", ".", ".."} for p in name.split("/")) and "\\" not in name,
                    "Unsafe artifact reference")
            full = prefix + name
            if optional and full not in names:
                return None
            require(full in names, f"Expected exactly one {name}")
            entry = archive.getinfo(full)
            require(not entry.is_dir() and entry.file_size <= MAX_MEMBER_BYTES
                    and consumed + entry.file_size <= MAX_READ_BYTES, "ZIP decompression byte limit")
            try:
                with archive.open(entry) as stream:
                    content = stream.read(min(MAX_MEMBER_BYTES, MAX_READ_BYTES - consumed) + 1)
                consumed += len(content)
                require(len(content) == entry.file_size and consumed <= MAX_READ_BYTES
                        and len(content) <= MAX_MEMBER_BYTES, "ZIP decompression byte limit")
                return content if binary else content.decode("utf-8")
            except (zipfile.BadZipFile, RuntimeError, EOFError) as error:
                raise ValueError("ZIP integrity failure: " + name) from error

        out = {"manifest": json.loads(member("manifest.json")),
               "progress": json.loads(member("progress.json")), "archive_sha256": sha256(raw)}
        for name in LEDGERS:
            text = member(name + ".jsonl", optional=True)
            out[name] = jsonl(text) if text is not None else []
        plan = member("latency_plan.json", optional=True)
        out["latency_plan"] = json.loads(plan) if plan is not None else None
        summary_text = member("preflight/summary.json", optional=True)
        summary = json.loads(summary_text) if summary_text is not None else {}
        references = set()
        for name, result in summary.items():
            folder = phase_folder(result)
            references.add(folder + "/" + name + ".json")
            if name == "selector_native_reference":
                references.update(folder + "/" + n for n in ("native_ref_in.json", "native_ref_out.json"))
            if result.get("phase_id"):
                references.update(folder + "/" + n for n in ("summary.json", "detailed_checks.jsonl", "native_reference_trace.json", "vllm_phase.log"))
        for swap in out["swaps"]:
            folder = phase_folder(swap)
            references.update(folder + "/" + n for n in ("summary.json", "detailed_checks.jsonl", "native_reference_trace.json", "vllm_phase.log"))
            references.update(folder + "/" + name + ".json" for name in swap.get("postflight", {}))
        out["artifacts"] = {}
        if summary_text is not None:
            out["artifacts"]["preflight/summary.json"] = summary
        for name in sorted(references):
            text = member(name, optional=True, binary=name.endswith(".log"))
            if text is not None:
                out["artifacts"][name] = text if name.endswith(".log") else jsonl(text) if name.endswith(".jsonl") else json.loads(text)
        return out


def timestamp(value):
    require(isinstance(value, str) and value.strip(), "Timestamp must be a nonempty string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Timestamp must have timezone")
    return parsed.timestamp()


def phase_folder(result):
    phase = result.get("phase_id")
    require(phase is None or (isinstance(phase, str) and re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", phase)), "Invalid phase UUID")
    return "preflight/phases/" + (phase or "pre_engines")


def check_public_provenance(metadata, seals, directory):
    allowed = {"dataset_version", "sealed", "status", "construction_seed", "latency_seed", "actual_n", "requested_n",
               "quota", "input_sha256", "builder_sha256", "review", "analysis_manifest_sha256", "policy_artifact_sha256",
               "source_revisions", "test_reserved_before_pilot", "latency_test_subset_only", "max_prompt_tokens", "max_abs_answer", "quota_axis"}
    require(set(metadata) <= allowed, "Private/unapproved fields in public dataset manifest")
    require(metadata.get("status") == "FROZEN" and metadata.get("construction_seed") == 20261002
            and metadata.get("latency_seed") == 20261003 and metadata.get("test_reserved_before_pilot") is True
            and metadata.get("latency_test_subset_only") is True, "Missing final public dataset policy commitments")
    review = metadata.get("review", {})
    require(metadata.get("quota_axis") == QUOTA_AXIS and metadata.get("quota") == SAMPLING_QUOTAS
            and review.get("source_sampling_tier_is_intrinsic_difficulty") is False,
            "Requires source_sampling_tier quota axis, not intrinsic-difficulty quotas")
    intrinsic = review.get("intrinsic_difficulty_counts", {})
    require(set(intrinsic) == {"test", "pilot"} and all(isinstance(intrinsic[s], dict)
            and set(intrinsic[s]) == set(LEVELS) and all(type(n) is int and n >= 0 for n in intrinsic[s].values())
            and sum(intrinsic[s].values()) == size for s, size in (("test", 500), ("pilot", 50))),
            "Missing actual intrinsic difficulty counts for550 agent-reviewed problems")
    require(isinstance(metadata.get("builder_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", metadata["builder_sha256"])
            and isinstance(metadata.get("analysis_manifest_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", metadata["analysis_manifest_sha256"])
            and seals.get("review_manifest.json") == metadata["analysis_manifest_sha256"],
            "Missing final review/source builder commitment (private review is not opened)")
    sources = metadata.get("source_revisions", {})
    require(isinstance(sources, dict) and sources and all(isinstance(v, dict) and set(v) <= {"repo", "revision", "license"}
            and all(isinstance(v.get(k), str) and v[k].strip() for k in ("repo", "revision")) for v in sources.values()),
            "Missing/nonpublic pinned source provenance")
    policies = metadata.get("policy_artifact_sha256", {})
    require(set(policies) == {"review_policy.json", "ELIGIBILITY_POLICY.md"}, "Missing public adjudication policy artifacts")
    for name, digest in policies.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                and seals.get(name) == digest == sha256(safe_read(Path(directory) / name)), "Public policy artifact SHA mismatch: " + name)
    policy = json.loads(safe_read(Path(directory) / "review_policy.json"))
    require(review.get("policy_id") == policy.get("policy_id") == REVIEW_POLICY_ID
            and policy.get("contract_sha256") == review.get("policy_sha256")
            and policy.get("quota_axis") == QUOTA_AXIS and policy.get("quota") == SAMPLING_QUOTAS,
            "Source-tier review policy identity/contract/quota binding mismatch")
    return {"analysis_manifest_sha256": metadata["analysis_manifest_sha256"], "source_revisions": sources,
            "builder_sha256": metadata["builder_sha256"], "policy_artifact_sha256": policies,
            "quota_axis": QUOTA_AXIS, "source_sampling_quota_per_domain": SAMPLING_QUOTAS,
            "intrinsic_difficulty_counts": intrinsic, "n_agent_reviewed": 550, "human_reviewed": False,
            "source_sampling_tier_is_intrinsic_difficulty": False}


def sampling_profile(rows, split):
    """Selection balances immutable source tiers; intrinsic labels are descriptive."""
    quota = SAMPLING_QUOTAS[split]
    require(len(rows) == 5 * sum(quota.values()) and all(isinstance(r.get("domain"), str) and r["domain"].strip()
            and r.get("sampling_tier") in LEVELS and r.get("difficulty") in LEVELS for r in rows),
            "Missing/invalid reviewed domain, source sampling tier or actual intrinsic difficulty")
    domains = sorted({r["domain"] for r in rows})
    require(len(domains) == 5 and all(Counter(r["sampling_tier"] for r in rows if r["domain"] == d) == quota for d in domains),
            split + " source sampling-tier quota mismatch")
    return {"quota_axis": QUOTA_AXIS, "n_problems": len(rows),
            "source_sampling_tier_counts": {level: sum(r["sampling_tier"] == level for r in rows) for level in LEVELS},
            "intrinsic_difficulty_counts": {level: sum(r["difficulty"] == level for r in rows) for level in LEVELS},
            "by_domain": {d: {"source_sampling_tier_counts": {level: sum(r["domain"] == d and r["sampling_tier"] == level for r in rows) for level in LEVELS},
                              "intrinsic_difficulty_counts": {level: sum(r["domain"] == d and r["difficulty"] == level for r in rows) for level in LEVELS}}
                          for d in domains}, "source_sampling_tier_is_intrinsic_difficulty": False}


def failure_rates(data, denominator):
    timeout, infra = set(), set()
    infra_statuses = {"infraerror", "infrastructure", "exception", "engine_error", "selector_error", "run_aborted", "aborted"}
    for row in data["predictions"] + data.get("failures", []):
        key = (row.get("problem_id"), row.get("seed"))
        status = row.get("kind", row.get("status"))
        if status == "timeout":
            timeout.add(key)
        if status in infra_statuses:
            infra.add(key)
    return {"denominator": denominator, "timeout_count": len(timeout), "infra_count": len(infra),
            "timeout_infra_count": len(timeout | infra), "timeout_rate": len(timeout) / denominator,
            "infra_rate": len(infra) / denominator, "timeout_infra_rate": len(timeout | infra) / denominator}


def check_approval(approval, pilot_report, manifest, frozen_bytes, condition, *, inputs, config, prompts):
    require(approval is not None, "Separate budget approval required")
    require(isinstance(approval, (str, Path)), "Budget approval must be a separate sentinel file")
    raw = safe_read(approval)
    approved = json.loads(raw)
    scope = "latency" if condition == "LATENCY" else "test"
    hours = approved.get("max_gpu_hours")
    require(approved.get("approved") is True and approved.get("config_sha256") == sha256(frozen_bytes)
            and approved.get("scope") in {scope, "test_and_latency"}
            and isinstance(approved.get("approved_by"), str) and approved["approved_by"].strip()
            and type(hours) in (int, float) and math.isfinite(hours) and 0 < hours <= 25
            and isinstance(approved.get("pilot_report_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", approved["pilot_report_sha256"]),
            "Invalid config/pilot-bound budget approval sentinel (max_gpu_hours must be <=25)")
    at = datetime.fromisoformat(str(approved.get("approved_utc", "")).replace("Z", "+00:00"))
    require(at.tzinfo is not None and at.utcoffset() == timedelta(0), "Approval timestamp must be UTC")
    require(at.timestamp() <= timestamp(manifest["created_utc"]) and at <= datetime.now(timezone.utc),
            "Approval must precede execution")
    authorization = manifest.get("authorization", {})
    require(authorization.get("required") is True and authorization.get("checked") is True
            and authorization.get("direct_api_gate") is True and authorization.get("gpu_hour_meter") == "operator-owned"
            and authorization.get("approval_sha256") == sha256(raw)
            and authorization.get("pilot_report_sha256") == approved["pilot_report_sha256"]
            and all(authorization.get(k) == approved[k] for k in ("approved_by", "approved_utc", "max_gpu_hours"))
            and authorization.get("scope") == scope, "Manifest authorization/approval SHA mismatch or unchecked consent")
    # No reserialization: even a legacy-named consent file is bound by its supplied bytes.
    require(isinstance(pilot_report, (str, Path)), "Separate pilot report required for approval binding")
    report_raw = safe_read(pilot_report)
    require(sha256(report_raw) == approved["pilot_report_sha256"], "Approval pilot report SHA mismatch")
    report = json.loads(report_raw)
    entries, aggregate = report.get("conditions", {}), report.get("aggregate", {})
    require(report.get("protocol_version") == "3" and report.get("code_version") == "0.3.0"
            and report.get("go") is True and report.get("decision") == "go" and report.get("errors") == []
            and report.get("budget_approval") is False and set(entries) == set(CONDITIONS)
            and aggregate.get("complete") is True and aggregate.get("denominator") == aggregate.get("observed_cases") == 600,
            "Complete technical-only pilot GO required")
    total_failures = 0
    artifact_hashes, tags = {}, set()
    operational = None
    for arm, entry in entries.items():
        count = 150 if arm in SAMPLED else 50
        verification, rates = entry.get("verification", {}), entry.get("rates", {})
        failures = rates.get("timeout_infra_count")
        require(entry.get("verified") is True and entry.get("denominator") == count
                and verification.get("ok") is True and verification.get("records") == count
                and verification.get("condition") == arm
                and all(isinstance(verification.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", verification[k])
                        for k in ("archive_sha256", "notebook_sha256", "config_hash"))
                and rates.get("denominator") == count and type(failures) is int and 0 <= failures < .05 * count
                and rates.get("timeout_infra_rate") == failures / count,
                "Incomplete/failed technical pilot arm: " + arm)
        total_failures += failures
        recorded = verification.get("archive", "")
        drive = re.fullmatch(r"/mnt/([a-z])/(.+)", recorded) if sys.platform == "win32" else None
        if drive:
            recorded = drive[1].upper() + ":/" + drive[2]
        recorded_path = Path(recorded)
        name = str(recorded_path).replace("\\", "/").rsplit("/", 1)[-1]
        require(name.endswith("_final.zip") and "gold" not in name.lower(), "Pilot requires public final archive paths")
        candidates = [recorded_path, Path(pilot_report).parent / recorded_path,
                      Path(pilot_report).parent / "incoming" / name, Path(pilot_report).parent / name,
                      ROOT / "incoming" / name, ROOT / "results" / "v3" / name]
        path = next((p for p in candidates if p.is_file()), None)
        require(path is not None, "Missing public pilot final: " + arm)
        snapshot = read_archive(path)
        require(snapshot["archive_sha256"] == verification["archive_sha256"], "Changed public pilot final: " + arm)
        pilot_manifest = snapshot["manifest"]
        pilot_cfg = pilot_manifest.get("config", {})
        params, coverage = pilot_cfg.get("params", {}), pilot_manifest.get("expected_coverage", {})
        operational = check_operational_transition(pilot_cfg.get("code_sha256"), manifest["config"].get("code_sha256"),
                                                  approved, manifest, sha256(frozen_bytes), sha256(report_raw))
        require(canonical_hash(pilot_cfg) == pilot_manifest.get("config_hash") == verification["config_hash"]
                and pilot_manifest.get("condition") == params.get("CONDITION") == arm
                and params.get("SPLIT") == "pilot" and params.get("SEEDS") == ("17,29,43" if arm in SAMPLED else "17")
                and coverage.get("n_problems") == 50 and coverage.get("n_predictions") == count
                and pilot_manifest.get("coverage_complete") is True and pilot_manifest.get("finished_utc")
                and all(pilot_cfg.get(k) == manifest["config"].get(k) for k in (
                    "protocol_version", "code_version", "experiment_config_sha256", "data_sha256", "prompt_sha256", "models", "profile"))
                and (operational is not None or pilot_cfg.get("code_sha256") == manifest["config"].get("code_sha256")),
                "Pilot config/data/coverage binding mismatch: " + arm)
        info = verify_records(snapshot, inputs=Path(inputs).with_name("pilot_inputs.jsonl"), config=config,
                              expected_count=50, split="pilot", run_tag=params["RUN_TAG"], condition=arm, prompts=prompts)
        require(info["records"] == verification["records"] and failure_rates(snapshot, count) == rates,
                "Technical pilot rates/coverage differ from actual bound ledgers: " + arm)
        notebooks = condition_notebooks(path.parent, arm)
        require(len(notebooks) == 1 and check_notebook(notebooks[0], pilot_manifest) == verification["notebook_sha256"],
                "Pilot notebook evidence missing/changed: " + arm)
        artifact_hashes[arm] = snapshot["archive_sha256"]
        tags.add(params.get("RUN_TAG"))
    require(aggregate.get("timeout_infra_count") == total_failures
            and aggregate.get("timeout_infra_rate") == total_failures / 600 < .05, "Invalid aggregate technical pilot rate")
    require(len(tags) == 1 and next(iter(tags)) and authorization.get("pilot_archive_sha256") == artifact_hashes,
            "Manifest authorization pilot archive binding mismatch")
    if operational is not None:
        from jevlab.v3.operational import build_calendar, validate_calendar

        directory = Path(inputs).parent
        source_raw, schedule_raw = safe_read(directory / "latency_plan.json"), safe_read(directory / "schedule.json")
        require(operational["latency_source_plan_sha256"] == manifest["config"]["data_sha256"].get("latency_plan.json")
                == sha256(source_raw) and operational["latency_schedule_sha256"]
                == manifest["config"]["data_sha256"].get("schedule.json") == sha256(schedule_raw),
                "Operational approval source plan/schedule binding mismatch")
        subset = jsonl(safe_read(directory / "latency_inputs.jsonl").decode("utf-8"))
        metadata = {r["item_id"]: r for r in json.loads(source_raw)["items"]}
        order = json.loads(schedule_raw)["order"]
        validate_calendar(build_calendar(subset, order, metadata), subset, order, metadata, operational)
    return {"approval_sha256": sha256(raw), "pilot_report_sha256": sha256(report_raw), "max_gpu_hours": hours,
            **({"operational_amendment": operational} if operational is not None else {})}


def check_preflight(data, cfg, condition, frozen):
    """Reduced manifest flags are not substitutes for the full executed checks."""
    full = data.get("artifacts", {}).get("preflight/summary.json", {})
    required = BLOCKING | ({"selector_native_reference", "selector_equivalence"}
                           if condition in {"JFINAL", "LATENCY"} else set())
    require(data["manifest"].get("preflight_full") == "preflight/summary.json"
            and required <= set(full), "Missing detailed full preflight artifact")
    for name in required:
        result = full[name]
        summary = data["manifest"]["preflight"].get(name, {})
        require(result.get("ok") is True and result.get("skipped") is not True
                and (name == "continuity" or result.get("blocking") is True)
                and all(summary.get(k) == result.get(k) for k in ("ok", "blocking", "phase_epoch", "phase_id", "gpu_uuid", "resident_roles", "cleanup_confirmed")),
                "Full preflight failed/skipped or differs from manifest: " + name)
        require(data.get("artifacts", {}).get(phase_folder(result) + "/" + name + ".json") == result,
                "Missing/mismatched individual detailed check: " + name)
    require(all(r.get("ok") is True and r.get("skipped") is not True for r in full.values() if r.get("blocking")),
            "Failed detailed blocking preflight")
    require(full["parser_selftest"].get("failures") == [] and full["lock"].get("params")
            and full["chat_template"].get("roles") and full["tokenizer"].get("roles")
            and full["prompt_length"].get("over") == {} and type(full["prompt_length"].get("n_prompts")) is int
            and full["prompt_length"]["n_prompts"] > 0, "Missing detailed pre-engine checks")
    if condition in {"JFINAL", "LATENCY"}:
        reference = full["selector_native_reference"]
        folder = phase_folder(reference)
        native = data.get("artifacts", {}).get(folder + "/native_ref_out.json")
        payload = data.get("artifacts", {}).get(folder + "/native_ref_in.json", {})
        native_inputs = payload.get("items")
        identity = reference.get("identity", {})
        require(native and isinstance(native_inputs, list) and len(native_inputs) == len(native.get("items", [])) >= 8
                and full["selector_native_reference"].get("native") == native,
                "Missing/mismatched native selector artifacts")
        require(type(reference.get("returncode")) is int and reference["returncode"] == 0 and identity.get("config_hash") == canonical_hash(cfg)
                and isinstance(identity.get("snapshot"), str) and identity["snapshot"].strip()
                and identity.get("model") == cfg["models"]["J"] and identity.get("runtime") == cfg["jevk5_runtime"]
                and identity.get("prompt_hash") == canonical_hash(native_inputs)
                and payload.get("identity") == native.get("identity") == identity
                and native.get("reference_id") == reference.get("reference_id") == canonical_hash(identity)
                and payload.get("native_execution_id") == native.get("native_execution_id") == reference.get("native_execution_id")
                and native.get("runtime_commit") == cfg["jevk5_runtime"]["commit"]
                and isinstance(native.get("runtime_version"), str) and native["runtime_version"].strip()
                and native.get("snapshot") == identity.get("snapshot") and str(native.get("model_device", "")).startswith("cuda")
                and native.get("model_dtype") == "torch.bfloat16" and native.get("cuda_hardware", {}).get("returncode") == 0
                and data["manifest"]["gpu_uuid"] in native.get("cuda_hardware", {}).get("stdout", ""),
                "Native selector model/runtime/execution/CUDA identity mismatch")
        require(isinstance(native.get("native_execution_id"), str) and native["native_execution_id"],
                "Missing actual native execution identity")
        phase_folder({"phase_id": native["native_execution_id"]})
        temperature = native.get("temperature")
        require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0
                and temperature == frozen.get("selector", {}).get("calibration_temperature_expected"),
                "Native selector calibration mismatch")
        check_equivalence(full["selector_equivalence"], native, native_inputs)
    else:
        native, native_inputs = None, None
    if condition != "LATENCY":
        check_postflight(full, "H" if condition == "JFINAL" else "B" if condition.startswith("B13") else
                         "O" if condition == "Q9_GREEDY" else "G", cfg, native, native_inputs, latency=False)
        metadata = data["manifest"].get("active_phase", {})
        require(metadata.get("phase_id") and type(metadata.get("phase_epoch")) is int and metadata["phase_epoch"] >= 1,
                "Missing successful resident phase identity")
        detail = data.get("artifacts", {}).get(phase_folder(metadata) + "/summary.json", {})
        require(all(full.get(name) == result for name, result in detail.items()), "Resident phase differs from full preflight summary")
        check_phase_artifacts(data, detail, metadata, native, native_inputs)
        require(all(all(r.get(k) == metadata.get(k) for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles", "cleanup_confirmed"))
                    for name in ("predictions", "metrics", "proposals", "candidates", "decisions", "rounds") for r in data.get(name, [])),
                "Case/trace outside its successful resident phase")
    return native, native_inputs


def check_equivalence(result, native, native_inputs):
    rows = result.get("rows", [])
    require(result.get("ok") is True and result.get("blocking") is True and result.get("skipped") is not True
            and len(rows) == len(native["items"]) == len(native_inputs)
            and result.get("identity") == native.get("identity") and result.get("reference_id") == native.get("reference_id")
            and result.get("native_execution_id") == native.get("native_execution_id"), "Incomplete/unbound selector equivalence")
    for index, (row, reference, item) in enumerate(zip(rows, native["items"], native_inputs)):
        n = len(item.get("question", {}).get("criteria", []))
        vectors = [row.get("vllm_logits"), row.get("native_logits"), row.get("vllm_probs"), row.get("native_probs")]
        require(n >= 2 and all(isinstance(v, list) and len(v) == n
                and all(type(x) in (int, float) and math.isfinite(x) for x in v) for v in vectors),
                "Missing/invalid selector equivalence vectors")
        vl, nl, vp, np = vectors
        require(row.get("same_ids") is True and reference.get("ids_equal_prompt_text") is True
                and isinstance(reference.get("ids_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", reference["ids_sha256"])
                and type(reference.get("n_ids")) is int and reference["n_ids"] > 0
                and row.get("n_ids") == reference["n_ids"]
                and nl == reference.get("raw_logits") and np == reference.get("probabilities"),
                "Native/equivalence token or output identity mismatch")
        ids = reference.get("ids")
        require(isinstance(ids, list) and len(ids) == reference["n_ids"] and all(type(x) is int and x >= 0 for x in ids)
                and reference["ids_sha256"] == sha256(json.dumps(ids).encode())
                and row.get("ids") == ids and row.get("ids_sha256") == reference["ids_sha256"]
                and row.get("native_ids") == ids and row.get("native_ids_sha256") == reference["ids_sha256"]
                and row.get("index") == reference.get("index") == index
                and row.get("reference_id") == reference.get("reference_id") == native.get("reference_id")
                and row.get("native_execution_id") == reference.get("native_execution_id") == native.get("native_execution_id")
                and all(row.get(k) == result.get(k) for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles")),
                "Missing actual native/equivalence token IDs")
        for logits, probs in ((vl, vp), (nl, np)):
            exponentials = [math.exp((x - max(logits)) / native["temperature"]) for x in logits]
            normalized = [x / sum(exponentials) for x in exponentials]
            require(all(0 <= p <= 1 and math.isclose(p, q, abs_tol=1e-5, rel_tol=1e-5)
                        for p, q in zip(probs, normalized)) and abs(sum(probs) - 1) <= 1e-5,
                    "Invalid calibrated selector probabilities")
        av, an = vp.index(max(vp)), np.index(max(np))
        sorted_logits = sorted(nl, reverse=True)
        margin = sorted_logits[0] - sorted_logits[1]
        ulp = 2.0 ** (math.floor(math.log2(max(abs(sorted_logits[0]), 1e-6))) - 7)
        tie = margin <= 2 * ulp and max(nl) - nl[av] <= 2 * ulp
        require(row.get("argmax_vllm") == av and row.get("argmax_native") == an
                and (av == an or tie), "Selector equivalence argmax fails bf16 tolerance")


def check_postflight(result, phase, cfg, native, native_inputs, *, latency=True):
    required = {"eager_vs_graph", "cache", "continuity", "stress", "detailed_trace_audit"} | ({"selector_equivalence", "batching_audit"} if phase == "H" else set())
    require(required <= set(result) and all(result[k].get("ok") is True
            and (k == "eager_vs_graph" or result[k].get("skipped") is not True) for k in required)
            and all(r.get("ok") is True and r.get("skipped") is not True for r in result.values() if r.get("blocking")),
            "Missing/failed fresh full phase postflight")
    roles = {"G", "J"} if phase == "H" else {"B"} if phase == "B" else {"O"} if phase == "O" else {"G"}
    cache = result["cache"]
    require(cache.get("blocking") is True and set(cache.get("roles", {})) == roles
            and all(v is True for v in cache["roles"].values()), "Phase cache/resident role mismatch")
    rows = result["continuity"].get("rows", [])
    require(len(rows) == 6 and {r.get("prefix") for r in rows} == {1, 63, 64, 65, 127, 128}
            and all(type(r.get("agree_16")) is int and 0 <= r["agree_16"] <= 16 for r in rows)
            and sum(r["agree_16"] for r in rows) >= .75 * 16 * len(rows), "Incomplete/failed continuity evidence")
    stress = result["stress"]
    generator = stress.get("generator", {})
    n = 4 if phase == "H" else 1
    require(stress.get("blocking") is True and stress.get("synthetic") is True
            and stress.get("outside_T_total") is True and stress.get("generator_role") == ("G" if phase == "H" else next(iter(roles)))
            and generator.get("n") == n and generator.get("tokens") == [cfg["params"]["MAX_OUTPUT_TOKENS"]] * n
            and stress.get("rendered_prompt") and stress.get("rendered_prompt_sha256") == sha256(stress["rendered_prompt"].encode()),
            "Missing/mismatched full synthetic stress evidence")
    if phase == "H":
        require(native is not None, "Phase has no native selector reference")
        check_equivalence(result["selector_equivalence"], native, native_inputs)


def check_phase_artifacts(data, detail, metadata, native, native_inputs):
    require(metadata.get("phase_id") and metadata.get("cleanup_confirmed") is True
            and metadata.get("gpu_uuid") == data["manifest"]["gpu_uuid"] and detail, "Missing successful resident phase evidence")
    folder = phase_folder(metadata)
    for name, result in detail.items():
        require(all(result.get(k) == metadata.get(k) for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles", "cleanup_confirmed"))
                and data.get("artifacts", {}).get(folder + "/" + name + ".json") == result,
                "Stale/missing detailed phase check identity")
    log = data.get("artifacts", {}).get(folder + "/detailed_checks.jsonl", [])
    require(log and all(all(record.get("result", {}).get(k) == metadata.get(k) for k in
            ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles")) for record in log), "Missing/stale incremental phase audit log")
    last = {record.get("check"): record.get("result") for record in log}
    require(all(last.get(name) == result for name, result in detail.items()), "Incremental checks differ from final detailed artifacts")
    audit = detail.get("detailed_trace_audit", {})
    raw = data.get("artifacts", {}).get(folder + "/vllm_phase.log")
    span = audit.get("engine_log", {})
    require(isinstance(raw, bytes) and raw.strip() and span.get("current_phase_only") is True
            and type(span.get("start_byte")) is int and span["start_byte"] >= 0
            and type(span.get("end_byte")) is int and span["end_byte"] - span["start_byte"] == len(raw)
            and span.get("sha256") == sha256(raw), "Missing/corrupt current-phase engine trace")
    if "J" in metadata.get("resident_roles", []):
        trace = data.get("artifacts", {}).get(folder + "/native_reference_trace.json", {})
        require(all(trace.get(k) == metadata.get(k) for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles"))
                and trace.get("output") == native and trace.get("input", {}).get("items") == native_inputs
                and trace.get("identity") == native.get("identity") and trace.get("reference_id") == native.get("reference_id"),
                "Missing/stale native phase equivalence trace")


def check_swaps(data, cfg, rows, gpu_uuid, native, native_inputs):
    swaps = data.get("swaps", [])
    require(swaps, "LATENCY requires successful phase epochs/swaps")
    previous, end = None, timestamp(data["manifest"]["created_utc"])
    files = set()
    for epoch, swap in enumerate(swaps, 1):
        phase = swap.get("to")
        roles = {"B"} if phase == "B" else {"G", "J"}
        start, finish = timestamp(swap["t_start_utc"]), timestamp(swap["t_end_utc"])
        require(phase in {"B", "H"} and swap.get("from") == previous and phase != previous
                and type(swap.get("phase_epoch")) is int and swap["phase_epoch"] == epoch
                and swap.get("gpu_uuid") == gpu_uuid and swap.get("method") == "reload"
                and swap.get("ok") is True and swap.get("error") is None
                and swap.get("cleanup_confirmed") is True and swap.get("outside_T_total") is True
                and swap.get("load_time_measured_as_latency") is False
                and len(swap.get("resident_roles", [])) == len(roles) and set(swap["resident_roles"]) == roles
                and end <= start <= finish
                and all(type(swap.get(k)) in (int, float) and math.isfinite(swap[k]) and swap[k] >= 0 for k in ("seconds", "load_seconds")),
                "Invalid/failed/foreign-GPU phase epoch or overlapping reload")
        closed = swap.get("closed", {})
        require(set(closed) == (set() if previous is None else {"B"} if previous == "B" else {"G", "J"}),
                "Previous resident roles not exclusively closed")
        for closure in closed.values():
            pids, exitcodes = closure.get("pids", []), closure.get("exitcodes", [])
            require(closure.get("cleanup_confirmed") is True and pids and len(pids) == len(set(pids)) == len(exitcodes)
                    and all(type(pid) is int and pid > 0 for pid in pids) and all(type(code) is int for code in exitcodes),
                    "Missing confirmed closed-process evidence")
        filename = phase_folder(swap) + "/summary.json"
        detail = data.get("artifacts", {}).get(filename)
        require(isinstance(filename, str) and filename.startswith("preflight/") and filename not in files
                and detail is not None and detail == swap.get("postflight"), "Missing/reused phase postflight artifact")
        files.add(filename)
        check_phase_artifacts(data, detail, swap, native, native_inputs)
        check_postflight(detail, phase, cfg, native, native_inputs)
        executed = [r for r in rows if r.get("phase_epoch") == epoch]
        require(executed and all(type(r.get("phase_epoch")) is int and r.get("phase") == phase
                and r.get("phase_id") == swap.get("phase_id") and r.get("resident_roles") == swap.get("resident_roles")
                and r.get("cleanup_confirmed") is True
                and finish <= timestamp(r["t_start_utc"]) for r in executed), "Case predates/mismatches its successful phase epoch")
        end = max(timestamp(r["t_end_utc"]) for r in executed)
        previous = phase
    require(all(type(r.get("phase_epoch")) is int and 1 <= r["phase_epoch"] <= len(swaps) for r in rows),
            "Case has missing/invalid phase epoch")


def case_provenance(cfg, condition):
    """Reconstruct the runtime's gold-free generation commitment, without engines."""
    params = cfg["params"]
    role = "O" if condition == "Q9_GREEDY" else "B" if condition.startswith("B13") else "G"
    sampling = {name: params[name.upper()] for name in ("temperature", "top_p", "top_k", "repetition_penalty",
                                                       "presence_penalty", "frequency_penalty")}
    if condition not in SAMPLED:
        sampling.update(temperature=0.0, top_p=1.0, top_k=0)
    return {"protocol_version": "3", "code_version": "0.3.0", "model": cfg["models"][role],
            "generator_prompt_sha256": cfg["prompt_sha256"]["generador"], "sampling": sampling,
            "limits": {"n_candidates": 4, "max_output_tokens": params["MAX_OUTPUT_TOKENS"],
                       "hybrid_candidate_budget": params["CANDIDATE_BUDGET"], "gb_context": params["GB_CONTEXT"]},
            "profile": cfg["profile"], "data_sha256": cfg["data_sha256"][params["SPLIT"] + "_inputs.jsonl"],
            "enable_thinking": False}


def branch_seeds(pid, seed):
    return [int.from_bytes(hashlib.sha256(f"{seed}|{pid}|G4_FINAL4_V3|0|{branch}|0".encode()).digest()[:8], "little")
            % (2**63 - 1) for branch in range(4)]


def check_traces(data, cfg, rows, inputs, condition, gpu_uuid):
    digest = canonical_hash(cfg)
    by_resume = {r["resume_key"]: r for r in rows}
    require(len(by_resume) == len(rows), "Duplicate completion resume identity")
    public_inputs = {r["id"]: r["problem"] for r in inputs}
    pools, candidates, decisions, failures = {}, {}, {}, {}
    rounds = {}
    engine_events = {}
    for name, grouped in (("proposals", pools), ("candidates", candidates), ("decisions", decisions),
                          ("rounds", rounds), ("failures", failures)):
        for trace in data.get(name, []):
            if name == "failures" and trace.get("kind") == "engine_exception":
                matches = [r for r in rows if r["status"] in ERROR_STATUSES
                           and (not trace.get("resume_key") or trace["resume_key"] == r["resume_key"])
                           and all(trace.get(k) == r.get(k) for k in ("phase_id", "phase_epoch", "gpu_uuid"))
                           and timestamp(r["t_start_utc"]) <= timestamp(trace.get("ts")) <= timestamp(r["t_end_utc"])]
                if not matches and not trace.get("resume_key"):
                    matches = [r for r in data.get("warmup", []) if r.get("status") in ERROR_STATUSES
                        and r.get("outside_T_total") is True and type(r.get("generation")) is int and r["generation"] >= 900
                        and all(trace.get(k) == r.get(k) for k in ("phase_id", "phase_epoch", "gpu_uuid"))
                        and trace.get("attempted_ids") == (r.get("engine_cleanup") or {}).get("attempted_ids")]
                require(len(matches) == 1 and type(trace.get("phase_epoch")) is int
                        and trace.get("config_hash") == digest and trace.get("error") and trace.get("traceback")
                        and trace.get("engine_role") in matches[0].get("resident_roles", [])
                        and isinstance(trace.get("attempted_ids"), list) and trace["attempted_ids"]
                        and all(isinstance(rid, str) and rid for rid in trace["attempted_ids"]), "Unbound engine failure trace")
                if trace.get("resume_key"):
                    require(all(trace.get(k) == matches[0].get(k) for k in ("problem_id", "seed", "condition"))
                            and type(trace.get("seed")) is int
                            and all(trace[k] == matches[0].get(k) for k in ("generation", "candidate_set_id", "item_id", "repetition", "execution_id", "block", "case_index", "domain", "difficulty", "sampling_tier") if k in trace),
                            "Engine request/case metadata linkage mismatch")
                group = (matches[0].get("resume_key", "warmup"), trace["engine_role"], tuple(trace["attempted_ids"]))
                engine_events.setdefault(group, []).append(trace)
                continue
            pred = by_resume.get(trace.get("resume_key"))
            require(pred is not None, "Orphan/foreign trace: " + name)
            identity = ("problem_id", "condition", "seed", "candidate_set_id", "generation")
            if condition == "LATENCY":
                identity += ("item_id", "repetition", "execution_id", "phase", "phase_epoch", "phase_id", "domain", "difficulty", "sampling_tier")
                if data["manifest"].get("authorization", {}).get("operational_amendment") is not None:
                    identity += ("block", "case_index", "cohort", "residency")
            require(trace.get("config_hash") == digest and trace.get("gpu_uuid") == gpu_uuid
                    and type(trace.get("seed")) is int and type(trace.get("generation")) is int and trace["generation"] == 0
                    and (condition != "LATENCY" or type(trace.get("repetition")) is int)
                    and all(trace.get(k) == pred.get(k) for k in identity), "Trace case/hash/GPU identity mismatch: " + name)
            grouped.setdefault(trace["resume_key"], []).append(trace)
    for events in engine_events.values():
        require(len(events) == 2 and events[0].get("cleanup_pending") is True and events[1].get("cleanup_pending") is False
                and events[0]["error"] == events[1]["error"] and events[1].get("engine_usable") is True
                and events[1].get("engine_cleanup", {}).get("cleanup_confirmed") is True,
                "Missing/unconfirmed terminal engine failure cleanup trace")
    for row in rows:
        key = row["resume_key"]
        expected_provenance = canonical_hash(case_provenance(cfg, row["condition"]))
        require(row.get("provenance_hash") == expected_provenance, "Generation provenance commitment mismatch")
        errors = failures.get(key, [])
        require((row["status"] in ERROR_STATUSES) == bool(errors), "Error outcome must have a matching failure trace")
        require(len(errors) <= 1 and all(f.get("status", f.get("kind")) == row["status"] for f in errors),
                "Failure status mismatch/duplicate")
        if row["condition"] != "JFINAL":
            require(not pools.get(key) and not decisions.get(key) and not rounds.get(key), "Reference arm has foreign hybrid traces")
            continue
        require(len(pools.get(key, [])) == 1, "Every JFINAL case requires exactly one persisted proposal set")
        pool = pools[key][0]
        seeds = branch_seeds(row["problem_id"], row["seed"])
        set_id = canonical_hash({"problem_id": row["problem_id"], "problem": public_inputs[row["problem_id"]],
                                 "provenance_hash": expected_provenance, "proposal_seeds": seeds,
                                 "proposal_condition": "G4_FINAL4_V3"})
        require(pool.get("candidate_set_id") == row.get("candidate_set_id") == set_id
                and pool.get("provenance_hash") == expected_provenance
                and pool.get("proposal_namespace") == "G4_FINAL4_V3"
                and type(pool.get("planned_candidate_slots")) is int and pool["planned_candidate_slots"] == 4
                and pool.get("seed_branches") == seeds,
                "Proposal stage/stochastic seed/set commitment mismatch")
        cs = pool["candidates"]
        require(len(rounds.get(key, [])) <= 1 and all(r.get("round") == 0 and r.get("provenance_hash") == expected_provenance
                for r in rounds.get(key, [])), "Foreign proposal round/stage")
        for candidate in cs:
            require(all(candidate.get(k) == row.get(k) for k in ("problem_id", "condition", "seed", "generation", "candidate_set_id", "provenance_hash"))
                    and type(candidate.get("round")) is int and candidate["round"] == 0 and type(candidate.get("seed_branch")) is int
                    and candidate["seed_branch"] == seeds[candidate["branch"]]
                    and ("config_hash" not in candidate or candidate["config_hash"] == digest),
                    "Embedded candidate has foreign identity/stage/seed/hash")
        require(pool["status"] != "complete" or not any(c["status"] in ERROR_STATUSES for c in cs),
                "Error/timeout candidate cannot be fabricated as a complete pool")
        require(row["status"] != "final" or pool["status"] == "complete", "Partial pool cannot produce a primary final")
        logged = candidates.get(key, [])
        fields = ("branch", "seed_branch", "public_output", "status", "round", "provenance_hash")
        require(Counter(tuple(c.get(k) for k in fields) for c in cs)
                == Counter(tuple(c.get(k) for k in fields) for c in logged), "Embedded/persisted candidate ledger mismatch")
        selection = decisions.get(key, [])
        require(len(selection) == int(pool["status"] == "complete"), "Missing/extra selector decision trace")
        if selection:
            decision = selection[0]
            require(type(decision.get("round")) is int and decision["round"] == 0 and decision.get("provenance_hash") == expected_provenance,
                    "Selector decision stage/provenance mismatch")
            raw = f"{row['seed']}|{row['problem_id']}|JFINAL|0|0|option_order".encode()
            perm_seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
            perm = list(range(4))
            random.Random(perm_seed).shuffle(perm)
            require(decision.get("perm_seed") == perm_seed and decision.get("permutation") == perm,
                    "Selector permutation stochastic identity mismatch")
            if decision.get("status") == "ok":
                winner = row.get("winner_branch")
                require(type(winner) is int and decision.get("winner_branch") == winner
                        and decision.get("winner_position") == perm.index(winner), "Selector decision/winner mismatch")
            else:
                require(decision.get("status") == row["status"] and row.get("winner_branch") is None
                        and decision.get("winner_branch") is None, "Failed selector cannot log a fabricated winner")


def check_notebook(path, manifest=None):
    raw = safe_read(path)
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        nb = ast.literal_eval(raw.decode("utf-8"))
    pm = nb.get("metadata", {}).get("papermill", {})
    require(pm.get("end_time") and "exception" in pm and pm["exception"] in (None, False),
            "Notebook incomplete or exception")
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code" and c.get("source")]
    require(cells and all(c.get("metadata", {}).get("papermill", {}).get("status") == "completed"
                         and not any(o.get("output_type") == "error" for o in c.get("outputs", []))
                          for c in cells), "Unexecuted or failed notebook cells")
    if manifest is not None:
        recorded = {}
        for cell in cells:
            tags = cell.get("metadata", {}).get("tags", [])
            if not set(tags) & {"parameters", "injected-parameters"}:
                continue
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        recorded[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        expected = manifest["config"]["params"]
        for key in ("CONDITION", "RUN_TAG", "SPLIT", "SEEDS", "CONFIG_FILE"):
            if key in expected:
                require(recorded.get(key) == expected[key], "Notebook parameter/run identity mismatch: " + key)
        output_text = json.dumps([output for cell in cells for output in cell.get("outputs", [])])
        require(manifest["run_name"] + "_final.zip" in output_text,
                "Notebook does not confirm the exact final archive")
    return sha256(raw)


def verify_records(data, *, inputs, config, expected_count, split, run_tag, condition,
                   prompts=ROOT / "prompts", approval=None, pilot_report=None, seeds=None):
    """Same validation for archives and analysis snapshots; no quality decisions."""
    require(condition in CONDITIONS + ("LATENCY",), "Unknown v3 physical condition")
    require(split in {"pilot", "test", "dev"}, "Unknown split")
    require(isinstance(run_tag, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag)
            and (split == "dev" or "smoke" not in run_tag.lower()), "Unsafe/development-only run tag")
    frozen_bytes = safe_read(config)
    frozen = json.loads(frozen_bytes)
    manifest = data["manifest"]
    cfg = manifest["config"]
    params = cfg["params"]
    digest = canonical_hash(cfg)
    for obj in (frozen, cfg):
        require(obj.get("protocol_version") == "3" and obj.get("code_version") == "0.3.0",
                "Requires protocol '3' and code '0.3.0'")
    require(manifest.get("config_hash") == digest, "Manifest config hash mismatch")
    require(isinstance(cfg.get("code_sha256"), dict) and cfg["code_sha256"] and all(
            isinstance(name, str) and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for name, value in cfg["code_sha256"].items()),
            "Missing executed-code provenance hashes")
    require(manifest.get("condition") == params.get("CONDITION") == condition, "Condition mismatch")
    require(params.get("SPLIT") == split and params.get("RUN_TAG") == run_tag, "Split/run tag mismatch")
    require(cfg.get("experiment_config_sha256") == sha256(frozen_bytes), "Frozen config SHA mismatch")
    require(cfg.get("models") == frozen.get("models") and set(cfg["models"]) == {"G", "J", "B", "O"},
            "Frozen G/J/B/O models mismatch")
    require(all(m.get("repo") and re.fullmatch(r"[0-9a-f]{40}", m.get("revision", ""))
                for m in cfg["models"].values()), "Models require full pinned revision SHAs")
    require(cfg["models"]["O"]["repo"] == "Qwen/Qwen3.5-9B"
            and cfg["models"]["O"]["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "O must be official pinned 9B")
    require(cfg.get("profile") == frozen.get("implementation_profile"), "Frozen profile mismatch")
    require(cfg.get("jevk5_runtime") == frozen.get("jevk5_runtime"), "Frozen runtime mismatch")
    for section in ("limits", "sampling", "runtime_defaults"):
        for name, value in frozen.get(section, {}).items():
            if section == "runtime_defaults" and name in {"PREFLIGHT_MODE", "ABORT_ON_PREFLIGHT_FAIL", "WARMUP_N",
                    "CHECKPOINT_EVERY", "NVML_HZ", "PERSIST_TO_DRIVE", "DRIVE_DIR", "LOG_LEVEL", "ROOT",
                    "HF_CACHE", "REQUIRED_GPU", "ENGINE_CLOSE_TIMEOUT_S", "N_PROBLEMS"}:
                continue
            parameter = {"hybrid_candidate_budget": "CANDIDATE_BUDGET", "n_candidates": "N_CANDIDATES"}.get(name, name.upper())
            if section == "sampling" and name in {"enable_thinking", "presence_penalty", "frequency_penalty"}:
                fixed = False if name == "enable_thinking" else 0.0
                require(value == fixed and (parameter not in params or params[parameter] == fixed),
                        "Frozen non-thinking/neutral-penalty mismatch: " + name)
                continue
            if section == "runtime_defaults" and name == "SEEDS":
                continue  # Per-arm schedules below, not the notebook's shared default.
            require(params.get(parameter) == value, f"Frozen budget/runtime mismatch: {parameter}")
    recorded_seeds = tuple(int(s) for s in str(params.get("SEEDS", "")).replace(" ", "").split(",") if s)
    required_seeds = SEEDS if condition in SAMPLED or condition == "LATENCY" else (17,)
    require(recorded_seeds == required_seeds and (seeds is None or tuple(seeds) == required_seeds),
            "Physical seed schedule mismatch")
    input_bytes = safe_read(inputs)
    input_rows = jsonl(input_bytes.decode("utf-8"))
    ids = [r["id"] for r in input_rows]
    require(len(ids) == len(set(ids)), "Duplicate input IDs")
    require(all(set(r) == {"id", "problem", "language"} and r["language"] == "en"
                and all(isinstance(v, str) and v.strip() for v in r.values()) for r in input_rows),
            "Non-input fields in inputs")
    target = 50 if split == "pilot" else 500 if split == "test" else expected_count
    require(len(ids) == target and expected_count == target, "Expected unique input count mismatch")
    hashes = cfg.get("data_sha256", {})
    require(hashes.get(Path(inputs).name) == sha256(input_bytes), "Inputs SHA mismatch")
    allowed_data = {"pilot_inputs.jsonl", "test_inputs.jsonl", "dev_inputs.jsonl", "schedule.json",
                    "dataset_manifest.json", "latency_inputs.jsonl", "latency_plan.json", "SHA256SUMS"}
    for name, value in hashes.items():
        require(name in allowed_data, "Unapproved data hash member (gold forbidden)")
        require(sha256(safe_read(Path(inputs).parent / name)) == value, f"Data SHA mismatch: {name}")
    seal_bytes = safe_read(Path(inputs).parent / "SHA256SUMS")
    require(hashes.get("SHA256SUMS") == sha256(seal_bytes), "Run must bind the public dataset seal SHA (without opening gold)")
    seals = {}
    for line in seal_bytes.decode().splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            require(name not in seals, "Duplicate seal entry")
            seals[name] = value
    require(seals.get(Path(inputs).name) == sha256(input_bytes), "Input seal mismatch")
    metadata_raw = safe_read(Path(inputs).parent / "dataset_manifest.json")
    require(seals.get("dataset_manifest.json") == sha256(metadata_raw), "Dataset manifest seal mismatch")
    metadata = json.loads(metadata_raw)
    require(metadata.get("input_sha256", {}).get(Path(inputs).name) == sha256(input_bytes), "Public dataset input commitment mismatch")
    review = metadata.get("review", {})
    counts = metadata.get("actual_n", {})
    require(metadata.get("dataset_version") == "v3" and metadata.get("sealed") is True
            and all(counts.get(k) == n for k, n in {"test": 500, "pilot": 50, "dev": 20}.items())
            and set(counts) <= {"test", "pilot", "dev", "latency"} and counts.get("latency", 100) == 100,
            "Unsealed/noncanonical v3 dataset")
    require(review.get("agent_reviewed_all") is True and review.get("human_reviewed") is False,
            "Requires agent review true, human review false")
    require(isinstance(review.get("policy_id"), str) and review["policy_id"].strip()
            and re.fullmatch(r"[0-9a-f]{64}", review.get("policy_sha256", "")), "Missing review policy identity")
    policy_hash = sha256(safe_read(Path(inputs).parent / "REVIEW_CONTRACT.md"))
    require(seals.get("REVIEW_CONTRACT.md") == policy_hash == review["policy_sha256"], "Review policy seal/hash mismatch")
    bound_policy = frozen.get("dataset_review_policy")
    if bound_policy:
        require(review["policy_id"] == bound_policy.get("id") and review["policy_sha256"] == bound_policy.get("sha256"),
                "Frozen review policy mismatch")
    source_provenance = check_public_provenance(metadata, seals, Path(inputs).parent)
    prompt_hashes = cfg.get("prompt_sha256", {})
    require(set(prompt_hashes) == {"generador", "criterio_paso", "criterio_final"}, "Missing prompt hashes")
    for name, value in prompt_hashes.items():
        text = safe_read(Path(prompts) / (name + ".txt")).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        require(sha256(text.encode()) == value, f"Prompt SHA mismatch: {name}")
    consent = check_approval(approval, pilot_report, manifest, frozen_bytes, condition, inputs=inputs, config=config,
                             prompts=prompts) if split == "test" or condition == "LATENCY" else None
    require(manifest.get("finished_utc") and manifest.get("stages")
             and all(s.get("ok") is True for s in manifest["stages"].values()), "Incomplete execution stages")
    created, finished = timestamp(manifest["created_utc"]), timestamp(manifest["finished_utc"])
    require(created < finished, "Invalid manifest execution interval")
    preflight = manifest.get("preflight", {})
    required = BLOCKING | ({"selector_native_reference", "selector_equivalence"}
                           if condition in {"JFINAL", "LATENCY"} else set())
    # The shared continuity probe labels itself advisory; v3 still requires it to pass.
    require(required <= set(preflight) and all((k == "continuity" or preflight[k].get("blocking") is True)
            and preflight[k].get("ok") is True and preflight[k].get("skipped") is not True for k in required)
            and all(p.get("ok") is True for p in preflight.values() if p.get("blocking")),
             "Missing or failed blocking preflight (including cache continuity)")
    native, native_inputs = check_preflight(data, cfg, condition, frozen)
    locks = manifest.get("model_lock", {})
    roles = {"G", "J", "B"} if condition == "LATENCY" else {"G", "J"} if condition == "JFINAL" else {
        "B"} if condition.startswith("B13") else {"O"} if condition == "Q9_GREEDY" else {"G"}
    require(roles <= set(locks) and all(l.get("ok") is True for l in locks.values()), "Model lock failed/missing")
    require(all(locks[r].get("repo") == cfg["models"][r]["repo"]
                and locks[r].get("revision") == cfg["models"][r]["revision"] for r in roles), "Model lock identity mismatch")
    env = manifest.get("environment", {})
    gpus = env.get("gpus", [])
    require(len(gpus) == 1 and "A100" in gpus[0].get("name", "")
            and 39 <= float(gpus[0].get("memory.total", 0)) / 1024 <= 41 and env.get("cuda_available") is True,
            "Requires one CUDA A100 40GB")
    gpu_uuid = gpus[0].get("uuid")
    require(isinstance(gpu_uuid, str) and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid)
            and manifest.get("gpu_uuid") == gpu_uuid, "Physical GPU UUID missing/invalid/mismatched")
    probe = manifest.get("hw_probe", {})
    require(probe.get("cuda") and "A100" in probe.get("device", "")
            and all(type(probe.get(k)) in (int, float) and math.isfinite(probe[k]) and probe[k] > 0
                    for k in ("bf16_gemm_tflops", "d2d_copy_gbps", "small_kernel_us", "python_loop_ms")),
            "Missing/failed CUDA hardware probe")
    for name in LEDGERS:
        require(manifest.get("counts", {}).get(name, 0) == len(data.get(name, [])), f"Count mismatch: {name}")
    require(set(manifest.get("counts", {})) <= set(LEDGERS), "Unknown counted ledger")
    rows = data["latency_metrics" if condition == "LATENCY" else "predictions"]
    if condition == "LATENCY":
        plan = data.get("latency_plan")
        require(split == "test" and plan is not None, "LATENCY needs test subset plan")
        items = plan["items"]
        selected = [item["item_id"] for item in items]
        require(len(selected) == len(set(selected)) == 100 and set(selected) <= set(ids), "Latency test subset mismatch")
        subset_path = Path(inputs).parent / "latency_inputs.jsonl"
        subset_raw = safe_read(subset_path)
        subset = jsonl(subset_raw.decode("utf-8"))
        require(hashes.get(subset_path.name) == seals.get(subset_path.name) == sha256(subset_raw),
                "Latency subset must be bound to frozen config and dataset seal")
        require(len(subset) == 100 and {r["id"] for r in subset} == set(selected), "Latency differs from frozen test subset")
        frozen_plan_path = Path(inputs).parent / "latency_plan.json"
        frozen_plan_bytes = safe_read(frozen_plan_path)
        require(hashes.get(frozen_plan_path.name) == seals.get(frozen_plan_path.name) == sha256(frozen_plan_bytes),
                "Run must bind the sealed public latency plan")
        frozen_plan = json.loads(frozen_plan_bytes)
        frozen_items = frozen_plan["items"]
        require(frozen_plan.get("quota_axis") == QUOTA_AXIS and frozen_plan.get("quota_per_domain") == SAMPLING_QUOTAS["latency"]
                and plan.get("quota_axis", QUOTA_AXIS) == QUOTA_AXIS, "Latency must bind source sampling-tier policy")
        allowed_item_fields = {"item_id", "problem", "language", "domain", "difficulty", "sampling_tier"}
        require(all(set(i) <= allowed_item_fields for i in items + frozen_items), "Latency plan may contain only blind item metadata")
        require(len(frozen_items) == 100 and {i["item_id"] for i in frozen_items} == set(selected), "Frozen latency plan coverage mismatch")
        frozen_by_id = {i["item_id"]: i for i in frozen_items}
        require(all(all(i.get(k) == frozen_by_id[i["item_id"]].get(k) for k in ("problem", "domain", "difficulty", "sampling_tier"))
                    for i in items), "Archived item labels/content differ from frozen latency plan")
        input_by_id = {r["id"]: r for r in input_rows}
        require(all(input_by_id.get(r["id"]) == r for r in subset), "Frozen latency/test input content differs")
        require(all(i.get("problem") == input_by_id[i["item_id"]]["problem"] for i in items),
                "Latency plan/input content mismatch")
        latency_profile = sampling_profile(items, "latency")
        require(frozen_plan.get("intrinsic_difficulty_counts") == latency_profile["intrinsic_difficulty_counts"],
                "Latency actual intrinsic difficulty counts mismatch")
        source_provenance["latency_profile"] = latency_profile
        policy = consent.get("operational_amendment") if consent else None
        if policy is not None:
            schedule_raw = safe_read(Path(inputs).parent / "schedule.json")
            require(hashes.get("schedule.json") == seals.get("schedule.json") == sha256(schedule_raw),
                    "Operational schedule must remain sealed")
            check_operational_calendar(data, subset, frozen_plan, sha256(frozen_plan_bytes), schedule_raw, policy)
        else:
            require(not any(k in plan for k in ("operational_amendment", "calendar", "residency_phases"))
                    and "operational_metadata" not in manifest, "Unapproved operational latency calendar")
        expected = {(pid, arm, seed, rep) for pid in selected for arm in ("JFINAL", "B13_GREEDY")
                    for seed in SEEDS for rep in (0, 1, 2)}
        keys = [(r["item_id"], r["condition"], r["seed"], r["repetition"]) for r in rows]
        require(all(r.get("clock_scope") == "candidate_generation_inclusive" for r in rows),
                "Latency clock must include candidate generation")
        require(len({r.get("execution_id") for r in rows}) == len(rows)
                 and all(r.get("execution_id") for r in rows), "Actual latency execution IDs required")
        require(all(isinstance(r["execution_id"], str) and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", r["execution_id"])
            for r in rows), "Latency execution IDs must be actual UUID4 identities")
        intervals = [(r["t_start_utc"], r["t_end_utc"]) for r in rows]
        require(len(set(intervals)) == len(rows), "Cloned latency timestamps")
        perf_intervals = [(r.get("t_start_perf"), r.get("t_end_perf")) for r in rows]
        require(all(all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                        for t in interval) for interval in perf_intervals)
                and len(set(perf_intervals)) == len(rows), "Actual unique perf-counter intervals required for latency repeats")
        ordered_perf = sorted(perf_intervals)
        require(all(a[1] <= b[0] for a, b in zip(ordered_perf, ordered_perf[1:])), "Overlapping/cloned latency perf intervals")
        row_key = lambda r: (r.get("item_id"), r.get("condition"), r.get("seed"), r.get("repetition"))
        for name in ("predictions", "metrics"):
            require(Counter(row_key(r) for r in data[name]) == Counter(keys), f"Latency {name} coverage mismatch")
        require(all(r.get("gpu_uuid") == gpu_uuid and r.get("config_hash") == digest
                    for name in ("predictions", "metrics") for r in data[name]), "Latency companion provenance mismatch")
        indexed = {row_key(r): r for r in rows}
        require(all(all(r.get(k) == indexed[row_key(r)].get(k) for k in
                        ("status", "effective_seed", "T_total", "t_start_utc", "resume_key", "execution_id", "phase", "phase_epoch",
                         "generation", "candidate_set_id", "provenance_hash", "public_output", "profile", "problem_id", "phase_id", "resident_roles", "cleanup_confirmed", "domain", "difficulty", "sampling_tier"))
                    for name in ("predictions", "metrics") for r in data[name]), "Latency companion clock/status mismatch")
        require(all(timestamp(r["t_end_utc"]) >= timestamp(indexed[row_key(r)]["t_end_utc"])
                    for name in ("predictions", "metrics") for r in data[name]),
                 "Latency companion end timestamp predates actual execution end")
        for row in rows:
            key = sha256(f"{digest}|{row['item_id']}|{row['condition']}|{row['seed']}|{row['repetition']}".encode())
            require(row.get("problem_id") == row["item_id"] and row.get("resume_key") == key
                    and row.get("phase") == ("H" if row["condition"] == "JFINAL" else "B"),
                    "Latency resume/item/phase identity mismatch")
            require(all(row.get(k) == frozen_by_id[row["item_id"]][k] for k in ("domain", "difficulty", "sampling_tier")),
                    "Latency case reviewed-domain/intrinsic/source-tier metadata mismatch")
        check_swaps(data, cfg, rows, gpu_uuid, native, native_inputs)
    else:
        expected = {(pid, condition, seed) for pid in ids for seed in required_seeds}
        keys = [(r["problem_id"], r["condition"], r["seed"]) for r in rows]
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, "Exact physical coverage mismatch")
    for row in rows:
        require(type(row.get("seed")) is int and (condition != "LATENCY" or type(row.get("repetition")) is int),
                "Integer seed/repetition slots required")
        effective = row["seed"] if row["condition"] in SAMPLED else None
        require("effective_seed" in row and row["effective_seed"] == effective, "Effective seed mismatch")
        require(row.get("config_hash") == digest and row.get("status") in STATUSES, "Row hash/status mismatch")
        require(row.get("gpu_uuid") == gpu_uuid, "Row physical GPU mismatch")
        require(row.get("profile") == cfg["profile"], "Row implementation profile mismatch")
        require(type(row.get("generation")) is int and row["generation"] == 0, "Confirmatory proposal generation must be stage zero")
        require(isinstance(row.get("public_output"), str), "Every outcome must explicitly log public_output, including failures")
        elapsed = timestamp(row["t_end_utc"]) - timestamp(row["t_start_utc"])
        require(elapsed > 0 and created <= timestamp(row["t_start_utc"]) < timestamp(row["t_end_utc"]) <= finished,
                "Invalid actual execution timestamps/manifest interval")
        total = row.get("T_total")
        require(isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(total)
                and total > 0 and total <= elapsed + 0.1, "Total clock outside actual execution interval")
        if condition == "LATENCY":
            require(row["t_end_perf"] > row["t_start_perf"] and
                    math.isclose(row["t_end_perf"] - row["t_start_perf"], total, rel_tol=1e-9, abs_tol=1e-9),
                    "Latency total differs from actual perf interval")
        first = row.get("T_first_accepted")
        require(first is None or (isinstance(first, (int, float)) and not isinstance(first, bool)
                and math.isfinite(first) and 0 <= first <= total), "First-accepted timepoint outside case clock")
        require(row["status"] != "final" or first is not None, "Final outcome needs first-accepted timepoint")
        if row["condition"] == "JFINAL":
            proposal_time, selector_time = row.get("T_proposals"), row.get("T_selector")
            require(all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t >= 0
                        for t in (proposal_time, selector_time)) and total + 0.001 >= proposal_time + selector_time,
                    "Total clock excludes proposals/selector")
            require(first is None or first + 0.001 >= proposal_time + selector_time,
                    "JFINAL first acceptance precedes generation/selection")
        for field, limit in (("accepted_tokens", "MAX_OUTPUT_TOKENS"),
                             ("candidate_tokens_total", "CANDIDATE_BUDGET" if row["condition"] == "JFINAL" else "MAX_OUTPUT_TOKENS")):
            if limit in params:
                value = row.get(field)
                require(type(value) is int and 0 <= value <= params[limit], "Measured token budget exceeded/missing: " + field)
        if "TIMEOUT_S" in params:
            # SafeEngine bounds both abort RPC and unfinished-request cleanup at five seconds.
            require(total <= params["TIMEOUT_S"] + (11.0 if row["status"] == "timeout" else 1.0),
                    "Measured timeout/confirmed cleanup budget exceeded")
        if row.get("engine_cleanup") is not None:
            require(row["engine_cleanup"].get("cleanup_confirmed") is True and row.get("engine_usable") is True,
                    "Request cleanup unconfirmed; outcome is not certifiable")
    ordered = sorted((timestamp(r["t_start_utc"]), timestamp(r["t_end_utc"])) for r in rows)
    require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), "Overlapping/cloned physical executions")
    if condition != "LATENCY":
        metrics = data["metrics"]
        metric_keys = [(r["problem_id"], r["condition"], r["seed"]) for r in metrics]
        require(Counter(metric_keys) == Counter(keys), "Metrics coverage mismatch")
        require(all(r.get("config_hash") == digest for r in metrics), "Metrics hash mismatch")
        indexed = {(r["problem_id"], r["seed"]): r for r in rows}
        require(all(all(r.get(k) == indexed[(r["problem_id"], r["seed"])].get(k) for k in
                        ("status", "effective_seed", "gpu_uuid", "T_total", "t_start_utc", "t_end_utc"))
                    for r in metrics), "Metrics clock/status/provenance mismatch")
        for row in rows + metrics:
            key = sha256(f"{digest}|{row['problem_id']}|{row['seed']}|{condition}|{cfg['profile']}".encode())
            require(row.get("resume_key") == key, "Resume identity mismatch")
        if condition == "JFINAL":
            proposals = data["proposals"]
            require(Counter((r["problem_id"], r["condition"], r["seed"]) for r in proposals) == Counter(keys),
                    "One candidate-set per JFINAL case required, including failures")
            require(len({r.get("candidate_set_id") for r in proposals}) == len(proposals)
                    and all(r.get("candidate_set_id") for r in proposals), "Candidate set identity mismatch")
            slots = 0
            for row in proposals:
                cs = row["candidates"]
                slots += len(cs)
                require(row.get("config_hash") == digest and row.get("status") in POOL_STATUSES
                         and len(cs) <= 4 and len({c["branch"] for c in cs}) == len(cs), "Invalid candidate set")
                require(all(isinstance(c.get("public_output"), str) and c.get("status") in STATUSES
                            and type(c.get("branch")) is int and c["branch"] in range(4)
                            and type(c.get("seed")) is int and c["seed"] == row["seed"]
                            for c in cs), "Invalid candidate fields")
                pred = indexed[(row["problem_id"], row["seed"])]
                require(pred.get("candidate_set_id") == row["candidate_set_id"], "Prediction/proposal linkage mismatch")
                require(row["status"] != "complete" or len(cs) == 4, "Complete candidate set needs four branches")
                require((not cs) == (row["status"] == "no_candidates"), "Empty candidates must be explicit no_candidates")
                require(pred["status"] != "final" or row["status"] == "complete", "Final prediction cannot use a failed candidate set")
                winner = pred.get("winner_branch")
                require(pred["status"] != "final" or type(winner) is int, "Final JFINAL outcome needs its selected branch")
                if winner is not None:
                    selected = [c for c in cs if c["branch"] == winner]
                    require(type(winner) is int and len(selected) == 1 and pred["public_output"] == selected[0]["public_output"]
                            and pred["status"] == selected[0]["status"], "JFINAL prediction is not its logged selected candidate")
            require(slots <= 6000, "Candidate slots exceed 6000")
    if condition == "LATENCY":
        proposals = data["proposals"]
        hybrids = {row_key(r): r for r in rows if r["condition"] == "JFINAL"}
        require(Counter(row_key(r) for r in proposals) == Counter(hybrids.keys()), "Latency JFINAL proposal coverage mismatch")
        for proposal in proposals:
            cs = proposal.get("candidates", [])
            require(proposal.get("candidate_set_id") == hybrids[row_key(proposal)].get("candidate_set_id")
                    and proposal.get("candidate_set_id") and proposal.get("config_hash") == digest
                    and proposal.get("status") in POOL_STATUSES,
                    "Latency proposal linkage mismatch")
            require(len(cs) <= 4 and len({c.get("branch") for c in cs}) == len(cs)
                    and all(isinstance(c.get("public_output"), str) and c.get("status") in STATUSES
                            and type(c.get("branch")) is int and c["branch"] in range(4)
                            and type(c.get("seed")) is int and c["seed"] == proposal["seed"] for c in cs),
                    "Invalid latency candidate fields")
            require(proposal["status"] != "complete" or len(cs) == 4, "Complete latency pool needs four branches")
            require((not cs) == (proposal["status"] == "no_candidates"), "Empty latency pool needs explicit no_candidates")
            pred = hybrids[row_key(proposal)]
            winner = pred.get("winner_branch")
            require(pred["status"] != "final" or type(winner) is int, "Final latency JFINAL outcome needs its selected branch")
            if winner is not None:
                selected = [c for c in cs if c["branch"] == winner]
                require(type(winner) is int and len(selected) == 1 and pred["public_output"] == selected[0]["public_output"]
                        and pred["status"] == selected[0]["status"], "Latency JFINAL is not its logged selected candidate")
    for failure in data.get("failures", []):
        if failure.get("kind") == "engine_exception":
            continue  # Phase/timestamp-bound pending and terminal traces are checked below.
        key = (failure.get("problem_id"), condition, failure.get("seed"))
        require((row_key(failure) in expected if condition == "LATENCY" else key in expected), "Failure outside planned coverage")
    check_traces(data, cfg, rows, input_rows, condition, gpu_uuid)
    final_roles = set(data["swaps"][-1]["resident_roles"]) if condition == "LATENCY" else roles
    cleanup = manifest.get("final_cleanup", {})
    require(set(cleanup) == final_roles and all(c.get("cleanup_confirmed") is True and c.get("pids")
            and len(c["pids"]) == len(c.get("exitcodes", [])) and all(type(pid) is int and pid > 0 for pid in c["pids"])
            and all(type(code) is int for code in c["exitcodes"]) for c in cleanup.values()),
            "Missing final confirmed closed-process evidence")
    require(data["progress"].get("done") == data["progress"].get("total") == len(expected), "Incomplete progress")
    return {"ok": True, "condition": condition, "records": len(rows), "config_hash": digest,
            "certificate_schema": "v3-offline-audit-1", "code_sha256": cfg["code_sha256"],
            "archive_sha256": data.get("archive_sha256"), "statuses": dict(Counter(r["status"] for r in rows)),
            "authorization": consent, "dataset_provenance": source_provenance}


def verify_archive(path, condition, **kwargs):
    path = Path(path)
    require(path.name.endswith("_final.zip"), "Final archive required")
    data = read_archive(path)
    info = verify_records(data, condition=condition, **kwargs)
    run = f"{condition}_{kwargs['run_tag']}_{info['config_hash'][:8]}"
    require(data["manifest"].get("run_name") == run and path.name == run + "_final.zip", "Archive identity mismatch")
    return {**info, "archive": str(path)}


def condition_notebooks(directory, condition):
    candidates = []
    for path in Path(directory).glob("*.out.*.ipynb"):
        matches = [c for c in CONDITIONS + ("LATENCY",) if re.match(
            r"^(?:[0-9]+_)?" + re.escape(c) + r"(?:_|\.)", path.name)]
        if matches and max(matches, key=len) == condition and not path.name.endswith(".canonical.ipynb"):
            candidates.append(path)
    return candidates


def verify_run(directory, *, conditions, notebooks=None, **kwargs):
    require(len(conditions) == len(set(conditions)), "Duplicate requested conditions")
    results = {}
    directory = Path(directory)
    for condition in conditions:
        try:
            paths = list(directory.glob(f"{condition}_{kwargs['run_tag']}_*_final.zip"))
            require(len(paths) == 1, f"Expected one final archive for {condition}, found {len(paths)}")
            info = verify_archive(paths[0], condition, **kwargs)
            if notebooks and condition in notebooks:
                candidates = [Path(notebooks[condition])]
            else:
                candidates = condition_notebooks(directory, condition)
            require(len(candidates) == 1, "Expected one condition-bound executed notebook")
            info["notebook_sha256"] = check_notebook(candidates[0], read_archive(paths[0])["manifest"])
            results[condition] = info
        except Exception as error:
            results[condition] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"ok": bool(results) and all(r["ok"] for r in results.values()), "runs": results,
            "note": "Integrity only. No gold access, grading, or implicit budget approval."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--conditions", nargs="+", required=True, choices=CONDITIONS + ("LATENCY",))
    for name in ("inputs", "config", "prompts", "approval", "pilot-report", "output"):
        parser.add_argument("--" + name, type=Path, required=name in {"inputs", "config"},
                            default=ROOT / "prompts" if name == "prompts" else None)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "pilot", "test"), required=True)
    parser.add_argument("--run-tag", required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    report = verify_run(**args)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output:
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
