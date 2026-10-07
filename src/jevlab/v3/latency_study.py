"""Sealed, counterbalanced two-arm latency study; lifecycle work is unmeasured."""

from __future__ import annotations

import time
from collections import Counter

from ..common import append_jsonl, config_hash, read_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json
from .operational import build_calendar, residency_phases, validate_calendar
from .runner import Experiment


class _PhaseView:
    """Expose the active arm to preflight without changing the experiment identity."""

    def __init__(self, exp, condition):
        # Stress preflight snapshots vars(view), so forwarded attributes alone do not suffice.
        self.__dict__.update(vars(exp))
        self.exp, self.cond = exp, condition

    def __getattr__(self, name):
        return getattr(self.exp, name)

    def roles(self):
        return list(self.exp.engines)


class LatencyStudy(Experiment):
    def __init__(self, **params):
        defaults = dict(LAT_N_ITEMS=100, LAT_REPETITIONS=3, LAT_BLOCK_SIZE=10,
                        LAT_CONDITIONS="JFINAL,B13_GREEDY", LATENCY_SWAP_MODE="reload",
                        SEEDS="17,29,43", SPLIT="test")
        super().__init__(**{**defaults, **params, "CONDITION": "LATENCY"})
        self.conds = [c.strip() for c in str(self.p["LAT_CONDITIONS"]).split(",")]
        if len(self.conds) != 2 or set(self.conds) != {"JFINAL", "B13_GREEDY"}:
            raise ValueError("LAT_CONDITIONS requires exactly JFINAL and B13_GREEDY")
        if self.p["LATENCY_SWAP_MODE"] != "reload":
            raise ValueError("latency supports reload only, never sleep")
        if self.p["SPLIT"] not in ("test", "dev"):
            raise ValueError("latency SPLIT must be test or explicit dev smoke")
        for name in ("LAT_N_ITEMS", "LAT_REPETITIONS", "LAT_BLOCK_SIZE"):
            value = self.p[name]
            if int(value) < 1 or float(value) != int(value):
                raise ValueError(f"{name} must be a positive integer")
        if self.p["SPLIT"] == "test" and (
                int(self.p["LAT_N_ITEMS"]) != 100 or int(self.p["LAT_REPETITIONS"]) != 3
                or self.seeds != [17, 29, 43]):
            raise ValueError("confirmatory latency requires 100 items x 3 repetitions x seeds 17,29,43")
        if self.p["SPLIT"] == "dev" and (
                int(self.p["LAT_N_ITEMS"]) != 2 or int(self.p["LAT_REPETITIONS"]) != 1
                or self.seeds != [17]):
            raise ValueError("dev smoke requires LAT_N_ITEMS=2, LAT_REPETITIONS=1, SEEDS=17")
        self.f.update({k: str(self.dir / f"{k}.jsonl") for k in ("latency_metrics", "swaps")})
        self.phase = None
        successful = [r.get("phase_epoch", 0) for r in read_jsonl(self.f["swaps"]) if r.get("ok") is True]
        self.phase_epoch = max([self.phase_epoch, *successful])
        self.phase_id = None
        self.phase_cleanup_confirmed = False
        self.ctx, self.selector = None, None
        items = self.latency_items()  # Validate the frozen subset before any GPU work.
        self.manifest["latency_metadata_schema"] = {
            "quota_axis": "source_sampling_tier", "difficulty": "actual reviewed intrinsic difficulty",
            "sampling_tier": "original source sampling tier, not intrinsic difficulty"}
        self.manifest["expected_coverage"] = {
            "split": self.p["SPLIT"], "problem_ids": [it["id"] for it in items],
            "n_problems": len(items), "seeds": self.seeds, "conditions": self.conds,
            "repetitions": int(self.p["LAT_REPETITIONS"]),
            "n_predictions": len(items) * len(self.seeds) * len(self.conds)
                             * int(self.p["LAT_REPETITIONS"])}
        self.save_manifest()

    def roles(self):
        return ["G", "J", "B"]

    def latency_items(self):
        self._validate_dataset()
        if self.p["SPLIT"] == "dev":
            rows = read_jsonl(self.data_dir / "dev_inputs.jsonl")
            self._validate_inputs(rows)
            if len(rows) < 2:
                raise ValueError("dev smoke requires two dev inputs")
            return rows[:2]
        manifest = read_json(self.data_dir / "dataset_manifest.json")
        path = self.data_dir / "latency_inputs.jsonl"
        counts = manifest.get("actual_n", {})
        if (manifest.get("dataset_version") != "v3" or manifest.get("sealed") is not True
                or any(counts.get(k) != v for k, v in {"test": 500, "pilot": 50, "dev": 20}.items())):
            raise ValueError("latency requires the canonical sealed v3 dataset")
        for name in ("latency_inputs.jsonl", "test_inputs.jsonl", "schedule.json"):
            expected = manifest.get("input_sha256", {}).get(name)
            if (not expected or sha256_file(self.data_dir / name) != expected
                    or self.dataset_seals.get(name) != expected):
                raise ValueError(f"sealed input SHA mismatch: {name}")
        rows, test = read_jsonl(path), read_jsonl(self.data_dir / "test_inputs.jsonl")
        self._validate_inputs(rows)
        self._validate_inputs(test)
        if len(rows) != 100 or len(test) != 500:
            raise ValueError("frozen latency subset must have 100 of 500 test inputs")
        indexed = {r["id"]: r for r in test}
        if any(indexed.get(r["id"]) != r for r in rows):
            raise ValueError("latency subset content must be identical to blind test inputs")
        order = read_json(self.data_dir / "schedule.json")["order"]
        if len(order) != 500 or len(set(order)) != 500 or set(order) != set(indexed):
            raise ValueError("schedule does not match test inputs")
        subset = {r["id"]: r for r in rows}
        self._frozen_latency_plan(subset)
        return [subset[i] for i in order if i in subset]

    def _frozen_latency_plan(self, subset):
        """Strata are blind metadata, never inferred from model output or gold."""
        path = self.data_dir / "latency_plan.json"
        public = read_json(self.data_dir / "dataset_manifest.json")
        if (not path.is_file() or self.dataset_seals.get(path.name) != sha256_file(path)
                or public.get("input_sha256", {}).get(path.name) != self.dataset_seals.get(path.name)):
            raise ValueError("latency requires sealed latency_plan.json with 100 blind item metadata records")
        plan = read_json(path)
        provenance = {"dataset_version": "v3", "sealed": True, "source_split": "test", "split": "test",
                      "quota_axis": "source_sampling_tier", "seeds": [17, 29, 43], "repetitions": 3,
                      "conditions": ["JFINAL", "B13_GREEDY"], "generation": 0, "expected_rows": 1800,
                      "quota_per_domain": {"easy": 6, "medium": 8, "hard": 6}}
        if (public.get("quota_axis") != "source_sampling_tier" or plan.get("sealed") is not True
                or type(plan.get("generation")) is not int or any(plan.get(k) != v for k, v in provenance.items())):
            raise ValueError("latency plan requires frozen test/source_sampling_tier provenance")
        rows = plan.get("items", [])
        if (len(rows) != 100 or len({r.get("item_id") for r in rows}) != 100
                or {r.get("item_id") for r in rows} != set(subset)):
            raise ValueError("latency plan items must exactly cover the frozen 100 inputs")
        allowed = {"item_id", "problem", "language", "domain", "difficulty", "sampling_tier"}
        domains = {"arithmetic", "algebra", "ratios_percentages", "number_theory", "counting_probability"}
        for row in rows:
            if (not set(row) <= allowed or not {"item_id", "problem", "domain", "difficulty", "sampling_tier"} <= set(row)
                    or row["problem"] != subset[row["item_id"]]["problem"]
                    or row.get("language", "en") != "en" or row["domain"] not in domains
                    or row["difficulty"] not in ("easy", "medium", "hard")
                    or row["sampling_tier"] not in ("easy", "medium", "hard")):
                raise ValueError("latency plan must contain identical blind content and approved strata, no gold")
        for domain in domains:
            if Counter(r["sampling_tier"] for r in rows if r["domain"] == domain) != {"easy": 6, "medium": 8, "hard": 6}:
                raise ValueError("latency source sampling tiers must be 6 easy / 8 medium / 6 hard per domain (20 items)")
        intrinsic = {level: sum(row["difficulty"] == level for row in rows) for level in ("easy", "medium", "hard")}
        actual_intrinsic = plan.get("intrinsic_difficulty_counts")
        if (actual_intrinsic != intrinsic or not isinstance(actual_intrinsic, dict)
                or any(type(n) is not int for n in actual_intrinsic.values())):
            raise ValueError("latency plan intrinsic difficulty counts must match its actual labels, not source quotas")
        return rows

    @staticmethod
    def _validate_inputs(rows):
        if any(set(r) != {"id", "problem", "language"}
               or any(not isinstance(r[k], str) or not r[k] for k in r) for r in rows):
            raise ValueError("blind inputs require exactly id, problem, language strings")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate input IDs")

    def problems(self):
        return self.latency_items()

    def items(self):
        return self.latency_items()

    def start_engines(self):
        self._check_authorization()
        if self.engines or set(self.paths) != set(self.roles()) or any(
                not self.lock.get(r, {}).get("ok") for r in self.roles()):
            raise RuntimeError("all latency snapshots must be verified before loading")

    def preflight_before_engines(self):
        from . import preflight

        with self.stage("preflight_pre_engines"):
            res = preflight.pre_engine_checks(self, "full")
            self.preflight_results.update(res)
            self._check_blocking(res)
            native = res.get("selector_native_reference", {})
            if not native.get("ok") or native.get("skipped"):
                raise RuntimeError("latency requires native selector reference before loading")

    def preflight_after_engines(self):
        # No resident engines yet: each load performs mandatory phase postflight.
        pass

    def warmup(self):
        # Only _swap may warm the active phase, using dev inputs exclusively.
        pass

    def _swap(self, phase):
        from . import preflight
        from .algorithms import Case, run_case

        if phase not in ("B", "H"):
            raise ValueError("phase must be B or H")
        self._validate_identity(context=False)
        self._assert_engines_usable()
        self.assert_same_gpu()
        if self.phase == phase:
            return
        started, wall_start, previous = time.perf_counter(), utc_now(), self.phase
        closed, load_seconds, error, res = {}, None, None, {}
        self.phase, self.ctx, self.selector = None, None, None
        condition = "B13_GREEDY" if phase == "B" else "JFINAL"
        roles = ["B"] if phase == "B" else ["G", "J"]
        self._begin_phase()
        self.phase_cleanup_confirmed = False
        try:
            closed = self._close_roles(list(self.engines))
            if self.engines or any(not r.get("cleanup_confirmed") for r in closed.values()):
                raise RuntimeError("previous phase cleanup not confirmed")
            self.phase_cleanup_confirmed = True
            self.assert_same_gpu()
            t_load = time.perf_counter()
            self._load_roles(roles)
            load_seconds = time.perf_counter() - t_load
            if set(self.engines) != set(roles):
                raise RuntimeError("latency phase has overlapping or missing resident engines")
            self.ctx = self._ctx(condition)
            self._validate_identity(condition)
            view = _PhaseView(self, condition)
            res = preflight.post_engine_checks(view, "full")
            self.preflight_results.update(res)
            self._check_blocking(res)
            required = ["continuity", "stress"] + (["selector_equivalence"] if phase == "H" else [])
            if any(not res.get(k, {}).get("ok") or res[k].get("skipped") for k in required):
                raise RuntimeError("mandatory latency phase postflight failed or skipped")
            for i, it in enumerate(self.dev_problems()[:int(self.p["WARMUP_N"])]):
                self._validate_identity(condition)
                self.assert_same_gpu()
                out = run_case(Case(it["id"], it["problem"], 17, condition, generation=900 + i),
                               self.ctx, self.p["TIMEOUT_S"], provenance=self._provenance(condition))
                append_jsonl(self.f["warmup"], {**out.run, "phase": phase,
                              **self._phase_metadata()})
                self._assert_engines_usable()
            self.assert_same_gpu()
            self.phase_epoch = self._pending_phase_epoch
            self.phase = phase
            self.manifest["active_phase"] = self._phase_metadata()
            self.save_manifest()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            res = {name: result for name, result in self.preflight_results.items()
                   if result.get("phase_id") == self.phase_id}
            self.ctx, self.selector, self.phase = None, None, None
            raise
        finally:
            append_jsonl(self.f["swaps"], {"from": previous, "to": phase, "method": "reload",
                          "gpu_uuid": self.gpu_uuid, "postflight": res,
                          "phase_epoch": self.phase_epoch, "phase_id": self.phase_id,
                          "attempted_phase_epoch": self._pending_phase_epoch, "closed": closed,
                         "cleanup_confirmed": error is None and all(
                             r.get("cleanup_confirmed") for r in closed.values()),
                           "load_seconds": load_seconds, "seconds": time.perf_counter() - started,
                           "t_start_perf": started, "t_end_perf": time.perf_counter(),
                          "load_time_measured_as_latency": False,
                         "outside_T_total": True, "t_start_utc": wall_start, "t_end_utc": utc_now(),
                          "resident_roles": list(self.engines), "ok": error is None, "error": error})
            self.__dict__.pop("_pending_phase_epoch", None)

    def _latency_key(self, item_id, condition, seed, repetition):
        return sha256_text(f"{self.chash}|{item_id}|{condition}|{seed}|{repetition}")

    def _prediction_key(self, row):
        if row["item_id"] != row["problem_id"]:
            raise ValueError("latency item/problem completion identity differs")
        return self._latency_key(row["item_id"], row["condition"], row["seed"], row["repetition"])

    def run(self):
        from .algorithms import Case, run_case

        self._check_authorization()
        items = self.latency_items()
        size = int(self.p["LAT_BLOCK_SIZE"])
        blocks = [items[i:i + size] for i in range(0, len(items), size)]
        plan_items = (self._frozen_latency_plan({it["id"]: it for it in items})
                      if self.p["SPLIT"] == "test" else
                      [{"item_id": it["id"], "problem": it["problem"], "language": it["language"]}
                       for it in items])
        metadata = {row["item_id"]: row for row in plan_items}
        policy = self.manifest.get("authorization", {}).get("operational_amendment")
        operational = {}
        if self.p["SPLIT"] == "test" and policy is not None:
            if size != 10:
                raise ValueError("operational calendar preserves original size-10 block identities")
            order = read_json(self.data_dir / "schedule.json")["order"]
            calendar = build_calendar(items, order, metadata)
            validate_calendar(calendar, items, order, metadata, policy)
            operational = {"operational_amendment": policy, "operational_policy_sha256": config_hash(policy),
                           "calendar_sha256": config_hash(calendar), "calendar": calendar,
                           "residency_phases": residency_phases}
            self.manifest["operational_metadata"] = operational
            self.save_manifest()
        else:
            calendar = [{"item_id": it["id"], "condition": "B13_GREEDY" if phase == "B" else "JFINAL",
                         "seed": seed, "repetition": repetition, "block": bi}
                        for repetition in range(int(self.p["LAT_REPETITIONS"]))
                        for bi, block in enumerate(blocks)
                        for phase in (("B", "H") if (bi + repetition) % 2 == 0 else ("H", "B"))
                        for seed in self.seeds for it in block]
        write_json(self.dir / "latency_plan.json", {
            "items": plan_items,
            "quota_axis": "source_sampling_tier",
            "source_plan_sha256": self.dataset_seals.get("latency_plan.json") if self.p["SPLIT"] == "test" else None,
            "intrinsic_difficulty_counts": {level: sum(row.get("difficulty") == level for row in plan_items)
                                            for level in ("easy", "medium", "hard")} if self.p["SPLIT"] == "test" else None,
            "blocks": [[it["id"] for it in block] for block in blocks],
            "seeds": self.seeds, "repetitions": int(self.p["LAT_REPETITIONS"]),
            "conditions": self.conds, "generation": 0, "split": self.p["SPLIT"], **operational})
        done = self._completed_keys()
        if self._orphan_keys(done):
            raise RuntimeError("uncommitted latency traces found; preserve this run and use a new RUN_TAG")
        case_index, n_new = 0, 0
        with self.stage("latency_loop"):
            indexed = {it["id"]: it for it in items}
            for descriptor in calendar:
                it = indexed[descriptor["item_id"]]
                condition, seed = descriptor["condition"], descriptor["seed"]
                repetition, bi = descriptor["repetition"], descriptor["block"]
                phase = "B" if condition == "B13_GREEDY" else "H"
                key = self._latency_key(it["id"], condition, seed, repetition)
                case_index += 1
                if key in done:
                    continue
                self._swap(phase)
                self._validate_identity(condition)
                self.assert_same_gpu()
                provenance = self._provenance(condition)
                extra = {**provenance, **self._case_context(), "gpu_uuid": self.gpu_uuid,
                         "item_id": it["id"], "seed": seed,
                         "domain": metadata[it["id"]].get("domain"),
                         "difficulty": metadata[it["id"]].get("difficulty"),
                         "sampling_tier": metadata[it["id"]].get("sampling_tier"),
                         "repetition": repetition, "phase": phase, "block": bi,
                         "phase_epoch": self.phase_epoch, "case_index": case_index,
                         "phase_id": self.phase_id, "resident_roles": list(self.engines),
                         "cleanup_confirmed": self.phase_cleanup_confirmed,
                         **{k: descriptor[k] for k in ("cohort", "residency") if k in descriptor}}
                if not extra.get("gpu_uuid"):
                    raise RuntimeError("latency provenance requires captured GPU UUID")
                wall_start = utc_now()
                self._active_case = {**extra, "resume_key": key, "problem_id": it["id"],
                                     "condition": condition}
                out = run_case(Case(it["id"], it["problem"], seed, condition, generation=0),
                               self.ctx, self.p["TIMEOUT_S"],
                               proposal_callback=self._proposal_callback(key, extra),
                               provenance=provenance)
                extra["output_sha256"] = sha256_text(out.run["raw_output"])
                extra["t_end_utc"] = utc_now()
                # Predictions written by _persist are the sole completion marker.
                latency_row = {
                    **{k: v for k, v in out.run.items() if k != "raw_output"},
                    **extra, "resume_key": key, "config_hash": self.chash, "t_start_utc": wall_start}
                self._persist(out, key, wall_start, extra=extra, completion_callback=lambda record:
                              append_jsonl(self.f["latency_metrics"], {
                                  **{k: v for k, v in record.items() if k != "raw_output"}, **latency_row}))
                self._active_case = {}
                done.add(key)
                self._assert_engines_usable()
                n_new += 1
                if n_new % int(self.p["CHECKPOINT_EVERY"]) == 0:
                    self.checkpoint()
        self.checkpoint()

    def finalize(self):
        self._assert_unpublished()
        coverage = self.manifest["expected_coverage"]
        expected = {self._latency_key(pid, condition, seed, repetition)
                    for pid in coverage["problem_ids"] for condition in self.conds
                    for seed in self.seeds for repetition in range(coverage["repetitions"])}
        predictions = read_jsonl(self.f["predictions"])
        actual = {r["resume_key"] for r in predictions}
        self._completed_keys()
        self.manifest.update(counts={k: len(read_jsonl(v)) for k, v in self.f.items()},
                             coverage_complete=actual == expected and len(predictions) == len(expected),
                             missing_predictions=len(expected - actual))
        return self._publish_final()

    def shutdown(self):
        self.phase, self.ctx, self.selector = None, None, None
        try:
            self._close_roles(list(self.engines))
        finally:
            if self.sampler:
                self.sampler.stop()


if __name__ == "__main__":
    from .runner import main

    main(default_condition="LATENCY")
