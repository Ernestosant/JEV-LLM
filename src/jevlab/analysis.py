"""Offline analysis for notebook 07 (protocol §12-§13, amendment v3.1).

Unit: problem. Accuracy is averaged over seeds within problem; latency is the per-problem
median over seeds. Paired bootstrap by problem id, stratified by domain, 10,000 resamples,
seed 271828; 95% (descriptive) and 99.1667% (Bonferroni over 6 primary comparisons).
Latencies from notebooks 01-06 come from different VMs and are DESCRIPTIVE; the confirmatory
latency comparison comes from notebook 08 (same VM, counterbalanced blocks).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config_hash, read_json, read_jsonl, sha256_file, write_json
from .evaluate import grade_prediction, grade_text, load_gold

log = logging.getLogger("jevlab.analysis")
PRIMARY = ["J64", "JSTEP", "JFINAL"]
ORDER = ["G_SINGLE", "B13", "J64", "JSTEP", "JFINAL", "B13_GREEDY"]
SECONDARY_V2 = [("J64", "B13"), ("JSTEP", "B13"), ("J64", "G_SINGLE"),
                ("JSTEP", "G_SINGLE"), ("JFINAL", "G_SINGLE")]
BLOCKING_V2 = {"parser_selftest", "lock", "chat_template", "prompt_length", "sampling_semantics", "tokenizer", "stress"}


# ----------------------------------------------------------------------- loading
def load_runs(results_root: str, gold_path: str, run_tag: str | None = None) -> dict:
    gold = load_gold(gold_path)
    preds, mets, cands, decs, fails, manifests, directories = [], [], [], [], [], {}, {}
    for p in Path(results_root).rglob("predictions.jsonl"):
        d = p.parent
        man = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
        tag = man.get("config", {}).get("params", {}).get("RUN_TAG")
        if run_tag and tag != run_tag:
            continue
        if d.name in manifests and (man.get("config", {}).get("protocol_version") == "2" or
                                    manifests[d.name].get("config", {}).get("protocol_version") == "2"):
            raise ValueError("Duplicate v2 run directory: " + d.name)
        manifests[d.name] = man
        directories[d.name] = d
        gpu = (man.get("environment", {}).get("gpus") or [{}])[0]
        for r in read_jsonl(p):
            r["run_name"] = d.name
            preds.append(r)
        for r in read_jsonl(d / "metrics.jsonl"):
            r["run_name"] = d.name
            r["gpu_name"] = gpu.get("name")
            r["gpu_uuid"] = gpu.get("uuid")
            r["host"] = man.get("environment", {}).get("host", {}).get("hostname")
            mets.append(r)
        cands += [dict(c, run_name=d.name) for c in read_jsonl(d / "candidates.jsonl")]
        decs += [dict(c, run_name=d.name) for c in read_jsonl(d / "decisions.jsonl")]
        fails += [dict(c, run_name=d.name) for c in read_jsonl(d / "failures.jsonl")]
    return {"gold": gold, "predictions": preds, "metrics": mets, "candidates": cands,
            "decisions": decs, "failures": fails, "manifests": manifests, "directories": directories}


def grade_all(runs: dict) -> pd.DataFrame:
    rows = []
    mets = {(m["problem_id"], m["seed"], m["condition"]): m for m in runs["metrics"]}
    for p in runs["predictions"]:
        g = runs["gold"].get(p["problem_id"])
        if g is None:
            continue
        r = grade_prediction(p, g)
        m = mets.get((p["problem_id"], p["seed"], p["condition"]), {})
        for k in ("T_total", "T_first_accepted", "T_proposals", "T_selector", "T_prefill", "T_control",
                  "candidate_tokens_total", "accepted_tokens", "overproduction_tokens", "rounds",
                  "selector_calls", "selector_input_tokens", "processed_prompt_tokens", "duplicate_rounds",
                  "empty_steps", "forced_boundaries", "boundary_overshoots", "discarded_fraction",
                  "nvml_peak_used_bytes", "gpu_name", "gpu_uuid", "host", "public_output_chars",
                  "t_start_utc", "t_end_utc"):
            r[k] = m.get(k)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- bootstrap
def per_problem(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    d = df[df.condition == cond]
    return d.groupby("problem_id").agg(acc=("correct", "mean"), lat=("T_total", "median"),
                                       domain=("domain", "first"))


def paired_bootstrap(df: pd.DataFrame, a: str, b: str = "B13", n_boot: int = 10000,
                     seed: int = 271828) -> dict | None:
    """Accuracy difference (a - b, in points) and speedup (median_b / median_a)."""
    pa, pb = per_problem(df, a), per_problem(df, b)
    common = sorted(set(pa.index) & set(pb.index))
    if not common:
        return None
    pa, pb = pa.loc[common], pb.loc[common]
    dom = pa["domain"].values
    strata = {d: np.where(dom == d)[0] for d in np.unique(dom)}
    rng = np.random.default_rng(seed)
    acc_a, acc_b = pa["acc"].values, pb["acc"].values
    lat_a, lat_b = pa["lat"].values.astype(float), pb["lat"].values.astype(float)
    diffs, speed = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata.values()])
        diffs[i] = 100 * (acc_a[idx].mean() - acc_b[idx].mean())
        with np.errstate(all="ignore"):
            speed[i] = np.nanmedian(lat_b[idx]) / np.nanmedian(lat_a[idx])

    def ci(x, level):
        lo, hi = np.nanpercentile(x, [50 * (1 - level), 100 - 50 * (1 - level)])
        return [float(lo), float(hi)]

    est_d = 100 * (acc_a.mean() - acc_b.mean())
    est_s = float(np.nanmedian(lat_b) / np.nanmedian(lat_a)) if np.nanmedian(lat_a) > 0 else float("nan")
    return {"method": a, "baseline": b, "n_problems": len(common),
            "acc_diff_pp": float(est_d), "acc_diff_ci95": ci(diffs, 0.95),
            "acc_diff_ci99_1667": ci(diffs, 0.991667),
            "speedup": est_s, "speedup_ci95": ci(speed, 0.95), "speedup_ci99_1667": ci(speed, 0.991667),
            "n_boot": n_boot, "seed": seed, "strata": {k: int(len(v)) for k, v in strata.items()}}


def acc_ci(df: pd.DataFrame, cond: str, n_boot: int = 10000, seed: int = 271828) -> list[float]:
    p = per_problem(df, cond)
    if p.empty:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    dom = p["domain"].values
    strata = [np.where(dom == d)[0] for d in np.unique(dom)]
    a = p["acc"].values
    bs = [a[np.concatenate([rng.choice(ix, len(ix)) for ix in strata])].mean() for _ in range(n_boot)]
    return [float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)]


# ----------------------------------------------------------------------- tables
def summary_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for c in [c for c in ORDER if c in set(df.condition)]:
        d = df[df.condition == c]
        pp = per_problem(df, c)
        bt = paired_bootstrap(df, c, "B13", n_boot) if c != "B13" and "B13" in set(df.condition) else None
        ttot = d["T_total"].astype(float)
        rows.append({
            "condition": c, "n_runs": len(d), "n_problems": d.problem_id.nunique(),
            "accuracy_pct": 100 * pp["acc"].mean() if len(pp) else float("nan"),
            "accuracy_ci95": acc_ci(df, c, n_boot),
            "diff_vs_B13_pp": bt["acc_diff_pp"] if bt else None,
            "diff_ci99_1667": bt["acc_diff_ci99_1667"] if bt else None,
            "latency_p50_s": float(np.nanmedian(pp["lat"])) if len(pp) else None,
            "latency_p95_s": float(np.nanpercentile(ttot.dropna(), 95)) if ttot.notna().any() else None,
            "speedup_vs_B13_descr": bt["speedup"] if bt else None,
            "speedup_ci99_1667_descr": bt["speedup_ci99_1667"] if bt else None,
            "cand_tokens_mean": d["candidate_tokens_total"].astype(float).mean(),
            "accepted_tokens_mean": d["accepted_tokens"].astype(float).mean(),
            "J_input_tokens_mean": d["selector_input_tokens"].astype(float).mean(),
            "J_tokens_per_accepted": (d["selector_input_tokens"].astype(float).sum() /
                                      max(1.0, d["accepted_tokens"].astype(float).sum())),
            "selection_cost_time_frac": (d["T_selector"].astype(float).sum() /
                                         max(1e-9, d["T_total"].astype(float).sum())),
            "discarded_fraction": d["discarded_fraction"].astype(float).mean(),
            "correct_per_hour": 3600 * d["correct"].sum() / max(1e-9, d["T_total"].astype(float).sum()),
            "valid_format_pct": 100 * d["valid_format"].mean(),
            "vram_peak_gib_nvml": (d["nvml_peak_used_bytes"].astype(float).max() or 0) / 2**30,
            "gpu": ",".join(sorted(set(map(str, d.gpu_name.dropna())))),
        })
    return pd.DataFrame(rows)


def jfinal_diagnostics(runs: dict, df: pd.DataFrame) -> dict:
    """oracle@4, expected uniform, majority vote, selector gap, conditional correct choice."""
    gold = runs["gold"]
    cands = [c for c in runs["candidates"] if c.get("condition") == "JFINAL"]
    if not cands:
        return {}
    by = {}
    for c in cands:
        by.setdefault((c["problem_id"], c["seed"]), []).append(c)
    rows = []
    chosen_ok = {(r.problem_id, r.seed): r.correct for r in df[df.condition == "JFINAL"].itertuples()}
    for (pid, seed), cs in by.items():
        cs = sorted(cs, key=lambda x: x["branch"])
        g = gold[pid]
        grades = [grade_text(c["text"], g) for c in cs]
        ci = [int(x["correct"]) for x in grades]
        votes = {}
        for b, x in enumerate(grades):
            if x["valid_format"]:
                votes.setdefault(x["answer_normalized"], []).append(b)
        if votes:
            best = max(votes.items(), key=lambda kv: (len(kv[1]), -min(kv[1])))
            maj = int(best[0] == g["gold_answer"])
        else:
            maj = 0
        ch = int(bool(chosen_ok.get((pid, seed), False)))
        rows.append({"problem_id": pid, "seed": seed, "uniform_expected": sum(ci) / len(ci),
                     "oracle4": max(ci), "majority": maj, "chosen": ch, "selector_gap": max(ci) - ch,
                     "n_correct_candidates": sum(ci), "n_unique_answers": len(votes)})
    t = pd.DataFrame(rows)
    has = t[t.oracle4 == 1]
    return {"n": len(t), "uniform_expected_pct": 100 * t.uniform_expected.mean(),
            "oracle4_pct": 100 * t.oracle4.mean(), "majority_pct": 100 * t.majority.mean(),
            "chosen_pct": 100 * t.chosen.mean(), "selector_gap_pp": 100 * t.selector_gap.mean(),
            "correct_choice_given_any_correct_pct": (100 * has.chosen.mean()) if len(has) else "N/A",
            "rows": rows}


def decision_stats(runs: dict) -> dict:
    out = {}
    decs = pd.DataFrame(runs["decisions"])
    if decs.empty:
        return out
    for c, d in decs.groupby("condition"):
        it = d["input_tokens"].astype(float)
        probs = d["probabilities"].dropna()
        conf = probs.apply(max) if len(probs) else pd.Series(dtype=float)
        out[c] = {"calls": len(d), "input_tokens_mean": it.mean(), "input_tokens_p95": it.quantile(0.95),
                  "share_over_2048": float((it > 2048).mean()),
                  "bands": {b: int(((it > lo) & (it <= hi)).sum()) for b, (lo, hi) in
                            {"<=512": (0, 512), "512-2048": (512, 2048), "2048-4096": (2048, 4096),
                             "4096-8192": (4096, 8192), ">8192": (8192, 1e9)}.items()},
                  "confidence_mean": float(conf.mean()) if len(conf) else None,
                  "winner_position_counts": d["winner_position"].value_counts().to_dict(),
                  "exact_ties": int(d.get("exact_tie", pd.Series(dtype=bool)).fillna(False).sum()),
                  "duplicate_rounds_share": float((d["n_unique_candidates"] < 4).mean()),
                  "status_counts": d["status"].value_counts().to_dict()}
    return out


def step_stats(runs: dict) -> dict:
    c = pd.DataFrame([x for x in runs["candidates"] if x.get("condition") == "JSTEP"])
    if c.empty:
        return {}
    lens = c["n_tokens_delivered"].astype(float)
    return {"n_candidates": len(c), "empty_steps": int(c.get("empty_step", False).sum()),
            "forced_boundaries": int(c.get("forced_boundary", False).sum()),
            "boundary_overshoots": int(c.get("boundary_overshoot", False).sum()),
            "step_len_mean": lens.mean(), "step_len_p50": lens.median(), "step_len_p95": lens.quantile(0.95),
            "step_len_hist": np.histogram(lens, bins=[0, 1, 2, 4, 8, 16, 32, 64, 128, 129])[0].tolist()}


def retokenize(df: pd.DataFrame, g_tokenizer=None) -> pd.DataFrame:
    """Offline common reference: final answers retokenized with G and character counts."""
    df = df.copy()
    if g_tokenizer is not None:
        df["answer_tokens_G"] = [len(g_tokenizer.encode(x or "", add_special_tokens=False))
                                 for x in df.get("public_output", [""] * len(df))]
    return df


def latency_study(results_root: str, n_boot: int = 10000) -> dict:
    """Confirmatory latency from notebook 08 (single VM, counterbalanced)."""
    rows = []
    for p in Path(results_root).rglob("latency_metrics.jsonl"):
        rows += read_jsonl(p)
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    d = d[d.status.notna()]
    out = {"n": len(d), "by_condition": {}, "speedup_vs_B13": {}}
    for c, g in d.groupby("condition"):
        out["by_condition"][c] = {"p50": float(g.T_total.median()), "p95": float(g.T_total.quantile(0.95)),
                                  "n": len(g)}
    if "B13" in set(d.condition):
        pb = d[d.condition == "B13"].groupby("item_id").T_total.median()
        rng = np.random.default_rng(271828)
        for c in PRIMARY + ["G_SINGLE"]:
            if c not in set(d.condition):
                continue
            pa = d[d.condition == c].groupby("item_id").T_total.median()
            common = sorted(set(pa.index) & set(pb.index))
            if not common:
                continue
            a, b = pa.loc[common].values, pb.loc[common].values
            bs = []
            for _ in range(n_boot):
                ix = rng.integers(0, len(common), len(common))
                bs.append(np.median(b[ix]) / np.median(a[ix]))
            out["speedup_vs_B13"][c] = {"estimate": float(np.median(b) / np.median(a)),
                                        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                        "ci99_1667": [float(np.percentile(bs, 0.41665)),
                                                      float(np.percentile(bs, 99.58335))],
                                        "n_items": len(common)}
    return out


# ----------------------------------------------------------------------- figures & report
def figures(df: pd.DataFrame, table: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    if not table.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ci = np.array([x if isinstance(x, list) else [np.nan, np.nan] for x in table.accuracy_ci95])
        acc = table.accuracy_pct.values
        ax.bar(table.condition, acc, yerr=[acc - ci[:, 0], ci[:, 1] - acc], capsize=4)
        ax.set_ylabel("accuracy % (95% CI)")
        ax.set_title("Accuracy by condition")
        f = out_dir / "accuracy.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
    if "T_total" in df and df.T_total.notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        conds = [c for c in ORDER if c in set(df.condition)]
        ax.boxplot([df[df.condition == c].T_total.dropna().astype(float) for c in conds], labels=conds)
        ax.set_yscale("log")
        ax.set_ylabel("T_total (s, log)")
        ax.set_title("Latency (descriptive, per-condition VMs)")
        f = out_dir / "latency.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
        fig, ax = plt.subplots(figsize=(8, 4))
        ct = pd.crosstab(df.condition, df.category)
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Outcome categories")
        f = out_dir / "categories.png"
        fig.tight_layout()
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
    return files


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(out_dir: Path, table: pd.DataFrame, comps: list, jfd: dict, dstats: dict, sstats: dict,
                 lat: dict, cats: pd.DataFrame, failures: list, meta: dict) -> str:
    L = ["# JEV-LLM · informe de resultados", "",
         f"Generado: {meta.get('generated_utc')} · run_tag: `{meta.get('run_tag')}` · "
         f"problemas con gold: {meta.get('n_gold')}", "",
         "> Latencias de las tablas 1-2 provienen de VMs distintas por condición: **descriptivas**. "
         "La comparación confirmatoria de latencia es la sección 5 (notebook 08, misma VM).", "",
         "## 1. Tabla final (§16)", ""]
    cols = ["condition", "n_problems", "accuracy_pct", "accuracy_ci95", "diff_vs_B13_pp", "diff_ci99_1667",
            "latency_p50_s", "latency_p95_s", "speedup_vs_B13_descr", "cand_tokens_mean",
            "accepted_tokens_mean", "J_input_tokens_mean", "vram_peak_gib_nvml", "gpu"]
    if not table.empty:
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for r in table.itertuples():
            L.append("| " + " | ".join(_fmt(getattr(r, c)) for c in cols) + " |")
    L += ["", "## 2. Comparaciones principales vs B13 (bootstrap pareado estratificado)", ""]
    for c in comps:
        if c:
            L.append(f"- **{c['method']}**: Δexactitud = {_fmt(c['acc_diff_pp'])} pp "
                     f"(IC95 {_fmt(c['acc_diff_ci95'])}, IC99.17 {_fmt(c['acc_diff_ci99_1667'])}); "
                     f"speedup descriptivo = {_fmt(c['speedup'])} (IC99.17 {_fmt(c['speedup_ci99_1667'])}); "
                     f"n={c['n_problems']}")
    L += ["", "Umbrales preestablecidos: +5 pp y speedup ≥ 1.10. «Más preciso y más rápido» exige que los "
          "IC ajustados excluyan 0 y 1 respectivamente.", "", "## 3. Diagnóstico JFINAL (§12.3)", ""]
    if jfd:
        for k in ("n", "chosen_pct", "uniform_expected_pct", "majority_pct", "oracle4_pct", "selector_gap_pp",
                  "correct_choice_given_any_correct_pct"):
            L.append(f"- {k}: {_fmt(jfd.get(k))}")
    L += ["", "## 4. Selector y pasos", "", "```json", json.dumps({"decisions": dstats, "jstep": sstats},
                                                                  indent=1, default=str)[:6000], "```",
          "", "## 5. Latencia confirmatoria (notebook 08)", "", "```json",
          json.dumps(lat, indent=1, default=str)[:4000] if lat else "(sin datos del notebook 08)", "```",
          "", "## 6. Categorías de resultado", ""]
    if not cats.empty:
        cols_c = list(cats.columns)
        L.append("| condition | " + " | ".join(map(str, cols_c)) + " |")
        L.append("|" + "---|" * (len(cols_c) + 1))
        for idx, row in cats.iterrows():
            L.append(f"| {idx} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    L += ["", f"## 7. Fallos registrados: {len(failures)}", ""]
    for f in failures[:30]:
        L.append(f"- {f.get('condition')} {f.get('problem_id')} [{f.get('kind')}]: {str(f.get('error'))[:200]}")
    L += ["", "## 8. Limitaciones obligatorias (§13 + enmienda v3.1)", "",
          "- Perfil SHARED4-VLLM-GRAPH (una copia de G); 4·P_G+P_J es nominal; sin R4 medido salvo chequeo de memoria.",
          "- Pesos desplegados próximos pero no iguales; cuatro réplicas idénticas no equivalen a un modelo mayor.",
          "- Entrenamiento distinto de B y J; baseline expandido por terceros; modo sin pensamiento.",
          "- Dataset muestreado de benchmarks públicos (posible contaminación); dificultad heurística; revisión humana pendiente.",
          "- Latencias entre VMs no intercambiables; ejecución con CUDA graphs en vLLM; muestreo no invariante al batch.",
          "- Corpus pequeño; p95 inestable; ninguna conclusión general sobre «todos los Jev»."]
    text = "\n".join(L)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def validate_v2(runs: dict, gold_path: str) -> dict:
    """Reject mixed, partial or duplicated test runs before any hypothesis grading."""
    gold_path = Path(gold_path)

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    gold_rows = read_jsonl(gold_path)
    require(len(gold_rows) == len(runs["gold"]) > 0, "Duplicate or empty v2 gold IDs")
    require(gold_path.name == "test_gold.jsonl", "v2 confirmatory analysis requires test gold, not pilot/smoke")
    seals = dict(line.split(None, 1)[::-1] for line in
                 (gold_path.parent / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line.strip())
    require(seals.get(gold_path.name) == sha256_file(gold_path), "Frozen test gold hash mismatch")
    inputs = read_jsonl(gold_path.with_name("test_inputs.jsonl"))
    require(len(inputs) == len(runs["gold"]) and {r["id"] for r in inputs} == set(runs["gold"]),
            "Test inputs/gold ID mismatch")
    labels, signatures, conditions, immutable = {}, [], [], {}
    expected = {(pid, 17) for pid in runs["gold"]}
    for name, man in runs["manifests"].items():
        cfg = man["config"]
        require(cfg.get("protocol_version") == "2", "Cannot mix v2 and historical runs")
        cond = man["condition"]
        conditions.append(cond)
        require(cond in ORDER and cfg["params"].get("CONDITION") == cond, "v2 condition mismatch")
        require(cfg["params"].get("SPLIT") == "test", "Cannot mix test with pilot/smoke")
        require(str(cfg["params"].get("SEEDS")).replace(" ", "") == "17", "v2 requires seed 17 only")
        require(cfg.get("profile") == "1COPY-G4-VLLM-GRAPH", "v2 one-copy profile mismatch")
        require(man.get("config_hash") == config_hash(cfg), "Manifest config hash mismatch")
        require(name == man.get("run_name") == f"{cond}_{cfg['params']['RUN_TAG']}_{man['config_hash'][:8]}",
                "Run identity mismatch")
        require(man.get("finished_utc") and man.get("stages") and
                all(s.get("ok") is True for s in man["stages"].values()), "Incomplete/failed v2 run")
        required = BLOCKING_V2 | ({"selector_native_reference", "selector_equivalence"} if cond in PRIMARY else set())
        require(required <= set(man.get("preflight", {})) and all(man["preflight"][k].get("blocking") is True
                and man["preflight"][k].get("ok") is True for k in required) and
                all(not p.get("blocking") or p.get("ok") is True
                for p in man["preflight"].values()), "Failed/missing blocking preflight")
        roles = {"B"} if cond.startswith("B13") else ({"G"} if cond == "G_SINGLE" else {"G", "J"})
        locks = man.get("model_lock", {})
        require(roles <= set(locks) and all(l.get("ok") is True for l in locks.values()), "Model lock failed")
        require(bool(man.get("condition_label")), "Missing descriptive condition_label")
        labels[cond] = man["condition_label"]
        require(cfg.get("models") and cfg.get("prompt_sha256") and cfg.get("data_sha256") and
                cfg.get("experiment_config_sha256") and cfg.get("code_version"), "Missing v2 provenance")
        config_file = Path(cfg["params"].get("CONFIG_FILE", ""))
        require(bool(str(config_file) != ".") and not config_file.is_absolute() and ".." not in config_file.parts,
                "Missing/unsafe frozen config path")
        frozen_paths = [p / config_file for p in gold_path.resolve().parents if (p / config_file).is_file()]
        require(bool(frozen_paths), "Frozen v2 config file unavailable")
        frozen = read_json(frozen_paths[0])
        require(sha256_file(frozen_paths[0]) == cfg["experiment_config_sha256"] and
                cfg["models"] == frozen["models"] and cfg["profile"] == frozen["implementation_profile"] and
                cfg["jevk5_runtime"] == frozen["jevk5_runtime"] and frozen["protocol_version"] == "2" and
                cfg["code_version"] == frozen.get("code_version", "0.2.0"), "Frozen models/config/runtime mismatch")
        signatures.append({**cfg, "params": {k: v for k, v in cfg["params"].items() if k != "CONDITION"}})
        for file, digest in cfg["data_sha256"].items():
            require(Path(file).name == file and (gold_path.parent / file).is_file() and
                    sha256_file(gold_path.parent / file) == digest == seals.get(file), "Frozen data hash mismatch: " + file)
        require(cfg["data_sha256"].get("test_inputs.jsonl") == sha256_file(gold_path.with_name("test_inputs.jsonl")),
                "Missing/mismatched test input hash")
        directory = runs["directories"][name]
        require(man.get("counts") and man["counts"].get("predictions") == len(expected) and
                man["counts"].get("metrics") == len(expected), "Incomplete v2 manifest counts")
        for ledger, count in man["counts"].items():
            require(len(read_jsonl(directory / (ledger + ".jsonl"))) == count, "Manifest ledger count mismatch")
        progress = read_json(directory / "progress.json")
        require(progress.get("done") == progress.get("total") == len(expected), "Incomplete final progress")
        for ledger in ("predictions", "metrics"):
            rows = [r for r in runs[ledger] if r["run_name"] == name]
            keys = [(r["problem_id"], r["seed"]) for r in rows]
            require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected,
                    "Incomplete or duplicate v2 " + ledger)
            require(all(r.get("condition") == cond and r.get("config_hash") == man["config_hash"] and
                        r.get("profile") == cfg["profile"] and r.get("t_start_utc") and r.get("t_end_utc")
                        for r in rows), "Prediction/metric provenance mismatch")
        immutable[name] = {"predictions_sha256": sha256_file(directory / "predictions.jsonl"),
                           "config_hash": man["config_hash"]}
    require(Counter(conditions) == Counter(ORDER), "v2 requires exactly one complete run per six fixed conditions")
    require(all(s == signatures[0] for s in signatures), "Models/config/data hashes differ across v2 conditions")
    require(not runs["failures"], "Execution failures present")
    allowed = {"final", "eos_invalid", "truncated", "max_rounds", "timeout", "selector_context_limit"}
    require(all(p.get("status") in allowed for p in runs["predictions"]), "Aborted/invalid execution status")
    jf_status = {(p["problem_id"], p["seed"]): p["status"] for p in runs["predictions"] if p["condition"] == "JFINAL"}
    auxiliary_counts = {}
    jf_manifest = next(m for m in runs["manifests"].values() if m["condition"] == "JFINAL")
    for ledger in ("candidates", "decisions"):
        rows = [r for r in runs[ledger] if r.get("condition") == "JFINAL"]
        counts = Counter((r["problem_id"], r["seed"]) for r in rows)
        auxiliary_counts[ledger] = counts
        require(set(counts) <= expected and all(counts[key] == (4 if ledger == "candidates" else 1) or
                (counts[key] == 0 and jf_status[key] == "timeout") for key in expected), "Incomplete JFINAL " + ledger)
        require(all(r.get("config_hash") == jf_manifest["config_hash"] and r.get("round") == 0 and
                    r.get("profile") == jf_manifest["config"]["profile"] for r in rows), "JFINAL auxiliary provenance mismatch")
    require(set(auxiliary_counts["candidates"]) == set(auxiliary_counts["decisions"]), "JFINAL auxiliary coverage differs")
    for key in expected:
        cs = [r for r in runs["candidates"] if r.get("condition") == "JFINAL" and
              (r["problem_id"], r["seed"]) == key]
        require(not cs or {r["branch"] for r in cs} == {0, 1, 2, 3}, "Duplicate JFINAL candidate branches")
    return {"labels": labels, "immutable_predictions": immutable,
            "gold_sha256": sha256_file(gold_path), "n_problems": len(expected)}


def paired_accuracy_v2(df: pd.DataFrame, a: str, b: str, n_boot: int = 10000) -> dict:
    """Fixed-ID stratified percentile bootstrap; exact two-sided McNemar on seed 17."""
    pa, pb = per_problem(df, a).sort_index(), per_problem(df, b).sort_index()
    if pa.empty or not pa.index.equals(pb.index) or not pa.domain.equals(pb.domain):
        raise ValueError("Complete identical paired IDs/domains required")
    aa, bb = pa.acc.to_numpy(), pb.acc.to_numpy()
    if not np.isin(aa, [0, 1]).all() or not np.isin(bb, [0, 1]).all():
        raise ValueError("Exact McNemar requires one binary observation per problem")
    strata = [np.flatnonzero(pa.domain.to_numpy() == dom) for dom in sorted(set(pa.domain))]
    rng = np.random.default_rng(271828)
    differences = 100 * (aa - bb)
    bs = np.array([differences[np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])].mean()
                   for _ in range(n_boot)])
    wins, losses = int(((aa == 1) & (bb == 0)).sum()), int(((aa == 0) & (bb == 1)).sum())
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    return {"method": a, "baseline": b, "n_problems": len(pa), "acc_diff_pp": float(differences.mean()),
            "acc_diff_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
            "acc_diff_ci99_bonferroni": np.percentile(bs, [0.5, 99.5]).tolist(),
            "wins": wins, "losses": losses, "mcnemar_p_two_sided_exact": p,
            "n_boot": n_boot, "seed": 271828, "strata": {d: int((pa.domain == d).sum()) for d in sorted(set(pa.domain))}}


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Step-down Holm: running maximum of (m - rank + 1) * sorted p."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def independent_grade_v2(pred: dict, gold: dict) -> dict:
    """Audit implementation independent of grade_text/extract_final/parse_strict_answer."""
    finals = []
    for line in (pred.get("public_output") or "").split("\n"):
        line = line.strip()
        if line[:6] == "FINAL:":
            finals.append(line[6:])
    raw = finals[-1] if finals else None
    value, status = None, "no_final"
    if len(finals) > 1:
        status = "multiple_final"
    elif finals:
        number = raw.translate(str.maketrans({chr(c): "-" for c in (0x2212, 0x2013, 0x2014, 0xfe63, 0xff0d)})).strip()
        integer = r"[+-]?\d+"
        decimal = r"[+-]?(?:\d+\.\d*|\.\d+)"
        valid = re.fullmatch(integer + "|" + decimal + "|" + integer + r"\s*/\s*" + integer, number)
        if valid:
            try:
                value = Fraction(number.replace(" ", "").replace("\t", "")) if "/" not in number else Fraction(
                    *[int(part.strip()) for part in number.split("/")])
            except (ValueError, ZeroDivisionError):
                value = None
        status = "ok" if value is not None else "invalid_format"
    return {"format_status": status, "answer_raw": raw,
            "answer_normalized": (str(value.numerator) if value.denominator == 1 else
                                  f"{value.numerator}/{value.denominator}") if value is not None else None,
            "valid_format": status == "ok", "correct": pred.get("status") == "final" and
            value == Fraction(gold["gold_numerator"], gold["gold_denominator"])}


def run_analysis_v2(runs: dict, gold_path: str, out: Path, results_root: str, n_boot: int, g_tokenizer=None) -> dict:
    from .common import utc_now

    # A failed rerun must not leave yesterday's successful H1 beside a new failed audit.
    for filename in ("analysis.json", "report.md", "graded_runs.csv", "final_table.csv", "primary_comparison.csv",
                     "secondary_comparisons.csv", "categories.csv", "accuracy_by_domain_difficulty.csv",
                     "parser_audit.json", "exploratory_last_number.csv", "exploratory_last_number.json"):
        (out / filename).unlink(missing_ok=True)
    for filename in ("accuracy.png", "latency.png", "categories.png"):
        (out / "figures" / filename).unlink(missing_ok=True)
    provenance = validate_v2(runs, gold_path)
    labels = provenance["labels"]
    df = grade_all(runs)
    # No gold-informed repair: strict grades are immutable inputs to H1 and H2-H4.
    audit_rows, exploratory = [], []
    fields = ("format_status", "answer_raw", "answer_normalized", "valid_format", "correct")
    for pred in runs["predictions"]:
        gold = runs["gold"][pred["problem_id"]]
        main, independent = grade_prediction(pred, gold), independent_grade_v2(pred, gold)
        disagreements = [k for k in fields if main[k] != independent[k]]
        audit_rows.append({"problem_id": pred["problem_id"], "condition": pred["condition"], "seed": pred["seed"],
                           "condition_label": labels[pred["condition"]], "disagreements": disagreements,
                           "main": {k: main[k] for k in fields}, "independent": independent})
        text = (pred.get("public_output") or "").translate(str.maketrans({chr(c): "-" for c in
                                                  (0x2212, 0x2013, 0x2014, 0xfe63, 0xff0d)}))
        numbers = re.findall(r"[+-]?(?:\d+\s*/\s*[+-]?\d+|\d+\.\d*|\.\d+|\d+)", text)
        try:
            last = Fraction(re.sub(r"\s+", "", numbers[-1])) if numbers else None
        except (ValueError, ZeroDivisionError):
            last = None
        exploratory.append({"problem_id": pred["problem_id"], "condition": pred["condition"], "seed": pred["seed"],
                            "condition_label": labels[pred["condition"]], "last_number": str(last) if last is not None else None,
                            "correct_exploratory": last == Fraction(gold["gold_numerator"], gold["gold_denominator"])})
    audit = {"n_predictions": len(audit_rows), "n_disagreements": sum(bool(r["disagreements"]) for r in audit_rows),
             "fields_checked": list(fields), "immutable_predictions": provenance["immutable_predictions"], "rows": audit_rows}
    write_json(out / "parser_audit.json", audit)
    pd.DataFrame(exploratory).to_csv(out / "exploratory_last_number.csv", index=False)
    write_json(out / "exploratory_last_number.json", {"scope": "Exploratory only; never used for H1/H2-H4",
               "rule": "Last numeric substring, exact rational; ignores termination/format status", "by_condition": {
                   c: {"condition_label": labels[c], "accuracy_pct": 100 * np.mean([r["correct_exploratory"] for r in
                       exploratory if r["condition"] == c])} for c in ORDER}})
    if audit["n_disagreements"]:
        raise ValueError("Independent parser disagrees; see parser_audit.json; no confirmatory report emitted")
    for name, evidence in provenance["immutable_predictions"].items():
        if sha256_file(runs["directories"][name] / "predictions.jsonl") != evidence["predictions_sha256"]:
            raise ValueError("Predictions changed during analysis")
    df = df.merge(pd.DataFrame(runs["predictions"])[["problem_id", "seed", "condition", "public_output"]],
                  on=["problem_id", "seed", "condition"], validate="one_to_one")
    df = retokenize(df, g_tokenizer)
    df["condition_label"] = df.condition.map(labels)
    df["public_output_chars"] = df.public_output.fillna("").str.len()
    df.drop(columns="public_output").to_csv(out / "graded_runs.csv", index=False)
    primary = paired_accuracy_v2(df, "JFINAL", "B13", n_boot)
    primary.pop("acc_diff_ci99_bonferroni")
    primary.update({"hypothesis": "H1", "more_accurate": primary["acc_diff_ci95"][0] > 0,
                    "practical_point_gt5pp": primary["acc_diff_pp"] > 5.0})
    secondary = [paired_accuracy_v2(df, a, b, n_boot) for a, b in SECONDARY_V2]
    adjusted = holm_adjusted_pvalues([r["mcnemar_p_two_sided_exact"] for r in secondary])
    for row, p in zip(secondary, adjusted):
        row.update({"holm_adjusted_p": p, "holm_reject_two_sided_05": p <= 0.05,
                    "more_accurate_holm": p <= 0.05 and row["acc_diff_pp"] > 0,
                    "simultaneous_ci_level": 1 - .05 / 5})
    sensitivity = paired_accuracy_v2(df, "JFINAL", "B13_GREEDY", n_boot)
    sensitivity.pop("acc_diff_ci99_bonferroni")
    for row in [primary, *secondary, sensitivity]:
        row.update({"method_label": labels[row["method"]], "baseline_label": labels[row["baseline"]]})
    table_rows = []
    for cond in ORDER:
        d = df[df.condition == cond]
        times = d.T_total.astype(float).dropna()
        table_rows.append({"condition": cond, "condition_label": labels[cond], "n_problems": len(d),
                          "accuracy_pct": 100 * float(d.correct.mean()), "accuracy_ci95": acc_ci(df, cond, n_boot),
                          "valid_format_pct": 100 * float(d.valid_format.mean()),
                          "latency_p50_s_descriptive": float(times.median()) if len(times) else None,
                          "latency_p95_s_descriptive": float(times.quantile(.95)) if len(times) else None,
                          "candidate_tokens_mean": float(d.candidate_tokens_total.astype(float).mean()),
                          "accepted_tokens_mean": float(d.accepted_tokens.astype(float).mean()),
                          "selector_input_tokens_mean": float(d.selector_input_tokens.astype(float).mean()),
                          "processed_prompt_tokens_mean_compute_proxy": float(d.processed_prompt_tokens.astype(float).mean()),
                          "system_tokens_mean_compute_proxy": float((d.candidate_tokens_total.astype(float) +
                              d.selector_input_tokens.astype(float) + d.processed_prompt_tokens.astype(float)).mean()),
                          "vram_peak_gib_nvml": float(d.nvml_peak_used_bytes.astype(float).max() / 2**30)})
    table = pd.DataFrame(table_rows)
    table.to_csv(out / "final_table.csv", index=False)
    pd.DataFrame([primary]).to_csv(out / "primary_comparison.csv", index=False)
    pd.DataFrame(secondary).to_csv(out / "secondary_comparisons.csv", index=False)
    cats = pd.crosstab(df.condition_label, df.category)
    cats.to_csv(out / "categories.csv")
    df.groupby(["condition", "condition_label", "domain", "difficulty"]).correct.mean().unstack("difficulty").to_csv(
        out / "accuracy_by_domain_difficulty.csv")
    jfd = jfinal_diagnostics(runs, df)
    jfd["oracle_capture_ratio"] = jfd["chosen_pct"] / jfd["oracle4_pct"] if jfd.get("oracle4_pct") else None
    jfd["n_prediction_cases"] = int((df.condition == "JFINAL").sum())
    jfd["n_cases_without_candidate_logs"] = jfd["n_prediction_cases"] - jfd.get("n", 0)
    jfd["scope"] = "Candidate diagnostics use complete logged candidate sets only; timeouts remain incorrect in full H1 denominator"
    jfd["condition_label"] = labels["JFINAL"]
    dstats, sstats = decision_stats(runs), step_stats(runs)
    for cond, row in dstats.items():
        row["condition_label"] = labels[cond]
    # Keep descriptive labels in v2 plots without changing the legacy plotter.
    display_df = df.assign(condition=df.condition_label)
    display_table = table.assign(condition=table.condition_label)
    figs = figures_v2(display_df, display_table, out / "figures")
    latency = latency_descriptive_v2(results_root, next(iter(runs["manifests"].values()))["config"], labels)
    meta = {"generated_utc": utc_now(), "protocol_version": "2", "code_version": next(iter(runs["manifests"].values()))["config"]["code_version"],
            "run_tag": next(iter(runs["manifests"].values()))["config"]["params"]["RUN_TAG"],
            "n_gold": len(runs["gold"]), "n_predictions": len(df), "runs": list(runs["manifests"]), **provenance}
    interval_note = ("Fixed five secondary contrasts; no post-hoc best hybrid. Decisions/p-values: Holm step-down "
                     "on two-sided paired exact McNemar tests. Intervals: conservative simultaneous Bonferroni "
                     "99% (=1-.05/5) stratified paired percentile bootstrap intervals, NOT Holm rank-dependent "
                     "intervals; 95% intervals are descriptive. Bootstrap coverage is approximate, not an exact coverage guarantee.")
    lines = ["# JEV-LLM v2 offline analysis", "", f"Test problems: {len(runs['gold'])}; immutable predictions: {len(df)}; seed 17.",
             "", "## Conditions", "", "| Code | Description | Accuracy % | Descriptive 95% CI | Latency median s (descriptive) |",
             "|---|---|---|---|---|"]
    for row in table_rows:
        lines.append(f"| {row['condition']} | {row['condition_label']} | {_fmt(row['accuracy_pct'])} | "
                     f"{_fmt(row['accuracy_ci95'])} | {_fmt(row['latency_p50_s_descriptive'])} |")
    lines += ["", "## Fixed Primary H1", "", f"{primary['method_label']} vs {primary['baseline_label']}: "
              f"difference {_fmt(primary['acc_diff_pp'])} pp; paired stratified bootstrap 95% CI {_fmt(primary['acc_diff_ci95'])}.",
              f"More accurate (lower CI > 0): {primary['more_accurate']}. Practical point estimate >5 pp: {primary['practical_point_gt5pp']}.",
              "Accuracy-only H1: no latency joint-superiority claim.", "", "## Fixed Secondary Contrasts", "", interval_note, "",
              "| Method | Baseline | Difference pp | Descriptive 95% CI | Simultaneous Bonferroni 99% CI | Exact p | Holm adjusted p | Reject |",
              "|---|---|---|---|---|---|---|---|"]
    for row in secondary:
        lines.append("| " + " | ".join(_fmt(row[k], 6 if "p" in k and k.endswith("p") else 2) for k in
                     ("method_label", "baseline_label", "acc_diff_pp", "acc_diff_ci95", "acc_diff_ci99_bonferroni",
                      "mcnemar_p_two_sided_exact", "holm_adjusted_p", "holm_reject_two_sided_05")) + " |")
    lines += ["", "## Greedy Sensitivity", "", f"JFINAL vs {sensitivity['baseline_label']}: {_fmt(sensitivity['acc_diff_pp'])} pp; "
              f"95% CI {_fmt(sensitivity['acc_diff_ci95'])}. Never select the better baseline per problem.",
              "", "## Candidate Diagnostics", "", "```json", json.dumps({k: v for k, v in jfd.items() if k != "rows"}, indent=2), "```",
              "", "## Audit And Exploration", "", f"Independent parser checked all {audit['n_predictions']} predictions: {audit['n_disagreements']} disagreements.",
              "Canonical format_status, answer_raw, answer_normalized, valid_format and rational correctness audited separately.",
              "Permissive last-number analysis is exploratory only in exploratory_last_number.csv/json; strict H1 is unaffected.",
              "", "## Same-GPU Latency Study", "", "Secondary descriptive only; never gates H1.", "```json",
              json.dumps(latency, indent=2), "```",
              "", "## Limitations", "", "- Adaptively motivated generator change after the historical test was opened; new test only.",
              "- reviewed=false: no completed human or blind secondary review; not claimed as reviewed.",
              "- One physical generator copy: 13.159554560B unique hybrid parameters vs 12.413584128B baseline; not four weight copies.",
              "- Parameter matching does not match compute: token counts are only a compute proxy; hybrid processes more tokens.",
              "- system_tokens_mean_compute_proxy sums sampled candidate tokens, processed prompt tokens and selector input tokens; not FLOPs.",
              "- All latency is secondary descriptive, including the separate same-GPU study; no joint accuracy/latency superiority.",
              "- One seed, small public benchmark test, heuristic difficulty and possible contamination; not equivalence or generalization.",
              "- Distinct training and expanded third-party baseline; non-thinking CUDA graphs, batch-dependent sampling; hard-stratum fallback."]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    write_json(out / "analysis.json", {"meta": meta, "primary": primary, "secondary": secondary,
               "comparisons": [primary, *secondary], "secondary_interval_note": interval_note,
               "sensitivity": sensitivity, "jfinal": jfd, "decisions": dstats, "jstep": sstats,
               "latency_scope": "secondary descriptive only", "latency_study": latency,
               "parser_audit": {k: v for k, v in audit.items() if k != "rows"}, "figures": figs})
    return {"table": table, "comparisons": [primary, *secondary], "primary": primary, "secondary": secondary,
            "jfinal": jfd, "report": report, "df": df, "parser_audit": audit}


def latency_descriptive_v2(results_root: str, config: dict, labels: dict) -> dict:
    rows = []
    for path in Path(results_root).rglob("latency_metrics.jsonl"):
        man = read_json(path.parent / "manifest.json")
        cfg = man["config"]
        if cfg["params"].get("RUN_TAG") != config["params"]["RUN_TAG"]:
            continue
        if any(cfg.get(k) != config.get(k) for k in ("protocol_version", "models", "profile", "data_sha256",
                                                    "prompt_sha256", "code_version", "experiment_config_sha256")):
            raise ValueError("Latency study provenance differs from v2 test")
        for key in ("TIMEOUT_S", "N_CANDIDATES", "MAX_OUTPUT_TOKENS", "CANDIDATE_BUDGET", "J64_BLOCK",
                    "JSTEP_STEP", "JSTEP_MAX_ROUNDS", "GB_CONTEXT", "J_MAX_INPUT", "TEMPERATURE", "TOP_P",
                    "TOP_K", "REPETITION_PENALTY"):
            if cfg["params"].get(key) != config["params"].get(key):
                raise ValueError("Latency timers/scoring configuration differs: " + key)
        swap_mode = cfg["params"].get("LATENCY_SWAP_MODE", "sleep")
        if swap_mode not in {"sleep", "reload"}:
            raise ValueError("Unknown latency swap mode")
        gpus = man.get("environment", {}).get("gpus", [])
        if len(gpus) != 1 or "A100" not in gpus[0].get("name", "") or int(gpus[0].get("memory.total", 0)) != 40960:
            raise ValueError("v2 latency requires one same A100 40GB")
        if man.get("config_hash") != config_hash(cfg) or not man.get("finished_utc"):
            raise ValueError("Incomplete/unverified v2 latency study")
        records = read_jsonl(path)
        if rows or len({(r["item_id"], r["condition"], r["seed"]) for r in records}) != len(records):
            raise ValueError("Duplicate v2 latency study/rows")
        if any(r.get("config_hash") != man["config_hash"] for r in records):
            raise ValueError("Latency row config mismatch")
        progress = read_json(path.parent / "progress.json")
        plan = read_json(path.parent / "latency_plan.json")
        items = [r["item_id"] for r in plan["items"]]
        expected = {(item, cond, 17) for item in items for cond in ORDER}
        keys = {(r["item_id"], r["condition"], r["seed"]) for r in records}
        if len(items) != len(set(items)) or keys != expected or progress.get("done") != len(expected) or \
                progress.get("total") != len(expected) or man.get("counts", {}).get("latency_metrics") != len(records):
            raise ValueError("Incomplete v2 latency plan/progress/coverage")
        if not man.get("stages") or any(s.get("ok") is not True for s in man["stages"].values()) or \
                read_jsonl(path.parent / "failures.jsonl"):
            raise ValueError("Failed v2 latency execution")
        if any(r.get("status") not in {"final", "eos_invalid", "truncated", "max_rounds", "timeout",
                                      "selector_context_limit"} for r in records):
            raise ValueError("Aborted/invalid latency status")
        rows.extend(records)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    return {"scope": "same-GPU secondary descriptive", "hardware": "same A100 40GB", "swap_mode": swap_mode,
            "swap_timing": "B versus G+J phase swaps outside measured case timers",
            "swap_description": ("Unload/reload B versus G+J on the same A100 40GB; reload swaps outside the chronometer"
                                 if swap_mode == "reload" else "Sleep/wake B versus G+J outside the chronometer"),
            "scoring": "Same six conditions, strict rational evaluator and timer/budget rules unchanged",
            "n": len(data), "by_condition": {
        c: {"condition_label": labels[c], "n": len(d), "latency_p50_s": float(d.T_total.median()),
            "latency_p95_s": float(d.T_total.quantile(.95)), "statuses": d.status.value_counts().to_dict()}
        for c, d in data.groupby("condition")}}


def figures_v2(df: pd.DataFrame, table: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = []
    for kind in ("accuracy", "latency", "categories"):
        fig, ax = plt.subplots(figsize=(12, 7))
        if kind == "accuracy":
            ci = np.asarray(table.accuracy_ci95.tolist())
            ax.barh(table.condition, table.accuracy_pct, xerr=[np.maximum(0, table.accuracy_pct - ci[:, 0]),
                                                             np.maximum(0, ci[:, 1] - table.accuracy_pct)], capsize=4)
            ax.set_xlabel("Accuracy % (descriptive 95% CI)")
        elif kind == "latency":
            conditions = table.condition.tolist()
            ax.boxplot([df.loc[df.condition == c, "T_total"].astype(float).dropna() for c in conditions],
                       labels=conditions, vert=False)
            ax.set_xlabel("Latency seconds (secondary descriptive)")
        else:
            pd.crosstab(df.condition, df.category).plot.barh(stacked=True, ax=ax)
            ax.set_xlabel("Outcome counts")
        fig.tight_layout()
        path = out / (kind + ".png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(str(path))
    return files


def run_analysis(results_root: str, gold_path: str, out_dir: str, run_tag: str | None = None,
                 n_boot: int = 10000, g_tokenizer=None) -> dict:
    from .common import utc_now

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results_root, gold_path, run_tag)
    if any(m.get("config", {}).get("protocol_version") == "2" for m in runs["manifests"].values()):
        return run_analysis_v2(runs, gold_path, out, results_root, n_boot, g_tokenizer)
    df = grade_all(runs)
    if df.empty:
        raise SystemExit("no graded predictions found")
    df = retokenize(df.merge(pd.DataFrame(runs["predictions"])[["problem_id", "seed", "condition",
                                                                 "public_output"]],
                             on=["problem_id", "seed", "condition"], how="left"), g_tokenizer)
    df["public_output_chars"] = df["public_output"].fillna("").str.len()
    df.drop(columns=["public_output"]).to_csv(out / "graded_runs.csv", index=False)
    table = summary_table(df, n_boot)
    table.to_csv(out / "final_table.csv", index=False)
    comps = [paired_bootstrap(df, c, "B13", n_boot) for c in PRIMARY + ["G_SINGLE", "B13_GREEDY"]
             if c in set(df.condition) and "B13" in set(df.condition)]
    jfd = jfinal_diagnostics(runs, df)
    dstats, sstats = decision_stats(runs), step_stats(runs)
    lat = latency_study(results_root, n_boot)
    cats = pd.crosstab(df.condition, df.category)
    cats.to_csv(out / "categories.csv")
    by_dd = df.groupby(["condition", "domain", "difficulty"]).correct.mean().unstack(["difficulty"])
    by_dd.to_csv(out / "accuracy_by_domain_difficulty.csv")
    figs = figures(df, table, out / "figures")
    meta = {"generated_utc": utc_now(), "run_tag": run_tag, "n_gold": len(runs["gold"]),
            "n_predictions": len(runs["predictions"]), "runs": list(runs["manifests"])}
    write_json(out / "analysis.json", {"meta": meta, "comparisons": comps, "jfinal": jfd,
                                       "decisions": dstats, "jstep": sstats, "latency_study": lat,
                                       "figures": figs})
    rep = write_report(out, table, comps, jfd, dstats, sstats, lat, cats, runs["failures"], meta)
    return {"table": table, "comparisons": comps, "jfinal": jfd, "report": rep, "df": df}
