"""Build the upload bundles.

  dist/jev_llm_bundle.zip           inference: code, config, prompts, inputs, schedule (NO gold)
  dist/jev_llm_analysis_bundle.zip  analysis: the above + test/dev gold + review files
Each bundle contains BUNDLE_MANIFEST.json with the SHA-256 of every file (checked in notebooks).
"""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
COMMON = ["config/experiment.json", "prompts/generador.txt", "prompts/criterio_paso.txt",
          "prompts/criterio_final.txt", "data/test_inputs.jsonl", "data/dev_inputs.jsonl",
          "data/schedule.json", "data/dataset_manifest.json"]
GOLD = ["data/test_gold.jsonl", "data/dev_gold.jsonl", "data/blind_review_ids.json", "data/review_sheet.csv"]


def files_for(gold: bool, series: str = "v1") -> list[str]:
    src = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src" / "jevlab").glob("*.py"))
    if series in ("v2", "v2-reload"):
        common = ["config/experiment_v2.json", "config/diagnostic_08b_a100.json", *[f"prompts/{n}.txt" for n in ("generador", "criterio_paso", "criterio_final")],
                  *[f"data/v2/{s}_inputs.jsonl" for s in ("test", "pilot", "dev")], "data/v2/schedule.json"]
        if series == "v2-reload":
            common.insert(1, "config/experiment_v2_reload.json")
        optional = ("dataset_manifest.json", "SHA256SUMS") if gold else ()
        common += [f"data/v2/{n}" for n in optional if (ROOT / "data" / "v2" / n).is_file()]
        refs = [f"data/v2/{s}_gold.jsonl" for s in ("test", "pilot", "dev")]
        refs += [f"data/v2/{n}" for n in ("blind_review_ids.json", "review_sheet.csv") if (ROOT / "data" / "v2" / n).is_file()]
        return src + common + (refs if gold else [])
    return src + COMMON + (GOLD if gold else [])


def build(name: str, gold: bool, series: str = "v1") -> Path:
    dest = DIST / series if series != "v1" else DIST
    files = files_for(gold, series)
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen {series} bundle inputs: {missing}")
    if not gold:
        for f in files:
            if f.endswith("_inputs.jsonl"):
                rows = [json.loads(line) for line in (ROOT / f).read_text(encoding="utf-8").splitlines() if line.strip()]
                if any(set(r) != {"id", "problem", "language"} for r in rows):
                    raise ValueError(f"Inference input contains metadata/gold: {f}")
    dest.mkdir(parents=True, exist_ok=True)
    man = {"bundle_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "created_utc": datetime.now(timezone.utc).isoformat(), "contains_gold": gold,
            "series": series,
           "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}
    if series == "v2-reload":
        man["code_version"] = json.loads((ROOT / "config/experiment_v2_reload.json").read_text())["code_version"]
    out = dest / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(man, indent=1))
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB, {len(files)} files, gold={gold})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build upload bundles without touching another series")
    parser.add_argument("--series", choices=("v1", "v2", "v2-reload"), default="v1")
    args = parser.parse_args()
    prefix = "jev_llm_v2" if args.series != "v1" else "jev_llm"
    build(f"{prefix}_bundle.zip", False, args.series)
    build(f"{prefix}_analysis_bundle.zip", True, args.series)
    sys.exit(0)
