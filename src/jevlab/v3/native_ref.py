"""Truthfully instrument the published native runtime in an isolated subprocess."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..common import config_hash, sha256_text, write_json


def main(inp: str, out: str, snapshot: str) -> None:
    payload = json.loads(Path(inp).read_text(encoding="utf-8"))
    identity = payload["identity"]
    result = {"identity": identity, "reference_id": config_hash(identity), "items": [],
              "native_execution_id": payload["native_execution_id"],
              "snapshot": str(Path(snapshot).resolve()),
              **{k: payload[k] for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles",
                                         "cleanup_confirmed", "outside_T_total")}}
    write_json(out, result)
    try:
        import torch
        import jevk5
        from jevk5 import JevK5
        from jevk5.prompt import decision_options, prompt_text

        distribution = importlib.metadata.distribution("jevk5")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        result.update(runtime_version=distribution.version,
                      runtime_module=str(Path(jevk5.__file__).resolve()),
                      runtime_direct_url=direct,
                      runtime_commit=direct.get("vcs_info", {}).get("commit_id"),
                      torch=str(torch.__version__), cuda_version=torch.version.cuda)
        # Capture real diagnostics, including failures; never claim CPU isolation.
        try:
            diagnostics = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            result["cuda_hardware"] = {"returncode": diagnostics.returncode,
                                       "stdout": diagnostics.stdout, "stderr": diagnostics.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["cuda_hardware"] = {"error": str(exc),
                                       "stdout": getattr(exc, "stdout", None),
                                       "stderr": getattr(exc, "stderr", None)}
        write_json(out, result)
        if result["snapshot"] != identity["snapshot"]:
            raise ValueError("native snapshot does not match input identity")
        if result["runtime_commit"] != identity["runtime"]["commit"]:
            raise ValueError("installed native runtime commit does not match frozen configuration")
        started = time.perf_counter()
        result["constructor_kwargs"] = {"device": "cuda", "dtype": "torch.bfloat16", "graphs": False}
        model = JevK5(snapshot, device="cuda", dtype=torch.bfloat16, graphs=False)
        result.update(load_s=time.perf_counter() - started, temperature=model.temperature)
        actual = getattr(model, "model", None)
        device, dtype = getattr(actual, "device", None), getattr(actual, "dtype", None)
        if actual is not None and (device is None or dtype is None):
            parameter = next(actual.parameters(), None)
            if parameter is not None:
                device, dtype = parameter.device, parameter.dtype
        device = device if device is not None else getattr(model, "device", None)
        dtype = dtype if dtype is not None else getattr(model, "dtype", None)
        result.update(model_device=str(device) if device is not None else None,
                      model_dtype=str(dtype) if dtype is not None else None,
                      model_device_map=getattr(actual, "hf_device_map", None))
        write_json(out, result)
        if device is None or dtype is None:
            raise ValueError("runtime does not expose actual model device/dtype")
        for index, item in enumerate(payload["items"]):
            state, question = item["state"], item["question"]
            options = decision_options(question)
            texts = [text for _, text in options]
            ids = model.encode(state, question["instructions"], texts)
            rendered = prompt_text(state, question["instructions"], texts)
            ids_pt = model.tok.encode(rendered, add_special_tokens=False)
            started = time.perf_counter()
            logits = model.letter_logits(ids, len(texts)).tolist()
            probs, tokens = model.probabilities(state, question)
            row = {"index": index, "ids": ids, "n_ids": len(ids),
                   "ids_sha256": sha256_text(json.dumps(ids)), "prompt_sha256": sha256_text(rendered),
                   "ids_equal_prompt_text": ids == ids_pt, "raw_logits": logits,
                   "probabilities": [probs[k] for k, _ in options], "input_tokens": tokens,
                   "seconds": time.perf_counter() - started,
                   "reference_id": result["reference_id"],
                   **{k: result[k] for k in ("phase_epoch", "phase_id", "gpu_uuid", "resident_roles")}}
            result["items"].append(row)
            write_json(out, result)
        result["peak_mem_gib"] = (torch.cuda.max_memory_allocated() / 2**30
                                  if torch.cuda.is_initialized() else None)
        write_json(out, result)
    except Exception:
        result.update(ok=False, traceback=traceback.format_exc())
        write_json(out, result)
        raise


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
