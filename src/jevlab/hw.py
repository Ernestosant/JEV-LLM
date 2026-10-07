"""Hardware/environment capture and NVML memory sampling (protocol §9.1, §10.3).

The notebook process must not create a CUDA context (vLLM engines live in subprocesses),
so GPU facts come from nvidia-smi / NVML and the compute probe runs in a subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time

log = logging.getLogger("jevlab.hw")


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def gpu_info() -> list[dict]:
    q = "name,uuid,memory.total,driver_version,compute_cap,pci.bus_id,clocks.max.sm,power.limit"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(dict(zip(q.split(","), parts)))
    return gpus


def cuda_version_smi() -> str:
    out = _run(["nvidia-smi"])
    for line in out.splitlines():
        if "CUDA Version" in line:
            return line.split("CUDA Version:")[-1].strip(" |")
    return "unknown"


def host_info() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
    except OSError:
        pass
    disk = _run(["df", "-h", "/content"]) if os.path.exists("/content") else ""
    return {"hostname": platform.node(), "cpu_model": cpu, "cpu_count": os.cpu_count(),
            "ram_total_gib": round(mem_kb / 2**20, 2), "python": sys.version,
            "platform": platform.platform(), "disk_content": disk}


def pip_freeze() -> list[str]:
    return _run([sys.executable, "-m", "pip", "freeze"], timeout=120).splitlines()


def package_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ("vllm", "torch", "transformers", "tokenizers", "huggingface-hub", "jevk5",
              "flash-linear-attention", "fla-core", "causal-conv1d", "triton", "flashinfer-python",
              "numpy", "nvidia-ml-py", "papermill"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def rss_gib() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")


class NvmlSampler:
    """Background NVML sampling of device memory (total used + per process).

    It is a *periodic* observation at `hz`; the reported peak is the max observed sample,
    not an exact peak (declared in every artifact)."""

    def __init__(self, hz: float = 10.0, out_path: str | None = None):
        self.hz, self.out_path = hz, out_path
        self.samples: list[tuple[float, int]] = []   # (perf_counter, used_bytes)
        self._stop = threading.Event()
        self._t = None
        self.ok = False
        self.proc_peak: dict[int, int] = {}
        self._lock = threading.Lock()

    def start(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nv = pynvml
            self.ok = True
        except Exception as e:  # noqa: BLE001
            log.warning("NVML unavailable: %s", e)
            return self
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        nv, h = self._nv, self._h
        period = 1.0 / self.hz
        k = 0
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                used = nv.nvmlDeviceGetMemoryInfo(h).used
                with self._lock:
                    self.samples.append((t, used))
                    if len(self.samples) > 2_000_000:
                        self.samples = self.samples[-1_000_000:]
                if k % int(self.hz) == 0:
                    for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
                        m = p.usedGpuMemory or 0
                        self.proc_peak[p.pid] = max(self.proc_peak.get(p.pid, 0), m)
            except Exception:  # noqa: BLE001
                pass
            k += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t)))

    def now_used(self) -> int | None:
        if not self.ok:
            return None
        return self._nv.nvmlDeviceGetMemoryInfo(self._h).used

    def window_peak(self, t0: float, t1: float) -> int | None:
        with self._lock:
            vals = [u for (t, u) in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None

    def processes(self) -> list[dict]:
        if not self.ok:
            return []
        out = []
        for p in self._nv.nvmlDeviceGetComputeRunningProcesses(self._h):
            out.append({"pid": p.pid, "used_bytes": p.usedGpuMemory,
                        "peak_observed_bytes": self.proc_peak.get(p.pid)})
        return out

    def stop(self):
        self._stop.set()


def run_probe(out_path: str, timeout: int = 600) -> dict:
    """Standardised compute/memory microbenchmark in a subprocess (sensitivity only)."""
    from .common import subprocess_env

    cmd = [sys.executable, "-m", "jevlab.hwprobe", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=subprocess_env())
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    with open(out_path) as f:
        return json.load(f)
