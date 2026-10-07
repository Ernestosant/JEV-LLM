"""Hardware calibration probe (run as a subprocess: python -m jevlab.hwprobe out.json).

Measures bf16 GEMM throughput, device-to-device copy bandwidth and a small-kernel launch
latency loop (host-bound). Used only as a sensitivity covariate for cross-VM latency,
never to normalise primary results.
"""

import json
import sys
import time


def main(out: str) -> None:
    import torch

    dev = torch.device("cuda")
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda}
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    res["bf16_gemm_tflops"] = 2 * 8192**3 * reps / dt / 1e12
    x = torch.empty(1 << 28, device=dev, dtype=torch.uint8)  # 256 MiB
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    res["d2d_copy_gbps"] = 2 * x.numel() * 50 / dt / 1e9
    s = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5000):
        s += 1
    torch.cuda.synchronize()
    res["small_kernel_us"] = (time.perf_counter() - t0) / 5000 * 1e6
    t0 = time.perf_counter()
    acc = 0
    for i in range(3_000_000):
        acc += i
    res["python_loop_ms"] = (time.perf_counter() - t0) * 1e3
    with open(out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
