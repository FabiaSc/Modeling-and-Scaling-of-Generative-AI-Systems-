# scripts/load_test_stub.py
"""
MixDQ-only load test (sweep) with p95/p99, CSV + TXT export.

Writes results under the MixDQ logs directory:
  <BASE_PATH>/queue_tests/<timestamp>/mixdq_queue_results.csv
  <BASE_PATH>/queue_tests/<timestamp>/mixdq_queue_results.txt
and you can generate plots into:
  <BASE_PATH>/queue_tests/<timestamp>/plots/*.png

BASE_PATH is taken from the --base_path argument inside BASE_QUANT (below).
Defaults to ./logs if not present.

Optional reproducibility:
  export MIXDQ_LOADTEST_SEED=123
"""

import csv
import math
import os
import random
import subprocess
import time
import concurrent.futures as cf
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


# --------- Base command (edit prompt here if you want) ---------

BASE_QUANT = [
    "python",
    "scripts/quant_txt2img.py",
    "--base_path",
    "./logs/sdxl_mixdq_eval",
    "--batch_size",
    "1",
    "--num_imgs",
    "1",
    "--fp16",
    "--prompt",
    "a corgi in sunglasses",
]

WBITS_LIST = (8, 6, 4)
BURST_CONC_LIST = (1, 2)
POISSON_LAM_LIST = (0.5, 1.0, 2.0, 4.0)
POISSON_CONC = 2

# 100 requests for everything (p95/p99)
BURST_TOTAL_REQUESTS = 100
POISSON_TOTAL_REQUESTS = 100


_SEED_ENV = os.environ.get("MIXDQ_LOADTEST_SEED")
if _SEED_ENV is not None:
    try:
        random.seed(int(_SEED_ENV))
    except Exception:
        pass


def _get_arg_value(cmd: List[str], key: str, default: str) -> str:
    try:
        i = cmd.index(key)
        return cmd[i + 1]
    except Exception:
        return default


def _resolve_log_dir() -> str:
    return _get_arg_value(BASE_QUANT, "--base_path", "./logs")


def infer_once(queue_len: int, wbits: int) -> float:
    """Wall-time latency of one quant_txt2img subprocess call (ms)."""
    cmd = list(BASE_QUANT) + ["--wbits", str(wbits), "--queue_len", str(queue_len)]
    t0 = time.perf_counter()
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return (time.perf_counter() - t0) * 1000.0


def compute_p95_p99(samples: List[float]) -> Tuple[float, float]:
    """Empirical p95/p99, ignoring NaN/inf."""
    xs = [x for x in samples if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    xs.sort()
    n = len(xs)

    def q(p: float) -> float:
        idx = int(p * (n - 1))
        return xs[idx]

    return q(0.95), q(0.99)


@dataclass
class ResultRow:
    pattern: str   # "burst" or "poisson"
    wbits: int
    policy: str    # "static" or "adaptive"
    conc: Optional[int] = None
    lam: Optional[float] = None
    p95_ms: float = float("nan")
    p99_ms: float = float("nan")


def run_batch(concurrency: int, total_requests: int, adaptive: bool, wbits: int) -> Tuple[float, float]:
    """Burst: submit as fast as possible with concurrency cap."""
    lat: List[float] = []
    futures = set()
    in_flight = 0

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        while in_flight < concurrency and total_requests > 0:
            q_for_call = in_flight if adaptive else 0
            futures.add(ex.submit(infer_once, q_for_call, wbits))
            in_flight += 1
            total_requests -= 1

        while futures:
            done, not_done = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            futures = not_done
            for f in done:
                try:
                    lat.append(float(f.result()))
                except Exception:
                    lat.append(float("nan"))
                in_flight -= 1

                if total_requests > 0:
                    q_for_call = in_flight if adaptive else 0
                    futures.add(ex.submit(infer_once, q_for_call, wbits))
                    in_flight += 1
                    total_requests -= 1

    return compute_p95_p99(lat)


def run_poisson(lam: float, total_requests: int, concurrency: int, adaptive: bool, wbits: int) -> Tuple[float, float]:
    """Poisson arrivals with rate lam (req/s), concurrency-limited."""
    lat: List[float] = []
    futures = set()
    in_flight = 0

    inter_arrivals = []
    for _ in range(total_requests):
        u = random.random()
        inter_arrivals.append(-math.log(1.0 - u) / lam)

    arrival_times = []
    t = 0.0
    for ia in inter_arrivals:
        t += ia
        arrival_times.append(t)

    t_start = time.perf_counter()
    next_req_idx = 0

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        while next_req_idx < total_requests or futures:
            elapsed = time.perf_counter() - t_start

            while (
                next_req_idx < total_requests
                and arrival_times[next_req_idx] <= elapsed
                and in_flight < concurrency
            ):
                q_for_call = in_flight if adaptive else 0
                futures.add(ex.submit(infer_once, q_for_call, wbits))
                in_flight += 1
                next_req_idx += 1

            done = {f for f in futures if f.done()}
            for f in done:
                futures.remove(f)
                try:
                    lat.append(float(f.result()))
                except Exception:
                    lat.append(float("nan"))
                in_flight -= 1

            time.sleep(0.005)

    return compute_p95_p99(lat)


def _write_csv(rows: List[ResultRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["pattern", "wbits", "policy", "conc", "lam", "p95_ms", "p99_ms"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_txt(lines: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    log_dir = _resolve_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(log_dir, "queue_tests", ts)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "mixdq_queue_results.csv")
    txt_path = os.path.join(out_dir, "mixdq_queue_results.txt")

    rows: List[ResultRow] = []
    lines: List[str] = []

    header = [
        f"MixDQ queue test sweep @ {ts}",
        f"BASE_PATH (logs dir): {log_dir}",
        f"BURST: total_requests={BURST_TOTAL_REQUESTS}, conc={BURST_CONC_LIST}, wbits={WBITS_LIST}",
        f"POISSON: total_requests={POISSON_TOTAL_REQUESTS}, conc={POISSON_CONC}, λ={POISSON_LAM_LIST}, wbits={WBITS_LIST}",
    ]
    if _SEED_ENV is not None:
        header.append(f"MIXDQ_LOADTEST_SEED={_SEED_ENV}")
    header.append("")
    lines.extend(header)

    # BURST
    for wbits in WBITS_LIST:
        for conc in BURST_CONC_LIST:
            p95_s, p99_s = run_batch(concurrency=conc, total_requests=BURST_TOTAL_REQUESTS, adaptive=False, wbits=wbits)
            p95_a, p99_a = run_batch(concurrency=conc, total_requests=BURST_TOTAL_REQUESTS, adaptive=True,  wbits=wbits)

            rows.append(ResultRow(pattern="burst", wbits=wbits, policy="static", conc=conc, p95_ms=p95_s, p99_ms=p99_s))
            rows.append(ResultRow(pattern="burst", wbits=wbits, policy="adaptive", conc=conc, p95_ms=p95_a, p99_ms=p99_a))

            line = (f"[BURST W{wbits} conc={conc}] "
                    f"STATIC p95={p95_s:.1f}ms p99={p99_s:.1f}ms  |  "
                    f"ADAPTIVE p95={p95_a:.1f}ms p99={p99_a:.1f}ms")
            print(line)
            lines.append(line)

    lines.append("")

    # POISSON
    for wbits in WBITS_LIST:
        for lam in POISSON_LAM_LIST:
            p95_s, p99_s = run_poisson(lam=lam, total_requests=POISSON_TOTAL_REQUESTS, concurrency=POISSON_CONC, adaptive=False, wbits=wbits)
            p95_a, p99_a = run_poisson(lam=lam, total_requests=POISSON_TOTAL_REQUESTS, concurrency=POISSON_CONC, adaptive=True,  wbits=wbits)

            rows.append(ResultRow(pattern="poisson", wbits=wbits, policy="static", conc=POISSON_CONC, lam=lam, p95_ms=p95_s, p99_ms=p99_s))
            rows.append(ResultRow(pattern="poisson", wbits=wbits, policy="adaptive", conc=POISSON_CONC, lam=lam, p95_ms=p95_a, p99_ms=p99_a))

            line = (f"[POISSON W{wbits} λ={lam:.1f} req/s] "
                    f"STATIC p95={p95_s:.1f}ms p99={p99_s:.1f}ms  |  "
                    f"ADAPTIVE p95={p95_a:.1f}ms p99={p99_a:.1f}ms")
            print(line)
            lines.append(line)

    _write_csv(rows, csv_path)
    _write_txt(lines, txt_path)

    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote TXT: {txt_path}")
    print(f"Next: python plot_mixdq_queue_results_v2.py {csv_path}")
