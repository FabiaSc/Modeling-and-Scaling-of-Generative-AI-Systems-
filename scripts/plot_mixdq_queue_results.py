# plot_mixdq_queue_results_v2.py
"""
Plot MixDQ queue test results (p95/p99 + Δp95) from the CSV written by load_test_stub.

Usage:
  python plot_mixdq_queue_results_v2.py /path/to/mixdq_queue_results.csv

Writes PNGs to:
  <same directory as CSV>/plots/
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_poisson(rows: List[dict], metric: str, outdir: str) -> str:
    filt = [r for r in rows if r["pattern"] == "poisson"]
    series: Dict[Tuple[int, str], List[Tuple[float, float]]] = defaultdict(list)
    for r in filt:
        w = int(r["wbits"])
        pol = r["policy"]
        lam = to_float(r["lam"])
        y = to_float(r[metric])
        series[(w, pol)].append((lam, y))
    for k in series:
        series[k].sort(key=lambda t: t[0])

    plt.figure()
    for w in sorted({k[0] for k in series.keys()}):
        for pol, marker in (("static", "o"), ("adaptive", "s")):
            pts = series.get((w, pol), [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker=marker, label=f"W{w} {pol}")
    plt.xlabel("λ (requests / second)")
    plt.ylabel(f"{metric} latency (ms)")
    plt.title(f"Poisson arrivals – {metric}")
    plt.legend()
    outpath = os.path.join(outdir, f"poisson_{metric}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath


def plot_poisson_delta(rows: List[dict], metric: str, outdir: str) -> str:
    filt = [r for r in rows if r["pattern"] == "poisson"]
    idx: Dict[Tuple[int, str, float], float] = {}
    for r in filt:
        w = int(r["wbits"])
        pol = r["policy"]
        lam = to_float(r["lam"])
        idx[(w, pol, lam)] = to_float(r[metric])

    wbits_set = sorted({int(r["wbits"]) for r in filt})
    lam_set = sorted({to_float(r["lam"]) for r in filt})

    plt.figure()
    for w in wbits_set:
        xs, ys = [], []
        for lam in lam_set:
            a = idx.get((w, "adaptive", lam))
            s = idx.get((w, "static", lam))
            if a is None or s is None:
                continue
            xs.append(lam)
            ys.append(a - s)
        if xs:
            plt.plot(xs, ys, marker="o", label=f"W{w}")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("λ (requests / second)")
    plt.ylabel(f"Δ {metric} (adaptive − static) (ms)")
    plt.title(f"Poisson arrivals – Δ {metric}")
    plt.legend()
    outpath = os.path.join(outdir, f"poisson_delta_{metric}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath


def plot_burst(rows: List[dict], metric: str, conc: int, outdir: str) -> str:
    filt = [r for r in rows if r["pattern"] == "burst" and int(r["conc"]) == conc]
    wbits_sorted = sorted({int(r["wbits"]) for r in filt})

    static_vals = []
    adaptive_vals = []
    for w in wbits_sorted:
        rs = [r for r in filt if int(r["wbits"]) == w and r["policy"] == "static"]
        ra = [r for r in filt if int(r["wbits"]) == w and r["policy"] == "adaptive"]
        static_vals.append(to_float(rs[0][metric]) if rs else float("nan"))
        adaptive_vals.append(to_float(ra[0][metric]) if ra else float("nan"))

    import numpy as np
    x = np.arange(len(wbits_sorted))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, static_vals, width, label="static")
    plt.bar(x + width/2, adaptive_vals, width, label="adaptive")
    plt.xticks(x, [f"W{w}" for w in wbits_sorted])
    plt.xlabel("MixDQ weight precision")
    plt.ylabel(f"{metric} latency (ms)")
    plt.title(f"Burst conc={conc} – {metric}")
    plt.legend()
    outpath = os.path.join(outdir, f"burst_conc{conc}_{metric}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath


def plot_burst_delta(rows: List[dict], metric: str, conc: int, outdir: str) -> str:
    filt = [r for r in rows if r["pattern"] == "burst" and int(r["conc"]) == conc]
    wbits_sorted = sorted({int(r["wbits"]) for r in filt})

    deltas = []
    for w in wbits_sorted:
        rs = [r for r in filt if int(r["wbits"]) == w and r["policy"] == "static"]
        ra = [r for r in filt if int(r["wbits"]) == w and r["policy"] == "adaptive"]
        s = to_float(rs[0][metric]) if rs else float("nan")
        a = to_float(ra[0][metric]) if ra else float("nan")
        deltas.append(a - s)

    plt.figure()
    plt.bar([f"W{w}" for w in wbits_sorted], deltas)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("MixDQ weight precision")
    plt.ylabel(f"Δ {metric} (adaptive − static) (ms)")
    plt.title(f"Burst conc={conc} – Δ {metric}")
    outpath = os.path.join(outdir, f"burst_conc{conc}_delta_{metric}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_mixdq_queue_results_v2.py <mixdq_queue_results.csv>")
        raise SystemExit(2)

    csv_path = sys.argv[1]
    rows = read_rows(csv_path)

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    outdir = os.path.join(base_dir, "plots")
    ensure_dir(outdir)

    outputs = []
    for metric in ("p95_ms", "p99_ms"):
        outputs.append(plot_poisson(rows, metric, outdir))
        outputs.append(plot_burst(rows, metric, 1, outdir))
        outputs.append(plot_burst(rows, metric, 2, outdir))

    # Delta plots (p95 only)
    outputs.append(plot_poisson_delta(rows, "p95_ms", outdir))
    outputs.append(plot_burst_delta(rows, "p95_ms", 1, outdir))
    outputs.append(plot_burst_delta(rows, "p95_ms", 2, outdir))

    print("Wrote plots:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()
