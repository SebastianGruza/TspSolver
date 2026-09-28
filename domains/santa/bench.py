"""Experiment gates (per Jakuczun's carousel): same start, configuration A vs B, 3 seeds x budget (default 10 min),
the MEDIAN counts; the result is recorded BEFORE the run as a threshold (min. improvement above noise). Every score recomputed from scratch.
Usage: bench.py --start <tour> --budget 600 --seeds 1,2,3 --tag name -- [run.py arguments for the variant, e.g. --pen 0.03 --Lblk 30]
Output: table + a row in ~/santa/results/bench.db (tag, cfg, seeds, scores, median, wall).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = Path.home() / "santa" / "results"
PY = Path.home() / "TspSolver" / ".venv" / "bin" / "python"


def run_variant(start, budget, seed, tag, extra):
    log = RES / f"bench_{tag}_s{seed}.log"
    cmd = [str(PY), str(ROOT / "domains/santa/run.py"), "--start", start, "--ils", "100000", "--ils-time", str(budget),
           "--seed", str(seed), "--tag", f"{tag}_s{seed}"] + extra
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, cwd=ROOT)
    txt = log.read_text()
    line = [l for l in txt.splitlines() if l.startswith("DONE")]
    score = float(line[-1].split("->")[1].split("(")[0]) if line else float("nan")
    return score, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=0.0, help="required median improvement vs start [cost units]")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    seeds = [int(s) for s in a.seeds.split(",")]
    scores = []
    for s in seeds:
        sc, wall = run_variant(a.start, a.budget, s, a.tag, extra)
        scores.append(sc); print(f"{a.tag} seed={s} score={sc:.1f} wall={wall:.0f}s", flush=True)
    med = statistics.median(scores)
    con = sqlite3.connect(RES / "bench.db")
    con.execute("CREATE TABLE IF NOT EXISTS bench (ts TEXT, tag TEXT, start TEXT, cfg TEXT, budget REAL, seeds TEXT, scores TEXT, "
                "median REAL, spread REAL, threshold REAL)")
    con.execute("INSERT INTO bench VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), a.tag, a.start, json.dumps(extra), a.budget, a.seeds, json.dumps(scores),
                 med, max(scores) - min(scores), a.threshold))
    con.commit(); con.close()
    print(f"{a.tag}: median={med:.1f} spread={max(scores) - min(scores):.1f} scores={[round(x, 1) for x in scores]}")


if __name__ == "__main__":
    main()
