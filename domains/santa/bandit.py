"""Configuration bandit (instance level): loop  rollout of arms -> choice -> play until plateau -> rollout from the new tour.
Arm = a set of run.py arguments. Rollout = short budget (default 300 s) x seeds, reward = median improvement/s.
Choice: best median, if it beats the baseline by >= threshold_x x spread; epsilon-greedy (eps) draws another arm so as not
to freeze exploration (the doctor's lesson: "learned selection lost to random draws"). Play: the winner for play_s with ILS
stall-stop; the tour after play = new start. Every decision -> ~/santa/results/bandit.db + log.
Usage: bandit.py --start <tour> --arms arms.json [--rollout 300 --seeds 1,2 --play 3600 --cycles 10 --eps 0.15]
arms.json: {"name": ["--K","10","--lk-W","1000", ...], ...}; common arguments in "_common".
"""
from __future__ import annotations

import argparse, json, random, sqlite3, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = Path.home() / "santa" / "results"
PY = Path.home() / "TspSolver" / ".venv" / "bin" / "python"


def run_once(start, budget, seed, tag, args):
    log = RES / f"bandit_{tag}_s{seed}.log"
    cmd = [str(PY), str(ROOT / "domains/santa/run.py"), "--start", start, "--ils", "1000000", "--ils-time", str(budget),
           "--seed", str(seed), "--tag", f"bandit_{tag}_s{seed}"] + args
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, cwd=ROOT)
    done = [l for l in log.read_text().splitlines() if l.startswith("DONE")]
    if not done:
        return None, None, time.time() - t0
    score = float(done[-1].split("->")[1].split("(")[0]); sub = done[-1].split("->")[-1].strip()
    return score, sub, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--rollout", type=float, default=300); ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--play", type=float, default=3600); ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.15); ap.add_argument("--threshold-x", type=float, default=2.0)
    ap.add_argument("--tag", default="bandit")
    a = ap.parse_args()
    arms = json.load(open(a.arms)); common = arms.pop("_common", [])
    seeds = [int(s) for s in a.seeds.split(",")]; rng = random.Random(0)
    con = sqlite3.connect(RES / "bandit.db")
    con.execute("CREATE TABLE IF NOT EXISTS decisions (ts TEXT, tag TEXT, cycle INT, start TEXT, score_start REAL, arm TEXT, "
                "kind TEXT, scores TEXT, median REAL, spread REAL, chosen INT, play_score REAL, play_s REAL, out TEXT)")
    start = a.start
    from domains.santa import data as D
    import numpy as np
    xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1)
    s_start = D.score(D.read_tour(start), xy, prime)
    for cyc in range(a.cycles):
        results = {}
        for name, args in arms.items():
            scores = []
            for sd in seeds:
                sc, sub, wall = run_once(start, a.rollout, sd, f"{a.tag}_c{cyc}_{name}", common + args)
                if sc is not None: scores.append(sc)
            med = statistics.median(scores) if scores else float("inf")
            spr = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
            results[name] = (med, spr, scores)
            print(f"[c{cyc}] rollout {name}: median={med:.1f} spread={spr:.1f} gain/s={(s_start - med) / a.rollout:.3f}", flush=True)
            con.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (time.strftime("%Y-%m-%dT%H:%M:%S"), a.tag, cyc, start, s_start, name, "rollout", json.dumps(scores), med, spr, 0, None, None, None))
            con.commit()
        ranked = sorted(results, key=lambda k: results[k][0])
        best = ranked[0]; base = ranked[-1]
        # epsilon-greedy: with prob. eps we play a random other arm (exploration)
        chosen = best if rng.random() >= a.eps or len(ranked) == 1 else rng.choice(ranked[1:])
        print(f"[c{cyc}] choice: {chosen} (best={best} median={results[best][0]:.1f}; eps-exploration={chosen != best})", flush=True)
        sc, sub, wall = run_once(start, a.play, seeds[0], f"{a.tag}_c{cyc}_PLAY_{chosen}", common + arms[chosen])
        con.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), a.tag, cyc, start, s_start, chosen, "play", "[]", results[chosen][0],
                     results[chosen][1], 1, sc, wall, sub)); con.commit()
        print(f"[c{cyc}] play {chosen}: {s_start:.1f} -> {sc} in {wall:.0f}s -> {sub}", flush=True)
        if sc is not None and sc < s_start - 1e-6:
            start = sub; s_start = sc
        else:
            print(f"[c{cyc}] no improvement in play - staying on the old tour", flush=True)
    print("BANDIT_DONE", start, s_start)


if __name__ == "__main__":
    main()
