"""Single entry point for every instance (Sebastian 24.09): Santa CSV or TSPLIB (EUC_2D / ATT / GEO), parameters and budget
from the size n, the same pipeline: start greedy(+2-opt of long edges+uncross) → LS/LK → ILS relaxed penalty 0 (frag only for large n)
→ post-processing (orlong → uncross without penalty → repair) → report with the length in the TSPLIB metric and the gap to optimum.
Usage: solve.py INSTANCE [--tag T --budget SEC --no-oropt --optimum X]
   INSTANCE = ~/santa/data/cities.csv (Santa, penalty) | .tsp file | name from ~/TspSolver/instances
Scaling (n): ILS budget = clamp(30 + 0.4·n s, 60 s, 4 h); LK W = min(n, 2000 for n≤20k, 3000 beyond); ILS windows
w = clamp(n/8, 12, 400); frag100 only for n ≥ 20k; DP k=12,14 for n ≤ 5k; or-opt in LK by default (n < 50k); LK 1 round <20k."""
from __future__ import annotations
import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/seba/mf-santa")
RES = Path.home() / "santa" / "results"; DATA = Path.home() / "santa" / "data" / "tsplib"; INST = Path.home() / "TspSolver" / "instances"
PY = Path.home() / "TspSolver" / ".venv" / "bin" / "python"; ROOT = Path.home() / "mf-santa"
RRR = 6378.388


def read_tsplib(path):
    name = Path(path).stem; wt = "EUC_2D"; rows = []; on = False
    for line in open(path):
        if line.startswith("EDGE_WEIGHT_TYPE"): wt = line.split(":")[1].strip()
        if line.startswith("NODE_COORD_SECTION"): on = True; continue
        if line.startswith("EOF"): break
        if on:
            p = line.split()
            if len(p) >= 3: rows.append((float(p[1]), float(p[2])))
    return name, wt, np.array(rows)


def geo_rad(v):
    deg = np.trunc(v); mn = v - deg; return math.pi * (deg + 5.0 * mn / 3.0) / 180.0


def tsplib_length(P, raw, wt):
    """exact length of tour P (cycle, last = first) in the TSPLIB metric on RAW coordinates."""
    A = raw[P[:-1]]; B = raw[P[1:]]
    if wt == "EUC_2D":
        return int(np.floor(np.sqrt(((A - B) ** 2).sum(1)) + 0.5).sum())        # TSPLIB nint = floor(x+0.5), not rint
    if wt == "ATT":
        r = np.sqrt(((A - B) ** 2).sum(1) / 10.0); t = np.floor(r + 0.5); t = np.where(t < r, t + 1, t); return int(t.sum())
    if wt == "GEO":
        la1 = geo_rad(A[:, 0]); lo1 = geo_rad(A[:, 1]); la2 = geo_rad(B[:, 0]); lo2 = geo_rad(B[:, 1])
        q1 = np.cos(lo1 - lo2); q2 = np.cos(la1 - la2); q3 = np.cos(la1 + la2)
        return int(np.floor(RRR * np.arccos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0).sum())
    raise ValueError(wt)


def opt_coords(raw, wt):
    """coordinates for OPTIMIZATION (our kernels: float Euclidean in 2D): EUC_2D = raw; ATT = /sqrt(10);
    GEO = equirectangular projection in km (approximation — the report is in the GEO metric anyway)."""
    if wt == "EUC_2D": return raw.astype(np.float64)
    if wt == "ATT": return raw / math.sqrt(10.0)
    if wt == "GEO":
        la = np.array([geo_rad(v) for v in raw[:, 0]]); lo = np.array([geo_rad(v) for v in raw[:, 1]])
        return np.stack([RRR * lo * np.cos(la.mean()), RRR * la], 1)
    raise ValueError(wt)


def schedule(n, budget=None, oropt=None):
    T = budget if budget else min(4 * 3600, max(60, 30 + 0.4 * n))
    W_lk = min(n - 2, 1000 if n < 2000 else (2000 if n < 20000 else 3000))  # pen==0: LK step O(1), the window can be wide
    nfrag = max(1, n // 128); copies = 1 if n >= 20000 else max(1, 2000 // nfrag)
    w = int(max(12, min(400, n // 8))); margin = int(max(2, min(300, w // 2)))
    big = n >= 20000
    p = {"lk_W": W_lk, "ils_w": w, "ils_margin": margin, "ils_rounds": 2 if big else 1,
         "frag": 100 if big else 30, "frag_copies": copies, "dp": "12,14" if n <= 5000 else "0",   # small n: frag with copies (blocks ≈ 2000)
         "stall_iters": 40 if big else (200 if n > 2000 else 500), "stall_gain": 100.0 if big else (5.0 if n > 2000 else 0.5),
         "W": min(1000, max(8, n - 3)), "K": 8 if n > 16 else max(3, n - 2),
         "time1": min(3600, max(20, 0.02 * n)), "relax": min(300, max(10, 0.003 * n)), "full": min(300, max(10, 0.003 * n)),
         "oropt": (n < 50000) if oropt is None else oropt, "T": T}
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("instance"); ap.add_argument("--tag", default=None)
    ap.add_argument("--budget", type=float, default=None); ap.add_argument("--no-oropt", action="store_true"); ap.add_argument("--oropt", action="store_true")
    ap.add_argument("--optimum", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop-max-n", type=int, default=5000, help="n ≤ threshold: population GPU memetic (solve_ga: T=8192, OX, colonies) + LK polish; above: fragment pipeline")
    ap.add_argument("--hybrid", action="store_true", help="population + descent with our LK/or-opt on the best islands after every chunk")
    ap.add_argument("--hybrid-k", type=int, default=16); ap.add_argument("--hybrid-rounds", type=int, default=3)
    ap.add_argument("--hybrid-mode", default="best", choices=["best", "worst", "kick"], help="what to polish: best islands | weakest (fresh OX child) | best after a double-bridge kick (our ILS feeds the population)")
    ap.add_argument("--ge", type=int, default=250, help="cap on population epochs (early-stop ends it anyway)")
    a = ap.parse_args()
    inst = a.instance
    if not os.path.exists(inst) and (INST / f"{inst}.tsp").exists(): inst = str(INST / f"{inst}.tsp")
    santa = inst.endswith("cities.csv")
    from domains.santa import data as D
    if santa:
        name = "santa"; wt = "SANTA"; raw = D.load_cities(inst); xy = raw; data_arg = inst; extra_pen = []
    else:
        name, wt, raw = read_tsplib(inst); xy = opt_coords(raw, wt); DATA.mkdir(parents=True, exist_ok=True)
        data_arg = str(DATA / f"{name}.csv")
        if not os.path.exists(data_arg):
            with open(data_arg, "w") as f:
                f.write("CityId,X,Y\n"); f.writelines(f"{i},{x},{y}\n" for i, (x, y) in enumerate(xy))
        extra_pen = ["--no-penalty"]
    n = len(xy); tag = a.tag or f"u_{name}"; p = schedule(n, a.budget, (False if a.no_oropt else (True if a.oropt else None)))
    optimum = a.optimum
    if optimum is None:
        try:
            for line in open(Path.home() / "TspSolver" / "README.md"):
                m = re.match(rf"\|{re.escape(name)}\|(\d+)\|(\d+)\|", line)
                if m: optimum = float(m.group(2)); break
        except Exception: pass
    print(f"[solve] {name} n={n} wt={wt} budget={p['T']:.0f}s params={json.dumps({k: v for k, v in p.items() if k != 'T'})}", flush=True)
    COMMON = ["--data", data_arg, *extra_pen, "--lk", "3" if n >= 20000 else "1", "--Lblk", "30", "--lk-depth", "5", "--lk-br", "8,5,3",
              "--K", str(p["K"]), "--W", str(p["W"]), "--lk-W", str(p["lk_W"]), "--cand", "alpha" if n > 32 else "knn", "--kick", "uniform",
              "--uncross", "300", "--orlong", "3" if n >= 20000 else "2", "--excursion", "6", "--xswap2", "20", "--ils-w", str(p["ils_w"]),
              "--ils-margin", str(p["ils_margin"]), "--seed", str(a.seed)] + (["--lk-oropt"] if p["oropt"] else []) + (["--dp", p["dp"]] if p["dp"] != "0" else [])
    t0 = time.time()
    if not santa and n <= a.pop_max_n:                            # POPULATION branch (Sebastian: <~2000 cities → GA with OX)
        from domains.tsp.io import load_tsplib, dist_matrix
        from domains.tsp.kernel import solve_ga
        c, _, e = load_tsplib(inst); Dm = dist_matrix(c, e)
        T = 8192 if n <= 2000 else (4096 if n <= 3500 else 2048)      # R[T*4, n] int32: n=2000 → 262 MB; fragments at 2–5k: window kicks undone by LK (pen 0), frag gives 0 gain
        polish = None
        if a.hybrid:                                              # LK with or-opt (our kernel, pen 0) on the best islands after every chunk
            from domains.santa.gpu_ls import GpuLS
            from domains.santa.gpu_lk import GpuLK
            from domains.santa.uncross import uncross_all
            from domains.santa.alpha import alpha_candidates
            prime0 = np.ones(n, np.uint8); neigh0, _ = alpha_candidates(xy, k=8, kk=16)
            ls0 = GpuLS(xy, prime0, neigh0, W=min(1000, n - 3)); lk0 = GpuLK(ls0, W=min(n - 2, 2000), depth=5, br=(8, 5, 3), oropt=True)
            rng0 = np.random.default_rng(a.seed)
            def polish(r, mode="best"):
                z = int(np.where(r == 0)[0][0]); r = np.concatenate([r[z:], r[:z]]).astype(np.int32)
                if mode == "kick":                                    # our ILS: double-bridge in windows, then LK/or-opt
                    from domains.santa.ils import double_bridge_window
                    w = max(24, min(400, n // 8)); stride = 2 * w; off = int(rng0.integers(0, stride))
                    for st in range(max(1, off), n - w, stride): double_bridge_window(r, st, w, rng0)
                r, _, _ = lk0.run(r, max_rounds=a.hybrid_rounds, pen=0.0)
                r, _, _ = ls0.run(r, max_rounds=2, pen=0.0)
                r, _, _ = uncross_all(r, xy, prime0, neigh0, rounds=5, pen=0.0)
                return r.astype(np.int64)
        b, ok, dt = solve_ga(Dm, T=T, grid_epochs=a.ge, seed=a.seed, ladder=True, discovery=True, trial_chunks=3,
                             db_path=str(RES / "ladder_u.db"), decel_win=1, d_min=1, decel_rho=0.5, use_tabu=1, tag=name, coords=c,
                             polish_cb=polish, polish_k=a.hybrid_k, polish_mode=a.hybrid_mode)
        bt = solve_ga.last_tour; z = int(np.where(bt == 0)[0][0]); bt = np.concatenate([bt[z:], bt[:z]])
        assert ok and len(bt) == n
        pop_tour = RES / f"sub_{tag}_pop_{b}.csv"; D.write_submission(np.append(bt, 0), str(pop_tour))
        print(f"[solve] population: best={b} ({(b / optimum - 1) * 100 if optimum else float('nan'):.3f}%) T={T} {dt:.0f}s → LK/uncross polish", flush=True)
        cmd = [str(PY), str(ROOT / "domains/santa/run.py"), "--start", str(pop_tour), "--time", "180", "--relax-time", "10", "--full-time", "10", "--pen", "0.0",
               "--ils", "0", "--tag", f"{tag}_polish"] + COMMON
        with open(RES / f"{tag}_polish.log", "w") as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "CUDA_HOME": str(Path.home() / "cudashim"), "PYTHONPATH": str(ROOT)})
        d2 = [l for l in open(RES / f"{tag}_polish.log") if l.startswith("DONE")]
        tour = d2[-1].split("->")[-1].strip() if d2 else str(pop_tour)
        Ppop = D.read_tour(str(pop_tour)); Lpop = tsplib_length(Ppop, raw, wt); Ppol = D.read_tour(tour); Lpol = tsplib_length(Ppol, raw, wt)
        if Lpol > Lpop: tour = str(pop_tour)                     # the polish computes in float — accept only if better in the TSPLIB metric
        P = D.read_tour(tour); L = tsplib_length(P, raw, wt); gap = (L / optimum - 1) * 100 if optimum else None
        rep = {"instance": name, "n": n, "metric": wt, "mode": (f"hybrid-{a.hybrid_mode}" if a.hybrid else "population"), "length": L, "pop_length": Lpop, "polish_length": Lpol, "optimum": optimum, "gap_pct": gap, "wall_s": time.time() - t0, "tour": tour, "T": T}
        print("RESULT " + json.dumps(rep), flush=True)
        with open(RES / "universal.jsonl", "a") as f: f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rep}) + "\n")
        return
    cmd = [str(PY), str(ROOT / "domains/santa/run.py"), "--start", "greedy", "--time", str(p["time1"]), "--relax-time", str(p["relax"]), "--full-time", str(p["full"]),
           "--pen", "0.0", "--ils", "1000000", "--ils-time", str(p["T"]), "--ils-rounds", str(p["ils_rounds"]), "--stall-iters", str(p["stall_iters"]),
           "--stall-gain", str(p["stall_gain"]), "--tag", tag] + (["--frag", str(p["frag"]), "--frag-copies", str(p["frag_copies"])] if p["frag"] else []) + COMMON
    log = RES / f"{tag}.log"
    with open(log, "w") as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "CUDA_HOME": str(Path.home() / "cudashim"), "PYTHONPATH": str(ROOT)})
    done = [l for l in open(log) if l.startswith("DONE")]
    if not done: print(open(log).read()[-3000:]); sys.exit("run.py without DONE")
    tour = done[-1].split("->")[-1].strip()
    if n >= 5000:                                              # post-processing (for small n the ILS has it in its loop)
        tmp1 = RES / f"{tag}.p1.csv"; tmp2 = RES / f"{tag}.p2.csv"; env = {**os.environ, "CUDA_HOME": str(Path.home() / "cudashim"), "PYTHONPATH": str(ROOT)}
        subprocess.run([str(PY), str(ROOT / "domains/santa/orlong.py"), tour, str(tmp1), "--rounds", "15"], capture_output=True, cwd=ROOT, env=env)
        subprocess.run([str(PY), str(ROOT / "domains/santa/uncross.py"), str(tmp1), str(tmp2), "--all", "--rounds", "40", "--pen", "0.0"], capture_output=True, cwd=ROOT, env=env)
        cmd2 = [str(PY), str(ROOT / "domains/santa/run.py"), "--start", str(tmp2), "--time", "600", "--ils", "0", "--tag", f"{tag}_post"] + COMMON
        with open(RES / f"{tag}_post.log", "w") as f: subprocess.run(cmd2, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        d2 = [l for l in open(RES / f"{tag}_post.log") if l.startswith("DONE")]
        if d2: tour = d2[-1].split("->")[-1].strip()
        for t in (tmp1, tmp2): t.exists() and t.unlink()
    P = D.read_tour(tour); assert len(np.unique(P[:-1])) == n
    if santa:
        prime = D.primes_upto(n - 1); s = D.score(P, xy, prime); metric = "santa"; L = s
    else:
        L = tsplib_length(P, raw, wt); metric = wt; s = float(np.sqrt(((xy[P[1:]] - xy[P[:-1]]) ** 2).sum(1)).sum())
    gap = (L / optimum - 1) * 100 if optimum else None
    rep = {"instance": name, "n": n, "metric": metric, "mode": "fragments", "length": L, "float_len": s, "optimum": optimum, "gap_pct": gap, "wall_s": time.time() - t0, "tour": tour, "params": p}
    print("RESULT " + json.dumps(rep), flush=True)
    with open(RES / "universal.jsonl", "a") as f: f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rep}) + "\n")


if __name__ == "__main__":
    main()
