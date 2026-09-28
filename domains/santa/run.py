"""Santa experiment harness: every run = a row in results/santa.db (config, score before/after, time, validation) + submission.
Usage:
  run.py --data cities.csv|synth --start hilbert|nn|<file .tour/.csv> [--W 1000 --L 3 --K 10 --rounds 2000 --time 3600] --tag name
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from domains.santa import data as D
from domains.santa.baseline import hilbert_order, moore_order, adaptive_moore_order, greedy_tour, greedy_moore_tour, fix_long_edges_2opt, knn, nn_tour
from domains.santa.gpu_ls import GpuLS, score_np

RES = Path.home() / "santa" / "results"


def log_db(db, rec):
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS runs (ts TEXT, tag TEXT, data TEXT, start TEXT, cfg TEXT, n INT, score_start REAL, "
                "score_end REAL, plain_end REAL, penalty_end REAL, rounds INT, seconds REAL, valid INT, submission TEXT)")
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rec); con.commit(); con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(D.DATA / "cities.csv"))
    ap.add_argument("--start", default="hilbert")
    ap.add_argument("--W", type=int, default=1000); ap.add_argument("--L", type=int, default=3); ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5000); ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--tag", default="ls"); ap.add_argument("--n", type=int, default=197_769)
    ap.add_argument("--ils", type=int, default=0, help="ILS iterations after LS (0 = LS only)")
    ap.add_argument("--ils-w", type=int, default=400); ap.add_argument("--ils-time", type=float, default=None)
    ap.add_argument("--ils-margin", type=int, default=300); ap.add_argument("--ils-rounds", type=int, default=8)
    ap.add_argument("--pen", type=float, default=None, help="relaxed penalty in LS/ILS (best always at full price)")
    ap.add_argument("--Lblk", type=int, default=30, help="Or-opt blocks 10..Lblk step 10 (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lk", type=int, default=0, help="LK rounds (alternating with LS) after the LS phase; 0 = no LK")
    ap.add_argument("--lk-W", type=int, default=1000)
    ap.add_argument("--lk-depth", type=int, default=5); ap.add_argument("--lk-br", default="5,3,2")
    ap.add_argument("--stall-iters", type=int, default=40); ap.add_argument("--stall-gain", type=float, default=150.0)
    ap.add_argument("--dp", type=str, default="0", help="exact DP windows: k or a list k1,k2,... (cycle: next iteration = next k; partition offsets 1..k in every sweep); 0 = off")
    ap.add_argument("--frag", type=int, default=0, help="form 2: one GPU block per 128-position fragment, mini-ILS with this many kicks after the DP phase (0 = off)")
    ap.add_argument("--kick", default="uniform", choices=["uniform", "bad"], help="ILS kicks: uniform (grid, random cuts) | bad (windows and cuts ∝ edge badness)")
    ap.add_argument("--kick-frac", type=float, default=1.0, help="fraction of windows kicked per iteration (for bad: number of windows = frac × grid)")
    ap.add_argument("--relax-current", action="store_true", help="after the relaxation phase continue from the relaxed tour (not from the best at full penalty) — test of the hypothesis 'plain first, penalty at the end'")
    ap.add_argument("--relax-time", type=float, default=None, help="time limit of the relaxation phase [s]")
    ap.add_argument("--full-time", type=float, default=None, help="time limit of the full-penalty phase after relaxation [s] (without a limit it eats the ILS budget)")
    ap.add_argument("--frag-F", type=int, default=128, help="fragment length (128 or 256 = threads per block)")
    ap.add_argument("--frag-copies", type=int, default=1, help="copies of each fragment with a different seed (small n: blocks = fragments × copies)")
    ap.add_argument("--frag-x", action="store_true", help="Or-3opt (exchange of two segments) in the fragment descent")
    ap.add_argument("--no-penalty", action="store_true", help="pure TSP (TSPLIB): all cities treated as prime → penalty always 0")
    ap.add_argument("--cell-C", type=int, default=256, help="cell size for --start cell")
    ap.add_argument("--lk-oropt", action="store_true", help="or-opt steps (segment 1-3 after t3, 2 variants) inside the LK chain alongside reversals")
    ap.add_argument("--excursion", type=float, default=0.0, help="excursion merging: segments between long edges (> TAU × 1-NN) moved to the best place (full scan over k, CPU); 0 = off")
    ap.add_argument("--orlong", type=int, default=0, help="Or-opt of long segments without distance limit (CPU, this many rounds after each Combo iteration; 0 = off)")
    ap.add_argument("--uncross", type=int, default=0, help="uncrossing of the N longest edges with 2-opt without distance limit (CPU) after each Combo iteration (0 = off)")
    ap.add_argument("--xswap2", type=int, default=0, help="generalized xswap: exchange u<->v between distant segments, each inserted at its best spot within window ±D (0 = off)")
    ap.add_argument("--xswap", type=int, default=0, help="exchange of equal-length segments ≤ L between arbitrary parts of the tour (0 = off)")
    ap.add_argument("--cand", default="knn", help="knn | alpha (alpha candidates from the 1-tree, K best by alpha)")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    xy = D.synth_cities(a.n) if a.data == "synth" else D.load_cities(a.data)
    n = len(xy); prime = D.primes_upto(n - 1).astype(np.uint8)
    if a.no_penalty: prime[:] = 1
    t0 = time.time()
    if a.cand == "alpha":
        from domains.santa.alpha import alpha_candidates
        neigh, _ = alpha_candidates(xy, k=a.K, kk=max(16, 2 * a.K)); print(f"ALPHA K={a.K} {time.time() - t0:.1f}s", flush=True)
    else:
        neigh = knn(xy, a.K); print(f"KNN K={a.K} {time.time() - t0:.1f}s", flush=True)
    if a.start == "hilbert":
        R = hilbert_order(xy)
    elif a.start == "moore":
        R = moore_order(xy)
    elif a.start == "amoore":
        R = adaptive_moore_order(xy)
    elif a.start == "cell":
        from domains.santa.celltour import cell_tour
        from domains.santa.uncross import uncross_tour
        t1 = time.time(); R = fix_long_edges_2opt(cell_tour(xy, a.cell_C), xy)
        R, s_un, mv = uncross_tour(R, xy, prime, top=400, rounds=150); R = R.astype(np.int32)
        print(f"cell tour C={a.cell_C} + long-edge 2-opt + uncross ({mv} moves) {time.time() - t1:.0f}s", flush=True)
    elif a.start == "gmoore":
        from domains.santa.uncross import uncross_tour
        t1 = time.time(); R = fix_long_edges_2opt(greedy_moore_tour(xy), xy)
        R, s_un, mv = uncross_tour(R, xy, prime, top=400, rounds=150); R = R.astype(np.int32)
        print(f"greedy-moore tour + long-edge 2-opt + uncross ({mv} moves) {time.time() - t1:.0f}s", flush=True)
    elif a.start == "greedy":
        from domains.santa.uncross import uncross_tour
        t1 = time.time(); R = fix_long_edges_2opt(greedy_tour(xy), xy)
        R, s_un, mv = uncross_tour(R, xy, prime, top=400, rounds=150); R = R.astype(np.int32)
        print(f"greedy tour + long-edge 2-opt + uncross ({mv} moves) {time.time() - t1:.0f}s", flush=True)
    elif a.start == "nn":
        t1 = time.time(); R = nn_tour(xy, neigh); print(f"NN tour {time.time() - t1:.0f}s", flush=True)
    else:
        R = D.read_tour(a.start)[:-1].astype(np.int32)
    D.validate(np.concatenate([R, [0]]), n)
    s0 = score_np(R, xy, prime); print(f"start={a.start} score={s0:.1f} plain={D.plain_len(np.concatenate([R, [0]]), xy):.1f}", flush=True)
    ls = GpuLS(xy, prime, neigh, W=a.W, Lmax=a.L, Lblk=a.Lblk)
    # --ils-time = WALL-CLOCK budget of the whole run (convergence + relaxation + ILS): gates/rollouts have a fixed cost regardless
    # of the operators (frag with a new seed converges for hours); --time additionally limits the LS/convergence phase
    deadline = (time.time() + a.ils_time) if a.ils_time else None
    def rem(cap=None):
        r = None if deadline is None else max(30.0, deadline - time.time())
        if cap is not None and (r is None or cap < r): r = cap
        return r
    R2, s1, hist = ls.run(R, max_rounds=a.rounds, log=lambda m: print(m, flush=True), time_limit=rem(a.time))
    engine = ls
    if a.lk:
        from domains.santa.gpu_lk import GpuLK

        class Combo:
            """LS (Or-opt/2-opt/swap/blocks) + LK alternating until convergence (+ optionally exact DP windows); .run interface like GpuLS."""
            def __init__(self, ls, lk, lk_rounds, dp=None, xs=None, fr=None, x2=None, un=0, ol=0, ex=0.0): self.ls, self.lk, self.lk_rounds, self.dp, self.xs, self.fr, self.x2, self.un, self.ol, self.ex = ls, lk, lk_rounds, dp, xs, fr, x2, un, ol, ex; self.dp_i = 0; self.save_cb = None; self.n = ls.n; self.xy = ls.xy; self.prime = ls.prime
            def run(self, R, max_rounds=10_000, log=None, min_gain=1e-6, time_limit=None, pen=0.1, keep_best=True):
                t0 = time.time(); hist = []; cur = score_np(R, self.xy, self.prime); best_R, best_s = R.copy(), cur
                for it in range(max_rounds):
                    R, s_lk, h1 = self.lk.run(R, max_rounds=self.lk_rounds, log=log, min_gain=min_gain, pen=pen); hist += h1
                    R, s_ls, h2 = self.ls.run(R, max_rounds=max(2, self.lk_rounds), log=log, min_gain=min_gain, pen=pen); hist += h2
                    if self.xs is not None:
                        R, s_xs, h4 = self.xs.run(R, max_rounds=self.lk_rounds, pen=pen, log=log); hist += h4
                    if self.dp:
                        dpk = self.dp[self.dp_i % len(self.dp)]; self.dp_i += 1      # cycle k=12,14,16,12,... (Sebastian's idea)
                        R, s_dp, h3 = dpk.run(R, sweeps=1, pen=pen, log=log); hist += h3
                    if self.fr is not None:
                        R, s_fr, h5 = self.fr.run(R, sweeps=1, pen=pen, log=log); hist += h5
                    if self.x2 is not None:
                        R, s_x2, h6 = self.x2.run(R, max_rounds=20, pen=pen, log=log); hist += h6
                    if self.ol:
                        from domains.santa.orlong import orlong
                        s_b = score_np(R, self.xy, self.prime)
                        Ro, s_o, mv = orlong(R, self.xy, self.prime, self.ls.neigh, rounds=self.ol, PEN=float(pen))
                        if mv > 0:
                            R = Ro.astype(np.int32)
                            if log: log(f"orlong: moves={mv} score={score_np(R, self.xy, self.prime):.1f} (pen {pen})")
                    if self.ex:
                        from domains.santa.orlong import excursion_merge
                        Re, s_e, mv = excursion_merge(R, self.xy, self.prime, self.ls.neigh, rounds=3, tau=self.ex, PEN=float(pen), max_len=5000)
                        if mv > 0:
                            R = Re.astype(np.int32)
                            if log: log(f"excursion: moves={mv} score={score_np(R, self.xy, self.prime):.1f} (pen {pen})")
                    if self.un:
                        from domains.santa.uncross import uncross_tour
                        from domains.santa.uncross import uncross_all
                        Ru, s_u, mv = uncross_all(R, self.xy, self.prime, self.ls.neigh, rounds=10, pen=float(pen))
                        Ru, s_u, mv2 = uncross_tour(Ru, self.xy, self.prime, top=self.un, rounds=5, pen=float(pen))
                        if mv + mv2 > 0:
                            R = Ru.astype(np.int32)
                            if log: log(f"uncross: moves={mv}+{mv2} score={score_np(R, self.xy, self.prime):.1f} (pen {pen})")
                    s_true = score_np(R, self.xy, self.prime)
                    if s_true < best_s:
                        best_s, best_R = s_true, R.copy()
                        if self.save_cb is not None: self.save_cb(best_R, best_s)      # saved every iteration (a kill does not lose the tour)
                    if cur - s_true < min_gain * 10: cur = s_true; break
                    cur = s_true
                    if time_limit and time.time() - t0 > time_limit: break
                return (best_R, best_s, hist) if (pen < 0.1 and keep_best) else (R, cur, hist)
        dp = None
        dp_ks = [int(x) for x in str(a.dp).split(",") if int(x) > 0]
        if dp_ks:
            from domains.santa.gpu_dp import GpuDP
            dp = [GpuDP(ls, k=k) for k in dp_ks]
        xs = None
        if a.xswap:
            from domains.santa.gpu_xswap import GpuXSwap
            xs = GpuXSwap(ls, Lmax=a.xswap)
        fr = None
        if a.frag:
            from domains.santa.gpu_frag import GpuFrag
            fr = GpuFrag(ls, kicks=a.frag, seed=a.seed, F=a.frag_F, use_x=a.frag_x, copies=a.frag_copies)
        x2 = None
        if a.xswap2:
            from domains.santa.gpu_xswap2 import GpuXSwap2
            x2 = GpuXSwap2(ls, D=a.xswap2)
        engine = Combo(ls, GpuLK(ls, W=a.lk_W, depth=a.lk_depth, br=tuple(int(x) for x in a.lk_br.split(",")), oropt=a.lk_oropt), a.lk, dp, xs, fr, x2, a.uncross, a.orlong, a.excursion)
        _gbest = {"s": float("inf")}                                     # write best_<tag>.csv ONLY when it beats the run's global record
        def _save_best(Rb, sb):
            if sb < _gbest["s"] - 1e-9:
                _gbest["s"] = sb; D.write_submission(np.concatenate([Rb, [0]]), RES / f"best_{a.tag}.csv")
        engine.save_cb = _save_best
        R2, s1, h = engine.run(R2, max_rounds=200, log=lambda m: print(m, flush=True), time_limit=rem(a.time)); hist += h
    if a.pen is not None:                                   # relaxation after full-penalty convergence
        R2, s1, h2 = engine.run(R2, max_rounds=a.rounds if not a.lk else 50, log=lambda m: print("relax " + m, flush=True), pen=a.pen, time_limit=rem(a.relax_time), keep_best=not a.relax_current); hist += h2
        if a.relax_current: print(f"relax-current: continuing from the relaxed tour, score={score_np(R2, xy, prime):.1f} plain={D.plain_len(np.concatenate([R2, [0]]), xy):.1f}", flush=True)
        R2, s1, h3 = engine.run(R2, max_rounds=a.rounds if not a.lk else 50, log=lambda m: print("full " + m, flush=True), time_limit=rem(a.full_time)); hist += h3
    if a.ils:
        from domains.santa.ils import ils
        def save_cb(Rb, sb):
            if a.lk: engine.save_cb(Rb, sb)
            else: D.write_submission(np.concatenate([Rb, [0]]), RES / f"best_{a.tag}.csv")
        R2, s1, ihist = ils(R2, xy, prime, engine, w=a.ils_w, margin=a.ils_margin, iters=a.ils, log=lambda m: print(m, flush=True),
                            time_limit=rem(), save_cb=save_cb, ls_rounds=a.ils_rounds, seed=a.seed,
                            stall_iters=a.stall_iters, stall_gain=a.stall_gain, kick=a.kick, kick_frac=a.kick_frac, pen=a.pen)
        hist += ihist
    path = np.concatenate([R2, [0]]).astype(np.int64)
    D.validate(path, n)
    s_chk = D.score(path, xy, prime); plain = D.plain_len(path, xy)
    assert abs(s_chk - s1) < 1e-3, (s_chk, s1)
    sub = RES / f"sub_{a.tag}_{int(s_chk)}.csv"; D.write_submission(path, sub)
    rec = (time.strftime("%Y-%m-%dT%H:%M:%S"), a.tag, a.data, a.start, json.dumps({"W": a.W, "L": a.L, "K": a.K, "Lblk": a.Lblk, "pen": a.pen, "ils": a.ils, "ils_w": a.ils_w, "seed": a.seed, "lk": a.lk, "lk_W": a.lk_W, "lk_depth": a.lk_depth, "lk_br": a.lk_br, "cand": a.cand, "dp": a.dp, "xswap": a.xswap, "frag": a.frag, "frag_F": a.frag_F, "frag_x": a.frag_x, "xswap2": a.xswap2, "uncross": a.uncross, "orlong": a.orlong, "excursion": a.excursion, "lk_oropt": a.lk_oropt}), n, s0, s_chk, plain,
           s_chk - plain, len(hist), round(time.time() - t0, 1), 1, str(sub))
    log_db(RES / "santa.db", rec)
    print(f"DONE {a.tag}: {s0:.1f} -> {s_chk:.1f} (plain {plain:.1f}, kara {s_chk - plain:.1f}) rounds={len(hist)} "
          f"{time.time() - t0:.0f}s -> {sub}", flush=True)


if __name__ == "__main__":
    main()
