"""Repair of very long edges (the Hilbert seam): a deep LK chain WITHOUT a window on CPU, starting from the given edge.
Standard LK (a sequence of 2-opts from a fixed t1): t1 -> t2 = neighbour in the tour (long edge), t3 in KNN(t2) with g_i > 0,
t4 = successor/predecessor of t3 according to orientation, segment reversal (on the array, O(n) - cheap at 197k),
depth up to `depth`, branching br0/br1 then 1; after every step a FULL recomputation of the score with penalty; we accept
the best prefix of the chain if it improves the true score. Usage: longfix.py TOUR.csv OUT.csv [--top 20 --depth 40]"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from numba import njit
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D
from domains.santa.baseline import knn


@njit(cache=True)
def score_arr(P, xy, prime):
    s = 0.0
    for k in range(1, len(P)):
        u = P[k - 1]; v = P[k]
        dx = xy[u, 0] - xy[v, 0]; dy = xy[u, 1] - xy[v, 1]; d = np.sqrt(dx * dx + dy * dy)
        if k % 10 == 0 and prime[u] == 0: d *= 1.1
        s += d
    return s


@njit(cache=True)
def _d(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]; return np.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def two_opt_apply(P, pos, i, j):
    """reverse P[i..j] (i<j), update pos."""
    while i < j:
        a = P[i]; b = P[j]; P[i] = b; P[j] = a; pos[b] = i; pos[a] = j; i += 1; j -= 1


def lk_chain(P0, xy, prime, neigh, i_edge, depth, br, log):
    """chain from the edge (P[i_edge], P[i_edge+1]); returns (best tour, score) after a DFS with branching."""
    n = len(P0) - 1
    best_s = score_arr(P0, xy, prime); best_P = P0.copy()
    P = P0[:-1].copy(); pos = np.empty(n, np.int64); pos[P] = np.arange(n)

    def rec(P, pos, t1, t2, G, level, hist):
        nonlocal best_s, best_P
        if level >= depth:
            return
        # t2 = free end; t1 predecessor/successor of t2 in the original edge; we look for t3 close to t2
        cands = []
        for t3 in neigh[t2]:
            g = G + _d(xy, t1, t2) - _d(xy, t2, t3)
            if g <= 0 or t3 == t1: continue
            cands.append((g, int(t3)))
        cands.sort(reverse=True)
        b = br[level] if level < len(br) else 1
        for g, t3 in cands[:b]:
            p1, p2, p3 = pos[t1], pos[t2], pos[t3]
            # orientation: edge (t1,t2) with t2 = P[p1+1] (forward) or t2 = P[p1-1]
            if p2 == p1 + 1:
                if p3 <= p1: continue               # t3 must be "past" t2, t4 = P[p3-1]... use the standard: reverse P[p2..p3-1]? -> reverse [p2, p3-1] when p3>p2
                if p3 <= p2: continue
                t4 = P[p3 - 1]
                i, j = p2, p3 - 1
            else:
                if p3 >= p1: continue
                if p3 >= p2: continue
                t4 = P[p3 + 1]
                i, j = p3 + 1, p2
            if i < 1 or j > n - 1 or i >= j: continue
            two_opt_apply(P, pos, i, j)
            full = np.append(P, 0); s = score_arr(full, xy, prime)
            closed_gain = g + _d(xy, t3, t4) - _d(xy, t4, t1)
            if s < best_s - 1e-9:
                best_s = s; best_P = full.copy()
                if log: log(f"  level {level} t3={t3} plain_gain_closed={closed_gain:.1f} TRUE {s:.1f}")
            rec(P, pos, t1, t4, g, level + 1, hist + [(i, j)])
            two_opt_apply(P, pos, i, j)   # undo
    a = P0[i_edge]; b_ = P0[i_edge + 1]
    for t1, t2 in ((a, b_), (b_, a)):
        rec(P, pos, t1, t2, 0.0, 0, [])
    return best_P, best_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tour"); ap.add_argument("out")
    ap.add_argument("--top", type=int, default=10); ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--br", default="8,4,2,1"); ap.add_argument("--K", type=int, default=12); ap.add_argument("--ratio", type=float, default=6.0)
    a = ap.parse_args()
    xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1); P = D.read_tour(a.tour)
    neigh = knn(xy, a.K); n = len(xy)
    nn1 = np.sqrt(((xy[neigh[:, 0]] - xy) ** 2).sum(1))
    s0 = score_arr(P, xy, prime); print(f"start {s0:.1f}", flush=True)
    br = [int(x) for x in a.br.split(",")]
    for it in range(a.top):
        C = xy[P]; d = np.sqrt(((C[1:] - C[:-1]) ** 2).sum(1)); ratio = d / np.maximum(nn1[P[:-1]], 1e-9)
        ratio[0] = 0; ratio[-1] = 0
        i = int(np.argmax(ratio))
        if ratio[i] < a.ratio: print("no edge >= ratio", a.ratio); break
        print(f"[{it}] edge pos {i}: {P[i]}->{P[i+1]} len={d[i]:.1f} ratio={ratio[i]:.1f}", flush=True)
        t = time.time(); P2, s2 = lk_chain(P, xy, prime, neigh, i, a.depth, br, lambda m: print(m, flush=True))
        print(f"    -> {s2:.1f} (gain {score_arr(P, xy, prime) - s2:.1f}) {time.time() - t:.0f}s", flush=True)
        if s2 < score_arr(P, xy, prime) - 1e-9:
            P = P2
        else:
            print("    no improvement for this edge - stopping"); break
    D.validate(P, n); s1 = score_arr(P, xy, prime); D.write_submission(P, a.out)
    print(f"DONE longfix: {s0:.1f} -> {s1:.1f} (gain {s0 - s1:.1f}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
