"""Uncrossing (Sebastian's idea, an operator from his aparapi version): for the longest edges (a,b) we search
GEOMETRICALLY (kd-tree along the segment) for tour edges that cross them and uncross with 2-opt: remove (a,b),(c,d),
add (a,c),(b,d) by reversing the segment between them — arbitrarily far apart in the tour (our LS has a 1000-position window).
The plain-length gain at a crossing is always positive (triangle inequality); the penalty is computed EXACTLY by a full
recompute after every move (there are hundreds of moves, an O(n) recompute is cheap on CPU). Usage:
  uncross.py TOUR.csv OUT.csv [--top 300 --rounds 20]   or   from domains.santa.uncross import uncross_tour"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D


@njit(cache=True)
def score_arr(P, xy, prime, pen=0.1):
    s = 0.0
    for k in range(1, len(P)):
        u = P[k - 1]; v = P[k]
        dx = xy[u, 0] - xy[v, 0]; dy = xy[u, 1] - xy[v, 1]; d = np.sqrt(dx * dx + dy * dy)
        if k % 10 == 0 and prime[u] == 0: d *= (1.0 + pen)
        s += d
    return s


def _cross(p1, p2, p3, p4):
    """proper intersection of segments p1p2 and p3p4 (vectorized over arrays p3,p4)."""
    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    o1 = orient(p1, p2, p3); o2 = orient(p1, p2, p4); o3 = orient(p3, p4, p1); o4 = orient(p3, p4, p2)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def uncross_tour(R, xy, prime, top=300, rounds=20, log=None, tree=None, pen=0.1):
    """R: tour without closing (R[0]=0). Returns (R, score, number of moves)."""
    R = R.astype(np.int64).copy(); n = len(R)
    tree = tree or cKDTree(xy); moves = 0
    cur = score_arr(np.append(R, 0), xy, prime, pen)
    for rnd in range(rounds):
        pos = np.empty(n, np.int64); pos[R] = np.arange(n)
        C = xy[R]; nxt = np.roll(C, -1, 0); d = np.sqrt(((nxt - C) ** 2).sum(1))
        cand_edges = np.argsort(-d)[:top]
        cands = []
        for i in cand_edges:
            a = R[i]; b = R[(i + 1) % n]; pa = xy[a]; pb = xy[b]; L = d[i]
            if L <= 0: continue
            k = max(2, int(L / 8)); ts = np.linspace(0, 1, k)
            pts = pa[None, :] + ts[:, None] * (pb - pa)[None, :]
            near = set()
            for lst in tree.query_ball_point(pts, r=max(8.0, L / k)):
                near.update(lst)
            near.discard(a); near.discard(b)
            if not near: continue
            cs = np.fromiter(near, np.int64); pc = pos[cs]
            for side in (0, 1):                         # edge (c, succ c) and (pred c, c)
                j = pc if side == 0 else (pc - 1) % n
                c = R[j]; dd = R[(j + 1) % n]
                m = (j != i) & (j != (i - 1) % n) & (j != (i + 1) % n)
                if not m.any(): continue
                x = _cross(pa, pb, xy[c[m]], xy[dd[m]])
                for jj in j[m][x]:
                    ii, jj = (i, jj) if i < jj else (jj, i)
                    if ii + 1 < 1 or jj > n - 1 or jj - ii < 2: continue
                    gain = d[ii] + d[jj] - np.linalg.norm(C[ii] - C[jj]) - np.linalg.norm(C[(ii + 1) % n] - C[(jj + 1) % n])
                    cands.append((gain, ii, jj))
        if not cands:
            break
        cands.sort(reverse=True); applied = 0; used = np.zeros(n, bool)
        for gain, ii, jj in cands:
            if gain <= 1e-9 or used[ii] or used[jj] or used[ii + 1] or used[jj + 1 if jj + 1 < n else 0]: continue
            R2 = R.copy(); R2[ii + 1:jj + 1] = R2[ii + 1:jj + 1][::-1]
            s2 = score_arr(np.append(R2, 0), xy, prime, pen)
            if s2 < cur - 1e-9:
                R = R2; cur = s2; applied += 1; moves += 1
                used[ii] = used[jj] = used[ii + 1] = True
                if jj + 1 < n: used[jj + 1] = True
        if log: log(f"uncross {rnd}: candidates={len(cands)} applied={applied} score={cur:.1f}")
        if applied == 0:
            break
    return R.astype(np.int32), cur, moves


@njit(cache=True)
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True, parallel=True)
def _scan_all(R, pos, xy, neigh, K, n, out_j, out_g):
    """for every position p (edge R[p]->R[p+1]) the best crossing with an edge at a KNN neighbor of either end;
    plain-length 2-opt gain. out_j[p] = position of the other edge (or -1)."""
    for p in prange(0, n - 1):
        a = R[p]; b = R[p + 1]
        ax = xy[a, 0]; ay = xy[a, 1]; bx = xy[b, 0]; by = xy[b, 1]
        dab = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        best = 1e-9; bj = -1
        for side in range(2):
            u = a if side == 0 else b
            for kk in range(K):
                c = neigh[u, kk]
                for d in range(2):
                    j = pos[c] if d == 0 else pos[c] - 1
                    if j < 0 or j >= n - 1 or j == p or j == p - 1 or j == p + 1: continue
                    cc = R[j]; dd = R[j + 1]
                    cx = xy[cc, 0]; cy = xy[cc, 1]; dx = xy[dd, 0]; dy = xy[dd, 1]
                    o1 = _orient(ax, ay, bx, by, cx, cy); o2 = _orient(ax, ay, bx, by, dx, dy)
                    o3 = _orient(cx, cy, dx, dy, ax, ay); o4 = _orient(cx, cy, dx, dy, bx, by)
                    if not (o1 * o2 < 0 and o3 * o4 < 0): continue
                    i2 = p if p < j else j; j2 = j if p < j else p
                    if i2 + 1 < 1 or j2 - i2 < 2: continue
                    # 2-opt: remove (R[i2],R[i2+1]),(R[j2],R[j2+1]); add (R[i2],R[j2]),(R[i2+1],R[j2+1])
                    x0 = R[i2]; x1 = R[i2 + 1]; y0 = R[j2]; y1 = R[j2 + 1]
                    g = (np.sqrt((xy[x0, 0] - xy[x1, 0]) ** 2 + (xy[x0, 1] - xy[x1, 1]) ** 2)
                         + np.sqrt((xy[y0, 0] - xy[y1, 0]) ** 2 + (xy[y0, 1] - xy[y1, 1]) ** 2)
                         - np.sqrt((xy[x0, 0] - xy[y0, 0]) ** 2 + (xy[x0, 1] - xy[y0, 1]) ** 2)
                         - np.sqrt((xy[x1, 0] - xy[y1, 0]) ** 2 + (xy[x1, 1] - xy[y1, 1]) ** 2))
                    if g > best: best = g; bj = j
        out_j[p] = bj; out_g[p] = best if bj >= 0 else 0.0


def uncross_all(R, xy, prime, neigh, rounds=10, log=None, pen=0.1):
    """uncrossing of ALL crossings (candidates from the KNN of both edge ends), 2-opt without distance limit in the tour,
    penalty by full recompute; returns (R, score, number of moves)."""
    R = R.astype(np.int64).copy(); n = len(R); K = neigh.shape[1]; neigh = neigh.astype(np.int64); moves = 0
    cur = score_arr(np.append(R, 0), xy, prime, pen)
    for rnd in range(rounds):
        pos = np.empty(n, np.int64); pos[R] = np.arange(n)
        oj = np.empty(n, np.int64); og = np.zeros(n)
        _scan_all(R, pos, xy, neigh, K, n, oj, og)
        cand = np.where(og > 1e-9)[0]
        if len(cand) == 0: break
        order = cand[np.argsort(-og[cand])]; applied = 0
        # a candidate = 4 cities; after every accepted reversal positions change → adjacency and gain are checked on the fly
        X0 = R[order]; X1 = R[order + 1]; Y0 = R[oj[order]]; Y1 = R[oj[order] + 1]
        for t in range(len(order)):
            x0, x1, y0, y1 = X0[t], X1[t], Y0[t], Y1[t]
            a0, a1, b0, b1 = pos[x0], pos[x1], pos[y0], pos[y1]
            if abs(a0 - a1) != 1 or abs(b0 - b1) != 1: continue          # the edge no longer exists
            e1 = min(a0, a1); e2 = min(b0, b1)
            if e1 > e2: e1, e2 = e2, e1
            if e1 < 0 or e2 + 1 >= n or e2 - e1 < 2: continue
            p0, p1, q0, q1 = R[e1], R[e1 + 1], R[e2], R[e2 + 1]
            g = (np.linalg.norm(xy[p0] - xy[p1]) + np.linalg.norm(xy[q0] - xy[q1])
                 - np.linalg.norm(xy[p0] - xy[q0]) - np.linalg.norm(xy[p1] - xy[q1]))
            if g <= 1e-9: continue
            R2 = R.copy(); R2[e1 + 1:e2 + 1] = R2[e1 + 1:e2 + 1][::-1]
            s2 = score_arr(np.append(R2, 0), xy, prime, pen)
            if s2 < cur - 1e-9:
                R = R2; cur = s2; applied += 1; moves += 1
                pos[R[e1 + 1:e2 + 1]] = np.arange(e1 + 1, e2 + 1)
            if applied >= 3000: break
        if log: log(f"uncross_all {rnd}: crossings={len(cand)} applied={applied} score={cur:.1f}")
        if applied == 0: break
    return R.astype(np.int32), cur, moves


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tour"); ap.add_argument("out")
    ap.add_argument("--top", type=int, default=300); ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--all", action="store_true"); ap.add_argument("--pen", type=float, default=0.1)
    a = ap.parse_args()
    xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1); P = D.read_tour(a.tour); R = P[:-1]
    s0 = score_arr(P, xy, prime); t = time.time()
    if a.all:
        from domains.santa.baseline import knn
        R, s1, mv = uncross_all(R, xy, prime, knn(xy, 10), a.rounds, log=lambda m: print(m, flush=True), pen=a.pen)
    else:
        R, s1, mv = uncross_tour(R, xy, prime, a.top, a.rounds, log=lambda m: print(m, flush=True), pen=a.pen)
    s1 = score_arr(np.append(R, 0), xy, prime)
    P = np.append(R, 0); D.validate(P, len(xy)); assert abs(D.score(P, xy, prime) - s1) < 1e-6
    D.write_submission(P, a.out)
    print(f"DONE uncross: {s0:.1f} -> {s1:.1f} (gain {s0 - s1:.1f}, moves {mv}, {time.time() - t:.0f}s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
