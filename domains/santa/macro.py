"""Macro-TSP (Sebastian 24.09: "more optimal around and one route; inside so that it connects"): ordering of REGIONS.
1. cut the tour at long edges (> tau x 1-NN) -> m segments (good pieces, kept whole);
2. small TSP over the segments: node = segment with two endpoints; ILS on CPU (2-opt + Or-opt L<=3 + double-bridge) on the order,
   order cost = sum of the best endpoint junctions after an exact orientation DP (2 states/segment);
3. rebuild the tour from the order and orientations; accept if the TRUE score (with penalty) dropped.
Local operators cannot see this: reordering regions is a move spanning the whole tour."""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from numba import njit
sys.path.insert(0, "/home/seba/mf-santa")


@njit(cache=True)
def _orient_dp(order, P0, P1):
    """order: segment order; P0[s], P1[s]: coordinates of the start/end of segment s. Closed cycle.
    Returns (cost, orientations) - DP over 2 states with the first segment's orientation fixed (both are tried)."""
    m = len(order); best_cost = 1e300; best_or = np.zeros(m, np.int64)
    for f0 in range(2):
        cost = np.zeros((m, 2)); back = np.zeros((m, 2), np.int64)
        cost[0, 1 - f0] = 1e300
        for t in range(1, m):
            s = order[t]; sp = order[t - 1]
            for o in range(2):
                head = P0[s] if o == 0 else P1[s]
                bc = 1e300; bo = 0
                for po in range(2):
                    tail = P1[sp] if po == 0 else P0[sp]
                    c = cost[t - 1, po] + np.sqrt((tail[0] - head[0]) ** 2 + (tail[1] - head[1]) ** 2)
                    if c < bc: bc = c; bo = po
                cost[t, o] = bc; back[t, o] = bo
        # closing: end of the last -> start of the first (orientation f0)
        s0 = order[0]; head0 = P0[s0] if f0 == 0 else P1[s0]
        for o in range(2):
            sl = order[m - 1]; tail = P1[sl] if o == 0 else P0[sl]
            c = cost[m - 1, o] + np.sqrt((tail[0] - head0[0]) ** 2 + (tail[1] - head0[1]) ** 2)
            if c < best_cost:
                best_cost = c; ors = np.zeros(m, np.int64); ors[m - 1] = o
                for t in range(m - 1, 0, -1): ors[t - 1] = back[t, ors[t]]
                best_or = ors
    return best_cost, best_or


@njit(cache=True)
def _seg_dist(P0, P1, a, b):
    """smallest distance between the endpoints of segments a and b (approximation for the order search)."""
    best = 1e300
    for pa in (P0[a], P1[a]):
        for pb in (P0[b], P1[b]):
            d = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            if d < best: best = d
    return best


@njit(cache=True)
def _ls_order(order, P0, P1, rounds):
    """2-opt + Or-opt (L<=3) on the segment order with the approximate cost _seg_dist (cycle)."""
    m = len(order)
    for _ in range(rounds):
        improved = False
        for i in range(m - 1):
            a = order[i]; b = order[i + 1]
            for j in range(i + 2, m):
                c = order[j]; d = order[(j + 1) % m]
                if d == a: continue
                delta = _seg_dist(P0, P1, a, c) + _seg_dist(P0, P1, b, d) - _seg_dist(P0, P1, a, b) - _seg_dist(P0, P1, c, d)
                if delta < -1e-9:
                    lo = i + 1; hi = j
                    while lo < hi:
                        t = order[lo]; order[lo] = order[hi]; order[hi] = t; lo += 1; hi -= 1
                    b = order[i + 1]; improved = True
        for L in range(1, 4):
            for s in range(0, m - L):
                p0 = order[(s - 1) % m]; p1 = order[s]; p2 = order[s + L - 1]; p3 = order[(s + L) % m]
                if p0 == p3: continue
                rem = _seg_dist(P0, P1, p0, p1) + _seg_dist(P0, P1, p2, p3) - _seg_dist(P0, P1, p0, p3)
                best = 1e-9; bk = -1
                for k in range(0, m):
                    if k >= s - 1 and k <= s + L - 1: continue
                    q0 = order[k]; q1 = order[(k + 1) % m]
                    if q0 == p1 or q1 == p1 or q0 == p2 or q1 == p2: continue
                    add = _seg_dist(P0, P1, q0, p1) + _seg_dist(P0, P1, p2, q1) - _seg_dist(P0, P1, q0, q1)
                    addr = _seg_dist(P0, P1, q0, p2) + _seg_dist(P0, P1, p1, q1) - _seg_dist(P0, P1, q0, q1)
                    if rem - add > best: best = rem - add; bk = k
                    if rem - addr > best: best = rem - addr; bk = k
                if bk >= 0:
                    seg = order[s:s + L].copy(); rest = np.concatenate((order[:s], order[s + L:]))
                    kk = bk if bk < s else bk - L
                    order = np.concatenate((rest[:kk + 1], seg, rest[kk + 1:])); improved = True
        if not improved: break
    return order


def macro_tsp(R, xy, prime, nn1, tau=3.0, iters=200, seed=0, pen=0.1, log=None):
    """returns (R2, score, m) - R2 = tour after the best segment order found (or R when there is no improvement)."""
    from domains.santa.uncross import score_arr
    from domains.santa.orlong import excursions
    R = R.astype(np.int64); n = len(R); s0 = score_arr(np.append(R, 0), xy, prime, pen)
    C = xy[R]; d = np.sqrt(((np.roll(C, -1, 0) - C) ** 2).sum(1))
    thr = tau * np.maximum(nn1[R], nn1[np.roll(R, -1)]); cuts = np.where(d > thr)[0]
    if len(cuts) < 3: return R.astype(np.int32), s0, len(cuts)
    # segments: positions (cuts[t]+1 .. cuts[t+1]) cyclically; segment 0 starts after the last cut (contains position 0)
    bounds = [(int(cuts[t]) + 1, int(cuts[t + 1])) for t in range(len(cuts) - 1)] + [(int(cuts[-1]) + 1, int(cuts[0]))]
    m = len(bounds); P0 = np.zeros((m, 2)); P1 = np.zeros((m, 2))
    for s_, (a, b) in enumerate(bounds):
        P0[s_] = xy[R[a % n]]; P1[s_] = xy[R[b % n]]
    rng = np.random.default_rng(seed); order = np.arange(m, dtype=np.int64)
    base_cost, _ = _orient_dp(order, P0, P1); best = (base_cost, order.copy())
    cur = order.copy()
    for it in range(iters):
        cand = cur.copy()
        if it > 0:                                          # double-bridge on the order
            if m >= 8:
                a, b, c = np.sort(rng.choice(np.arange(1, m), 3, replace=False))
                cand = np.concatenate([cand[:a], cand[b:c], cand[a:b], cand[c:]])
        cand = _ls_order(cand, P0, P1, 50)
        cost, _ = _orient_dp(cand, P0, P1)
        if cost < best[0] - 1e-9: best = (cost, cand.copy()); cur = cand
        elif cost < _orient_dp(cur, P0, P1)[0] + 1e-9: cur = cand
    cost, ors = _orient_dp(best[1], P0, P1)
    out = []
    for t, s_ in enumerate(best[1]):
        a, b = bounds[s_]; seg = [R[p % n] for p in range(a, b + 1)] if b >= a else [R[p % n] for p in range(a, b + n + 1)]
        if ors[t] == 1: seg = seg[::-1]
        out.extend(seg)
    R2 = np.array(out, np.int64); assert len(R2) == n and len(np.unique(R2)) == n
    z = int(np.where(R2 == 0)[0][0]); R2 = np.concatenate([R2[z:], R2[:z]])
    s2 = score_arr(np.append(R2, 0), xy, prime, pen)
    if log: log(f"macro tau={tau}: segments={m} junction cost {base_cost:.1f} -> {cost:.1f}; score {s0:.1f} -> {s2:.1f}")
    return (R2.astype(np.int32), s2, m) if s2 < s0 - 1e-9 else (R.astype(np.int32), s0, m)


def main():
    from domains.santa import data as D
    from domains.santa.baseline import knn
    ap = argparse.ArgumentParser(); ap.add_argument("tour"); ap.add_argument("out"); ap.add_argument("--data", default=None)
    ap.add_argument("--taus", default="6,4,3,2.5"); ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    xy = D.load_cities(a.data) if a.data else D.load_cities(); prime = D.primes_upto(len(xy) - 1)
    if a.data: prime[:] = 1
    neigh = knn(xy, 10); nn1 = np.sqrt(((xy[neigh[:, 0]] - xy) ** 2).sum(1))
    P = D.read_tour(a.tour); R = P[:-1]; s0 = D.score(P, xy, prime); print(f"start {s0:.1f}", flush=True)
    for tau in [float(t) for t in a.taus.split(",")]:
        t = time.time(); R, s, m = macro_tsp(R, xy, prime, nn1, tau=tau, iters=a.iters, log=lambda msg: print(msg, flush=True))
        print(f"  tau {tau}: {s:.1f} ({time.time() - t:.0f}s)", flush=True)
    P2 = np.append(R, 0); D.validate(P2, len(xy)); assert abs(D.score(P2, xy, prime) - s) < 1e-6
    D.write_submission(P2, a.out); print(f"DONE macro: {s0:.1f} -> {s:.1f} (gain {s0 - s:.1f}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
