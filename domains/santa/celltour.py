"""Cell-based start (Sebastian 24.09: "points close in space = close on the path"): Karp partition.
1. k-d split by the median along the LONGER side until a cell has ≤ C cities (cells compact, equal-sized);
2. cell order: Moore curve over centroids (closed, adjacent cells consecutive);
3. inside a cell one open path: entry = city nearest the exit of the previous cell, exit = nearest to the
   centroid of the next one; NN from the entry with the exit forced at the end, then 2-opt + Or-opt on the open path (numba).
Every cell visited once → geometric neighbors are close in the tour except across cell boundaries."""
from __future__ import annotations
import numpy as np
from numba import njit


def kd_cells(xy, C=128):
    n = len(xy); cells = []; stack = [np.arange(n)]
    while stack:
        idx = stack.pop()
        if len(idx) <= C:
            cells.append(idx); continue
        P = xy[idx]; ext = P.max(0) - P.min(0); ax = 0 if ext[0] >= ext[1] else 1
        o = np.argsort(P[:, ax], kind="stable"); h = len(idx) // 2
        stack.append(idx[o[:h]]); stack.append(idx[o[h:]])
    return cells


@njit(cache=True)
def _path_opt(P, order, rounds):
    """2-opt + Or-opt (L≤3) on an open path; order[0] and order[-1] frozen. P: coordinates (m,2)."""
    m = len(order)
    for _ in range(rounds):
        improved = False
        for i in range(0, m - 2):
            a = order[i]; b = order[i + 1]
            for j in range(i + 2, m - 1):
                c = order[j]; d = order[j + 1]
                delta = (np.sqrt((P[a, 0] - P[c, 0]) ** 2 + (P[a, 1] - P[c, 1]) ** 2) + np.sqrt((P[b, 0] - P[d, 0]) ** 2 + (P[b, 1] - P[d, 1]) ** 2)
                         - np.sqrt((P[a, 0] - P[b, 0]) ** 2 + (P[a, 1] - P[b, 1]) ** 2) - np.sqrt((P[c, 0] - P[d, 0]) ** 2 + (P[c, 1] - P[d, 1]) ** 2))
                if delta < -1e-9:
                    lo = i + 1; hi = j
                    while lo < hi:
                        t = order[lo]; order[lo] = order[hi]; order[hi] = t; lo += 1; hi -= 1
                    b = order[i + 1]; improved = True
        # Or-opt: segment [s, s+L) inserted between k and k+1
        for L in range(1, 4):
            for s in range(1, m - L):
                if s + L > m - 1: break
                p0 = order[s - 1]; p1 = order[s]; p2 = order[s + L - 1]; p3 = order[s + L]
                rem = (np.sqrt((P[p0, 0] - P[p1, 0]) ** 2 + (P[p0, 1] - P[p1, 1]) ** 2) + np.sqrt((P[p2, 0] - P[p3, 0]) ** 2 + (P[p2, 1] - P[p3, 1]) ** 2)
                       - np.sqrt((P[p0, 0] - P[p3, 0]) ** 2 + (P[p0, 1] - P[p3, 1]) ** 2))
                best = 1e-9; bk = -1; brev = 0
                for k in range(0, m - 1):
                    if k >= s - 1 and k <= s + L - 1: continue
                    q0 = order[k]; q1 = order[k + 1]
                    base = np.sqrt((P[q0, 0] - P[q1, 0]) ** 2 + (P[q0, 1] - P[q1, 1]) ** 2)
                    add = np.sqrt((P[q0, 0] - P[p1, 0]) ** 2 + (P[q0, 1] - P[p1, 1]) ** 2) + np.sqrt((P[p2, 0] - P[q1, 0]) ** 2 + (P[p2, 1] - P[q1, 1]) ** 2) - base
                    addr = np.sqrt((P[q0, 0] - P[p2, 0]) ** 2 + (P[q0, 1] - P[p2, 1]) ** 2) + np.sqrt((P[p1, 0] - P[q1, 0]) ** 2 + (P[p1, 1] - P[q1, 1]) ** 2) - base
                    if rem - add > best: best = rem - add; bk = k; brev = 0
                    if rem - addr > best: best = rem - addr; bk = k; brev = 1
                if bk >= 0:
                    seg = order[s:s + L].copy()
                    if brev: seg = seg[::-1].copy()
                    rest = np.concatenate((order[:s], order[s + L:]))
                    kk = bk if bk < s else bk - L
                    order = np.concatenate((rest[:kk + 1], seg, rest[kk + 1:]))
                    improved = True
        if not improved: break
    return order


def cell_path(P, entry, exit_, rounds=50):
    """NN from entry with exit forced at the end, then _path_opt. P: (m,2). Returns the order of local indices."""
    m = len(P)
    if m == 1: return np.array([0], np.int64)
    if m == 2: return np.array([entry, exit_], np.int64) if entry != exit_ else np.array([0, 1], np.int64)
    if entry == exit_:
        d = ((P - P[entry]) ** 2).sum(1); d[entry] = -1; exit_ = int(np.argmax(d))
    left = np.ones(m, bool); left[entry] = False; left[exit_] = False
    order = [entry]; cur = entry
    for _ in range(m - 2):
        d = ((P - P[cur]) ** 2).sum(1); d[~left] = np.inf; nxt = int(np.argmin(d)); order.append(nxt); left[nxt] = False; cur = nxt
    order.append(exit_)
    return _path_opt(P, np.array(order, np.int64), rounds)


def cell_tour(xy, C=128, rounds=50):
    from domains.santa.baseline import moore_order
    cells = kd_cells(xy, C); cent = np.array([xy[c].mean(0) for c in cells])
    order = moore_order(cent, 16)
    # start from the cell containing city 0 (moore_order rotates to index 0 anyway -> cell 0; we want the cell with city 0)
    c0 = next(i for i, c in enumerate(cells) if (c == 0).any()); z = int(np.where(order == c0)[0][0]); order = np.concatenate([order[z:], order[:z]])
    out = []; prev_exit_xy = None
    for t, ci in enumerate(order):
        idx = cells[ci]; P = xy[idx]
        nxt_cent = cent[order[(t + 1) % len(order)]]
        exit_ = int(np.argmin(((P - nxt_cent) ** 2).sum(1)))
        if prev_exit_xy is None: entry = int(np.where(idx == 0)[0][0])
        else:
            d = ((P - prev_exit_xy) ** 2).sum(1); d[exit_] = np.inf if len(P) > 1 else d[exit_]; entry = int(np.argmin(d))
        loc = cell_path(P, entry, exit_, rounds)
        out.extend(idx[loc].tolist()); prev_exit_xy = xy[idx[loc[-1]]]
    R = np.array(out, np.int64); assert len(R) == len(xy) and len(np.unique(R)) == len(xy)
    z = int(np.where(R == 0)[0][0])
    return np.concatenate([R[z:], R[:z]]).astype(np.int32)
