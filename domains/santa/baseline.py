"""Starting tours: (a) Hilbert curve (instant, ~25-40% above optimum), (b) greedy NN over KNN (a few minutes),
(c) LKH-3 (hours, ~1-2% above optimum without penalties). Plus KNN via cKDTree."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

LKH = Path.home() / "solvers" / "LKH-3.0.13" / "LKH"


def knn(xy: np.ndarray, K: int = 10) -> np.ndarray:
    t = cKDTree(xy)
    _, idx = t.query(xy, k=K + 1)
    return idx[:, 1:].astype(np.int32)


def hilbert_order(xy: np.ndarray, bits: int = 16) -> np.ndarray:
    x = ((xy[:, 0] - xy[:, 0].min()) / (np.ptp(xy[:, 0]) + 1e-9) * ((1 << bits) - 1)).astype(np.int64)
    y = ((xy[:, 1] - xy[:, 1].min()) / (np.ptp(xy[:, 1]) + 1e-9) * ((1 << bits) - 1)).astype(np.int64)
    d = np.zeros(len(xy), dtype=np.int64)
    s = 1 << (bits - 1)
    while s > 0:
        rx = (x & s) > 0; ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        # rot
        m = ~ry
        swap = m & rx
        x[swap], y[swap] = (s - 1 - x[swap]), (s - 1 - y[swap])
        x[m], y[m] = y[m].copy(), x[m].copy()
        s >>= 1
    order = np.argsort(d)
    z = int(np.where(order == 0)[0][0])
    return np.concatenate([order[z:], order[:z]]).astype(np.int32)


def _hilbert_d(x, y, bits):
    """Hilbert index for integer x,y in [0, 2^bits) (vectorized); curve from (0,0) to (2^bits-1, 0)."""
    x = x.copy(); y = y.copy(); d = np.zeros(len(x), dtype=np.int64); s = 1 << (bits - 1)
    while s > 0:
        rx = (x & s) > 0; ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        m = ~ry; swap = m & rx
        x[swap], y[swap] = (s - 1 - x[swap]), (s - 1 - y[swap])
        x[m], y[m] = y[m].copy(), x[m].copy()
        s >>= 1
    return d


_SYM = [lambda x, y, h: (x, y), lambda x, y, h: (y, x), lambda x, y, h: (h - 1 - x, y), lambda x, y, h: (x, h - 1 - y),
        lambda x, y, h: (h - 1 - y, x), lambda x, y, h: (y, h - 1 - x), lambda x, y, h: (h - 1 - x, h - 1 - y),
        lambda x, y, h: (h - 1 - y, h - 1 - x)]
_QUAD = [(0, 0), (0, 1), (1, 1), (1, 0)]          # BL -> TL -> TR -> BR -> (BL): loop
_MOORE_T = None


def _moore_transforms():
    """choice of Hilbert symmetry per quadrant so that the 4 curves form a CLOSED loop (Moore curve): brute force on a small grid."""
    global _MOORE_T
    if _MOORE_T is not None:
        return _MOORE_T
    import itertools
    bits = 3; h = 1 << (bits - 1)
    gx, gy = np.meshgrid(np.arange(h), np.arange(h)); gx = gx.ravel(); gy = gy.ravel()
    curves = []
    for T in _SYM:
        tx, ty = T(gx, gy, h); d = _hilbert_d(tx.astype(np.int64), ty.astype(np.int64), bits - 1)
        o = np.argsort(d); curves.append((gx[o], gy[o]))
    for combo in itertools.product(range(8), repeat=4):
        pts = []
        for q, t in enumerate(combo):
            qx, qy = _QUAD[q]; cx, cy = curves[t]; pts.append(np.stack([cx + qx * h, cy + qy * h], 1))
        P = np.concatenate(pts); nxt = np.roll(P, -1, 0)
        if np.all(np.abs(P - nxt).sum(1) == 1):
            _MOORE_T = combo; return combo
    raise RuntimeError("no Moore arrangement found")


def moore_order(xy: np.ndarray, bits: int = 16) -> np.ndarray:
    """Moore curve = closed version of Hilbert (4 quadrants in a loop): no 'corner to corner' seam when closing into a cycle."""
    x = ((xy[:, 0] - xy[:, 0].min()) / (np.ptp(xy[:, 0]) + 1e-9) * ((1 << bits) - 1)).astype(np.int64)
    y = ((xy[:, 1] - xy[:, 1].min()) / (np.ptp(xy[:, 1]) + 1e-9) * ((1 << bits) - 1)).astype(np.int64)
    h = 1 << (bits - 1); combo = _moore_transforms()
    qx = (x >= h).astype(np.int64); qy = (y >= h).astype(np.int64)
    qidx = np.array([_QUAD.index((a, b)) for a, b in zip(qx, qy)], dtype=np.int64) if len(x) < 0 else \
        np.select([(qx == 0) & (qy == 0), (qx == 0) & (qy == 1), (qx == 1) & (qy == 1)], [0, 1, 2], 3)
    key = np.empty(len(x), np.int64)
    for q in range(4):
        m = qidx == q
        if not m.any(): continue
        lx = x[m] - _QUAD[q][0] * h; ly = y[m] - _QUAD[q][1] * h
        tx, ty = _SYM[combo[q]](lx, ly, h)
        key[m] = q * (h * h) + _hilbert_d(tx.astype(np.int64), ty.astype(np.int64), bits - 1)
    order = np.argsort(key)
    z = int(np.where(order == 0)[0][0])
    return np.concatenate([order[z:], order[:z]]).astype(np.int32)


# --- adaptive curve (Sebastian's idea 23.09): Moore/Hilbert with splits at point MEDIANS ---------------------
_CS = [lambda x, y: (x, y), lambda x, y: (y, x), lambda x, y: (-x, y), lambda x, y: (x, -y),
       lambda x, y: (-y, x), lambda x, y: (y, -x), lambda x, y: (-x, -y), lambda x, y: (-y, -x)]
_CHILD = [1, 0, 0, 7]          # orientations of the U-curve children: BL transposition, TL/TR identity, BR anti-transposition


_CLAMP = 0.25       # split threshold = median clamped to [lo+c·ext, hi−c·ext] of the cell extent (c=0.5 → plain midpoint)


def _thr(v, c):
    lo, hi = v.min(), v.max(); m = np.median(v)
    return min(max(m, lo + c * (hi - lo)), hi - c * (hi - lo))


def _adaptive_hilbert(idx, X, Y, T, out, pos):
    """idx: point indices of the cell; X,Y: coordinates; T: symmetry (function); writes the order to out[pos:], returns the new pos.
    Split in the transformed frame: median of y' (bottom/top), then median of x' in each half; order BL, TL, TR, BR."""
    m = len(idx)
    if m <= 1:
        if m == 1: out[pos] = idx[0]; pos += 1
        return pos
    x, y = T(X[idx], Y[idx])
    if np.ptp(x) == 0 and np.ptp(y) == 0:
        out[pos:pos + m] = idx; return pos + m
    ty = _thr(y, _CLAMP); mb = y <= ty
    if mb.all() or not mb.any(): mb = np.arange(m) < m // 2 if np.ptp(y) == 0 else (y <= np.median(y)) & (np.arange(m) % 2 == 0)
    bot = idx[mb]; top = idx[~mb]
    def split_x(sub):
        if len(sub) <= 1: return sub, sub[:0]
        xs, _ = T(X[sub], Y[sub]); tx = _thr(xs, _CLAMP); ml = xs <= tx
        if ml.all() or not ml.any(): ml = np.arange(len(sub)) < len(sub) // 2
        return sub[ml], sub[~ml]
    bl, br = split_x(bot); tl, tr = split_x(top)
    for q, cell in enumerate((bl, tl, tr, br)):
        R = _CS[_CHILD[q]]
        Tq = (lambda T, R: (lambda a, b: R(*T(a, b))))(T, R)
        pos = _adaptive_hilbert(cell, X, Y, Tq, out, pos)
    return pos


def _adaptive_moore_raw(X, Y, combo):
    n = len(X); idx = np.arange(n); out = np.empty(n, np.int64); pos = 0
    ty = _thr(Y, _CLAMP); bot = idx[Y <= ty]; top = idx[Y > ty]
    def split_x(sub):
        tx = _thr(X[sub], _CLAMP); return sub[X[sub] <= tx], sub[X[sub] > tx]
    bl, br = split_x(bot); tl, tr = split_x(top)
    for q, cell in enumerate((bl, tl, tr, br)):
        pos = _adaptive_hilbert(cell, X, Y, _CS[combo[q]], out, pos)
    return out


_AMOORE_T = None


def _amoore_combo():
    """symmetries of the 4 quadrants giving a CLOSED loop (brute-force test on a 16×16 grid: max step after closing == 1)."""
    global _AMOORE_T
    if _AMOORE_T is not None: return _AMOORE_T
    import itertools
    g = np.arange(16); gx, gy = np.meshgrid(g, g); X = gx.ravel().astype(float); Y = gy.ravel().astype(float)
    best = None
    for combo in itertools.product(range(8), repeat=4):
        o = _adaptive_moore_raw(X, Y, combo); P = np.stack([X[o], Y[o]], 1); nxt = np.roll(P, -1, 0)
        mx = np.abs(P - nxt).sum(1).max()
        if mx == 1: _AMOORE_T = combo; return combo
        if best is None or mx < best[0]: best = (mx, combo)
    raise RuntimeError(f"no closed arrangement, best max step {best}")


def adaptive_moore_order(xy: np.ndarray, clamp: float = 0.25) -> np.ndarray:
    """Closed Moore curve with splits at point medians clamped to the middle of the cell extent (clamp):
    density taken into account, cell proportions kept (pure median → thin strips and long zigzags)."""
    global _CLAMP; _CLAMP = clamp
    X = xy[:, 0].astype(float); Y = xy[:, 1].astype(float)
    order = _adaptive_moore_raw(X, Y, _amoore_combo())
    z = int(np.where(order == 0)[0][0])
    return np.concatenate([order[z:], order[:z]]).astype(np.int32)


# --- greedy-edge (grid-free construction, follows density) -------------------------------------------------------------
def greedy_tour(xy: np.ndarray, K: int = 10) -> np.ndarray:
    """Greedy edge: KNN(K) edges in ascending order, added when degree ≤ 2 and no cycle (union-find) → fragments; joining
    fragments by nearest ends (kd-tree over the ends); closing; rotation so as to start from city 0."""
    from numba import njit
    n = len(xy); neigh = knn(xy, K)
    I = np.repeat(np.arange(n), K); J = neigh.ravel().astype(np.int64)
    m = I < J; I = I[m]; J = J[m]
    d = np.sqrt(((xy[I] - xy[J]) ** 2).sum(1)); o = np.argsort(d, kind="stable"); I = I[o]; J = J[o]

    @njit(cache=True)
    def _greedy(I, J, n):
        parent = np.arange(n); deg = np.zeros(n, np.int64); adj = -np.ones((n, 2), np.int64)
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for e in range(len(I)):
            a = I[e]; b = J[e]
            if deg[a] >= 2 or deg[b] >= 2: continue
            ra = find(a); rb = find(b)
            if ra == rb: continue
            parent[ra] = rb; adj[a, deg[a]] = b; deg[a] += 1; adj[b, deg[b]] = a; deg[b] += 1
        return adj, deg
    adj, deg = _greedy(I, J, n)
    # fragments: walk from every end (deg<2)
    ends = np.where(deg < 2)[0]; seen = np.zeros(n, bool); frags = []
    for e in ends:
        if seen[e]: continue
        path = [e]; seen[e] = True; prev = -1; cur = e
        while True:
            nxt = -1
            for t in adj[cur]:
                if t >= 0 and t != prev and not seen[t]: nxt = t; break
            if nxt < 0: break
            path.append(nxt); seen[nxt] = True; prev, cur = cur, nxt
        frags.append(np.array(path, np.int64))
    assert seen.all(), "cycle in greedy?"
    # joining fragments: GLOBALLY by nearest pairs of ends (sorted, union-find over fragments),
    # until 2 fragments remain; the last join + closing chosen so that their sum is minimal
    from scipy.spatial import cKDTree
    paths = [list(f) for f in frags]
    while len(paths) > 2:
        F = len(paths); E = np.array([xy[p[0]] for p in paths] + [xy[p[-1]] for p in paths]); own = np.array(list(range(F)) * 2)
        side = np.array([0] * F + [1] * F); tree = cKDTree(E); k = min(8, 2 * F)
        dist, idx = tree.query(E, k=k)
        a = np.repeat(np.arange(2 * F), k); b = idx.ravel(); dd = dist.ravel(); m = own[a] != own[b]
        a, b, dd = a[m], b[m], dd[m]; o = np.argsort(dd); a, b = a[o], b[o]
        parent = np.arange(F); used = np.zeros(2 * F, bool); joins = []
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, j in zip(a, b):
            if used[i] or used[j]: continue
            ri, rj = find(own[i]), find(own[j])
            if ri == rj: continue
            parent[ri] = rj; used[i] = used[j] = True; joins.append((int(i), int(j)))
            if len(joins) >= F - 2: break
        if not joins:
            break
        # build new paths: fragment graph with edges joins (every fragment of degree ≤ 2, no cycles) → chains
        nb = {f: [] for f in range(F)}
        for i, j in joins: nb[own[i]].append((int(side[i]), own[j], int(side[j]))); nb[own[j]].append((int(side[j]), own[i], int(side[i])))
        seen = np.zeros(F, bool); newp = []
        for f0 in range(F):
            if seen[f0] or len(nb[f0]) == 2: continue
            cur = []; f = f0; enter = None
            while True:
                seen[f] = True; seg = paths[f]
                if enter == 1: seg = seg[::-1]              # entering by the end -> reverse
                cur.extend(seg); exit_side = 1 if enter in (None, 0) else 0
                if enter is None and len(nb[f]) == 1: exit_side = nb[f][0][0]; cur = seg[::-1] if exit_side == 0 else list(seg)
                nxt = [(g, gs) for (s_, g, gs) in nb[f] if s_ == exit_side and not seen[g]]
                if not nxt: break
                f, enter = nxt[0]
            newp.append(cur)
        for f0 in range(F):
            if not seen[f0]: newp.append(list(paths[f0])); seen[f0] = True
        paths = newp
    if len(paths) == 2:
        A, B = paths; c1 = np.linalg.norm(xy[A[-1]] - xy[B[0]]) + np.linalg.norm(xy[B[-1]] - xy[A[0]])
        c2 = np.linalg.norm(xy[A[-1]] - xy[B[-1]]) + np.linalg.norm(xy[B[0]] - xy[A[0]])
        cur = A + (B if c1 <= c2 else B[::-1])
    else:
        cur = paths[0]
    R = np.array(cur, np.int64); assert len(R) == n and len(np.unique(R)) == n
    z = int(np.where(R == 0)[0][0])
    return np.concatenate([R[z:], R[:z]]).astype(np.int32)


def fix_long_edges_2opt(R: np.ndarray, xy: np.ndarray, top: int = 60, rounds: int = 20) -> np.ndarray:
    """2-opt exclusively between the longest edges (pairing long ones with each other): removes fragment joins/seams that
    local operators cannot see. Criterion: plain length (start, before LS). R without closing (n cities, R[0]=0)."""
    R = R.astype(np.int64).copy(); n = len(R)
    for _ in range(rounds):
        P = np.append(R, R[0]); C = xy[P]; d = np.sqrt(((C[1:] - C[:-1]) ** 2).sum(1))
        cand = np.argsort(-d)[:top]; best = (1e-9, None)
        for ii in range(len(cand)):
            i = cand[ii]
            for jj in range(ii + 1, len(cand)):
                j = cand[jj]
                a, b = (i, j) if i < j else (j, i)
                if b == a + 1: continue
                # edges (P[a],P[a+1]) and (P[b],P[b+1]) -> (P[a],P[b]) + (P[a+1],P[b+1]) by reversing P[a+1..b]
                gain = d[a] + d[b] - np.linalg.norm(C[a] - C[b]) - np.linalg.norm(C[a + 1] - C[b + 1])
                if gain > best[0]: best = (gain, (a, b))
        if best[1] is None: break
        a, b = best[1]
        if a + 1 >= 1 and b <= n - 1:
            R[a + 1:b + 1] = R[a + 1:b + 1][::-1]
        else:
            break
    z = int(np.where(R == 0)[0][0])
    return np.concatenate([R[z:], R[:z]]).astype(np.int32)


def greedy_fragments(xy: np.ndarray, K: int = 10):
    """greedy-edge fragments (paths of degree ≤2, no cycles) as a list of city arrays."""
    from numba import njit
    n = len(xy); neigh = knn(xy, K)
    I = np.repeat(np.arange(n), K); J = neigh.ravel().astype(np.int64)
    m = I < J; I = I[m]; J = J[m]
    d = np.sqrt(((xy[I] - xy[J]) ** 2).sum(1)); o = np.argsort(d, kind="stable"); I = I[o]; J = J[o]

    @njit(cache=True)
    def _greedy(I, J, n):
        parent = np.arange(n); deg = np.zeros(n, np.int64); adj = -np.ones((n, 2), np.int64)
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for e in range(len(I)):
            a = I[e]; b = J[e]
            if deg[a] >= 2 or deg[b] >= 2: continue
            ra = find(a); rb = find(b)
            if ra == rb: continue
            parent[ra] = rb; adj[a, deg[a]] = b; deg[a] += 1; adj[b, deg[b]] = a; deg[b] += 1
        return adj, deg
    adj, deg = _greedy(I, J, n)
    ends = np.where(deg < 2)[0]; seen = np.zeros(n, bool); frags = []
    for e in ends:
        if seen[e]: continue
        path = [e]; seen[e] = True; prev = -1; cur = e
        while True:
            nxt = -1
            for t in adj[cur]:
                if t >= 0 and t != prev and not seen[t]: nxt = t; break
            if nxt < 0: break
            path.append(nxt); seen[nxt] = True; prev, cur = cur, nxt
        frags.append(np.array(path, np.int64))
    assert seen.all()
    return frags


def greedy_moore_tour(xy: np.ndarray, K: int = 10, bits: int = 16, max_len: int = 400) -> np.ndarray:
    """Greedy fragments cut into pieces ≤ max_len, ordered along the Moore curve by centroid (region
    coherence), orientation of each piece by nearest ends (greedily in curve order). No "left to right"
    filaments: the curve dictates the order of visiting regions, and density — the greedy inside a piece."""
    frags = greedy_fragments(xy, K)
    pieces = []
    for f in frags:
        for a in range(0, len(f), max_len): pieces.append(f[a:a + max_len])
    cent = np.array([xy[p].mean(0) for p in pieces])
    order = moore_order(cent, bits)                     # order of pieces along the closed curve (start: piece 0)
    out = []; head = None
    for pi in order:
        p = pieces[pi]
        if head is not None and np.linalg.norm(xy[p[-1]] - xy[head]) < np.linalg.norm(xy[p[0]] - xy[head]): p = p[::-1]
        out.extend(p.tolist()); head = out[-1]
    R = np.array(out, np.int64); assert len(R) == len(xy) and len(np.unique(R)) == len(xy)
    z = int(np.where(R == 0)[0][0])
    return np.concatenate([R[z:], R[:z]]).astype(np.int32)


def nn_tour(xy: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor over KNN lists, fallback: nearest unvisited via kd-tree."""
    n = len(xy); visited = np.zeros(n, bool); tour = np.empty(n, np.int32)
    tree = cKDTree(xy); cur = 0; visited[0] = True; tour[0] = 0
    for t in range(1, n):
        nxt = -1
        for c in neigh[cur]:
            if not visited[c]:
                nxt = int(c); break
        if nxt < 0:
            k = 32
            while True:
                _, idx = tree.query(xy[cur], k=k)
                free = [int(c) for c in idx if not visited[c]]
                if free:
                    nxt = free[0]; break
                k *= 4
        visited[nxt] = True; tour[t] = nxt; cur = nxt
    return tour


def run_lkh(xy: np.ndarray, workdir: Path, time_limit_s: int = 3600, runs: int = 1, seed: int = 1,
            candidates: str = "POPMUSIC", initial_tour: Path | None = None, extra: dict | None = None) -> Path:
    """Runs LKH-3 on EUC_2D (coordinates ×1000). Returns the .tour path."""
    from domains.santa.data import write_tsplib
    workdir.mkdir(parents=True, exist_ok=True)
    tsp = workdir / "santa.tsp"; tour = workdir / f"lkh_s{seed}.tour"; par = workdir / f"lkh_s{seed}.par"
    if not tsp.exists():
        write_tsplib(xy, tsp)
    lines = [f"PROBLEM_FILE = {tsp}", f"OUTPUT_TOUR_FILE = {tour}", f"TOUR_FILE = {tour}", f"RUNS = {runs}", f"SEED = {seed}",
             f"TIME_LIMIT = {time_limit_s}", "MAX_TRIALS = 1000000", f"CANDIDATE_SET_TYPE = {candidates}",
             "INITIAL_PERIOD = 1000", "MAX_CANDIDATES = 8", "TRACE_LEVEL = 1"]
    if initial_tour:
        lines.append(f"INITIAL_TOUR_FILE = {initial_tour}")
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {v}")
    par.write_text("\n".join(lines) + "\n")
    log = workdir / f"lkh_s{seed}.log"
    with open(log, "w") as lf:
        subprocess.run([str(LKH), str(par)], stdout=lf, stderr=subprocess.STDOUT, check=False)
    return tour
