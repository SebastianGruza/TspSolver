"""Landmass start (Sebastian 24.09: strategy of crossing long edges AT THE START, then every segment independently):
1. regions from DENSITY: connected components of the KNN graph with edges < eps (eps = ef × median distance to the 1-NN); large components
   split k-d by the median along the longer side down to ≤ C cities; small ones (< min_size) attached to the nearest region;
2. macro-tour over regions: cost(A,B) = min distance between extreme points of the outlines (16 directions); order via
   ILS (2-opt + Or-opt + double-bridge, macro._ls_order); start = Moore order over centroids;
3. ports: entry = region city nearest the previous exit, exit = nearest to the next region's outline;
4. interior: open NN path + 2-opt/Or-opt with frozen ends (celltour.cell_path).
Then the usual pipeline (LS/LK/ILS) — local operators do not cross long edges, so they optimize regions independently."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def regions_by_density(xy, K=10, ef=3.0, C=400, min_size=8):
    n = len(xy); tree = cKDTree(xy); d, nb = tree.query(xy, k=K + 1); nb = nb[:, 1:]; d = d[:, 1:]
    eps = ef * np.median(d[:, 0])
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    I = np.repeat(np.arange(n), K); J = nb.ravel(); m = d.ravel() < eps
    for a, b in zip(I[m], J[m]):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    roots = np.array([find(a) for a in range(n)]); comps = {}
    for a, r in enumerate(roots): comps.setdefault(r, []).append(a)
    comps = [np.array(v) for v in comps.values()]
    # large components → k-d down to ≤ C
    regions = []; stack = list(comps)
    while stack:
        idx = stack.pop()
        if len(idx) <= C: regions.append(idx); continue
        P = xy[idx]; ext = P.max(0) - P.min(0); ax = 0 if ext[0] >= ext[1] else 1
        o = np.argsort(P[:, ax], kind="stable"); h = len(idx) // 2; stack.append(idx[o[:h]]); stack.append(idx[o[h:]])
    # small regions → attach to the nearest large one (by nearest point)
    big = [r for r in regions if len(r) >= min_size]; small = [r for r in regions if len(r) < min_size]
    if big and small:
        lab = np.empty(n, np.int64); lab[:] = -1
        for i, r in enumerate(big): lab[r] = i
        bigpts = np.concatenate(big); bt = cKDTree(xy[bigpts])
        merged = [list(r) for r in big]
        for r in small:
            _, j = bt.query(xy[r].mean(0)); merged[lab[bigpts[j]]].extend(r.tolist())
        regions = [np.array(r) for r in merged]
    return regions, eps


def extreme_points(P, k=16):
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False); dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    idx = np.unique(np.argmax(P @ dirs.T, axis=0)); return idx


def region_dist_matrix(xy, regions):
    m = len(regions); ext = [xy[r[extreme_points(xy[r])]] for r in regions]
    Dm = np.zeros((m, m))
    for a in range(m):
        A = ext[a]
        for b in range(a + 1, m):
            B = ext[b]; dd = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(2)).min(); Dm[a, b] = Dm[b, a] = dd
    return Dm, ext


def macro_order(Dm, cent, iters=300, seed=0):
    from domains.santa.baseline import moore_order
    from numba import njit
    m = len(Dm)
    @njit(cache=True)
    def cost(order, Dm):
        c = 0.0
        for t in range(len(order)): c += Dm[order[t], order[(t + 1) % len(order)]]
        return c
    @njit(cache=True)
    def ls(order, Dm, rounds):
        m = len(order)
        for _ in range(rounds):
            imp = False
            for i in range(m - 1):
                for j in range(i + 2, m):
                    a = order[i]; b = order[i + 1]; c = order[j]; d = order[(j + 1) % m]
                    if d == a: continue
                    delta = Dm[a, c] + Dm[b, d] - Dm[a, b] - Dm[c, d]
                    if delta < -1e-9:
                        lo = i + 1; hi = j
                        while lo < hi:
                            t = order[lo]; order[lo] = order[hi]; order[hi] = t; lo += 1; hi -= 1
                        imp = True
            for L in range(1, 4):
                for s in range(0, m - L):
                    p0 = order[(s - 1) % m]; p1 = order[s]; p2 = order[s + L - 1]; p3 = order[(s + L) % m]
                    if p0 == p3: continue
                    rem = Dm[p0, p1] + Dm[p2, p3] - Dm[p0, p3]; best = 1e-9; bk = -1; brev = 0
                    for k in range(0, m):
                        if k >= s - 1 and k <= s + L - 1: continue
                        q0 = order[k]; q1 = order[(k + 1) % m]
                        if q0 == p1 or q1 == p1 or q0 == p2 or q1 == p2: continue
                        add = Dm[q0, p1] + Dm[p2, q1] - Dm[q0, q1]; addr = Dm[q0, p2] + Dm[p1, q1] - Dm[q0, q1]
                        if rem - add > best: best = rem - add; bk = k; brev = 0
                        if rem - addr > best: best = rem - addr; bk = k; brev = 1
                    if bk >= 0:
                        seg = order[s:s + L].copy()
                        if brev: seg = seg[::-1].copy()
                        rest = np.concatenate((order[:s], order[s + L:])); kk = bk if bk < s else bk - L
                        order = np.concatenate((rest[:kk + 1], seg, rest[kk + 1:])); imp = True
            if not imp: break
        return order
    order = moore_order(cent, 16).astype(np.int64) if m > 4 else np.arange(m, dtype=np.int64)
    order = ls(order, Dm, 100); best = (cost(order, Dm), order.copy()); cur = order.copy(); rng = np.random.default_rng(seed)
    for it in range(iters):
        cand = cur.copy()
        if m >= 8:
            a, b, c = np.sort(rng.choice(np.arange(1, m), 3, replace=False)); cand = np.concatenate([cand[:a], cand[b:c], cand[a:b], cand[c:]])
        cand = ls(cand, Dm, 100); cc = cost(cand, Dm)
        if cc < best[0] - 1e-9: best = (cc, cand.copy()); cur = cand
        elif cc <= cost(cur, Dm) + 1e-9: cur = cand
    return best[1], best[0]


def landmass_tour(xy, K=10, ef=3.0, C=400, iters=300, path_rounds=50, log=None):
    from domains.santa.celltour import cell_path
    regions, eps = regions_by_density(xy, K, ef, C)
    cent = np.array([xy[r].mean(0) for r in regions]); Dm, ext = region_dist_matrix(xy, regions)
    order, mc = macro_order(Dm, cent, iters)
    if log: log(f"landmass: eps={eps:.1f} regions={len(regions)} (median {int(np.median([len(r) for r in regions]))}, max {max(len(r) for r in regions)}) macro_cost={mc:.0f}")
    c0 = next(i for i, r in enumerate(regions) if (r == 0).any()); z = int(np.where(order == c0)[0][0]); order = np.concatenate([order[z:], order[:z]])
    out = []; prev_exit = None; m = len(order)
    for t, ri in enumerate(order):
        idx = regions[ri]; P = xy[idx]; nxt = ext[order[(t + 1) % m]]
        dn = np.sqrt(((P[:, None, :] - nxt[None, :, :]) ** 2).sum(2)).min(1); exit_ = int(np.argmin(dn))
        if prev_exit is None: entry = int(np.where(idx == 0)[0][0])
        else:
            d = ((P - prev_exit) ** 2).sum(1)
            if len(P) > 1: d[exit_] = np.inf
            entry = int(np.argmin(d))
        loc = cell_path(P, entry, exit_, path_rounds) if len(P) > 2 else np.arange(len(P))
        out.extend(idx[loc].tolist()); prev_exit = xy[idx[loc[-1]]]
    R = np.array(out, np.int64); assert len(R) == len(xy) and len(np.unique(R)) == len(xy)
    z = int(np.where(R == 0)[0][0]); return np.concatenate([R[z:], R[:z]]).astype(np.int32)


# --- version 2: macro with PORTS (cost of traversing a region between specific ports, DP) --------------------------------
def _interior_table(xy, regions, ports, rounds=20):
    """inter[r][i][j] = length of the open path in region r from port i to port j (cell_path); i==j → inf."""
    from domains.santa.celltour import cell_path
    inter = []
    for r, pr in zip(regions, ports):
        P = xy[r]; K = len(pr); T = np.full((K, K), np.inf)
        for i in range(K):
            for j in range(K):
                if i == j: continue
                if len(P) <= 2: T[i, j] = np.linalg.norm(P[pr[i]] - P[pr[j]]); continue
                loc = cell_path(P, int(pr[i]), int(pr[j]), rounds); Q = P[loc]
                T[i, j] = np.sqrt(((Q[1:] - Q[:-1]) ** 2).sum(1)).sum()
        inter.append(T)
    return inter


def landmass_tour2(xy, K=10, ef=4.0, C=600, nports=12, iters=150, path_rounds=50, log=None, seed=0):
    from numba import njit
    from domains.santa.celltour import cell_path
    from domains.santa.baseline import moore_order
    regions, eps = regions_by_density(xy, K, ef, C); m = len(regions)
    ports = []
    for r in regions:
        P = xy[r]; idx = extreme_points(P, nports)
        if len(idx) < 2: idx = np.array([0, len(P) - 1]) if len(P) > 1 else np.array([0, 0])
        ports.append(idx)
    Kp = max(len(p) for p in ports)
    inter = _interior_table(xy, regions, ports)
    INT = np.full((m, Kp, Kp), 1e300); PX = np.zeros((m, Kp, 2)); NP = np.zeros(m, np.int64)
    for a in range(m):
        k = len(ports[a]); NP[a] = k; INT[a, :k, :k] = inter[a]; PX[a, :k] = xy[regions[a][ports[a]]]
    cent = np.array([xy[r].mean(0) for r in regions])

    @njit(cache=True)
    def dp_cost(order, INT, PX, NP, x0):
        """cycle cost: region order[0] starts at port x0 (entry); DP over (t, exit port)."""
        m = len(order); K = INT.shape[1]
        cur = np.full(K, 1e300); r0 = order[0]
        for j in range(NP[r0]):
            if j != x0: cur[j] = INT[r0, x0, j]
        for t in range(1, m):
            r = order[t]; rp = order[t - 1]; nxt = np.full(K, 1e300)
            for e in range(NP[r]):
                # best entry e: min over the previous exit
                be = 1e300
                for pj in range(NP[rp]):
                    if cur[pj] >= 1e299: continue
                    d = np.sqrt((PX[rp, pj, 0] - PX[r, e, 0]) ** 2 + (PX[rp, pj, 1] - PX[r, e, 1]) ** 2)
                    if cur[pj] + d < be: be = cur[pj] + d
                for j in range(NP[r]):
                    if j == e: continue
                    v = be + INT[r, e, j]
                    if v < nxt[j]: nxt[j] = v
            cur = nxt
        rl = order[m - 1]; best = 1e300
        for j in range(NP[rl]):
            if cur[j] >= 1e299: continue
            d = np.sqrt((PX[rl, j, 0] - PX[r0, x0, 0]) ** 2 + (PX[rl, j, 1] - PX[r0, x0, 1]) ** 2)
            if cur[j] + d < best: best = cur[j] + d
        return best

    @njit(cache=True)
    def ls_order(order, INT, PX, NP, x0, rounds):
        m = len(order); base = dp_cost(order, INT, PX, NP, x0)
        for _ in range(rounds):
            imp = False
            for i in range(1, m - 1):                      # order[0] fixed (the region with city 0)
                for j in range(i + 1, m):
                    cand = order.copy(); lo = i; hi = j
                    while lo < hi:
                        t = cand[lo]; cand[lo] = cand[hi]; cand[hi] = t; lo += 1; hi -= 1
                    c = dp_cost(cand, INT, PX, NP, x0)
                    if c < base - 1e-9: order = cand; base = c; imp = True
            for L in range(1, 4):
                for s in range(1, m - L):
                    for k in range(0, m):
                        if k >= s - 1 and k <= s + L - 1: continue
                        seg = order[s:s + L].copy(); rest = np.concatenate((order[:s], order[s + L:])); kk = k if k < s else k - L
                        for rev in range(2):
                            sg = seg[::-1].copy() if rev == 1 else seg
                            cand = np.concatenate((rest[:kk + 1], sg, rest[kk + 1:]))
                            c = dp_cost(cand, INT, PX, NP, x0)
                            if c < base - 1e-9: order = cand; base = c; imp = True; break
            if not imp: break
        return order, base

    c0 = next(i for i, r in enumerate(regions) if (r == 0).any())
    order = moore_order(cent, 16).astype(np.int64); z = int(np.where(order == c0)[0][0]); order = np.concatenate([order[z:], order[:z]])
    x0 = int(np.where(ports[c0] == int(np.where(regions[c0] == 0)[0][0]))[0][0]) if (ports[c0] == int(np.where(regions[c0] == 0)[0][0])).any() else 0
    order, base = ls_order(order, INT, PX, NP, x0, 30); best = (base, order.copy()); cur = order.copy(); rng = np.random.default_rng(seed)
    for it in range(iters):
        cand = cur.copy()
        if m >= 8:
            a, b, c = np.sort(rng.choice(np.arange(1, m), 3, replace=False)); cand = np.concatenate([cand[:a], cand[b:c], cand[a:b], cand[c:]])
        cand, cc = ls_order(cand, INT, PX, NP, x0, 30)
        if cc < best[0] - 1e-9: best = (cc, cand.copy()); cur = cand
        elif cc <= dp_cost(cur, INT, PX, NP, x0) + 1e-9: cur = cand
    order = best[1]
    if log: log(f"landmass2: eps={eps:.1f} regions={m} ports≤{Kp} DP_cost={best[0]:.0f}")
    # port reconstruction: DP with memory (Python, m×K²)
    K = Kp; cur = np.full(K, np.inf); back = []; r0 = order[0]
    for j in range(NP[r0]):
        if j != x0: cur[j] = INT[r0, x0, j]
    ent = [{j: x0 for j in range(NP[r0])}]
    for t in range(1, m):
        r = order[t]; rp = order[t - 1]; nxt = np.full(K, np.inf); bp = {}; be_d = {}
        for e in range(NP[r]):
            dd = [(cur[pj] + np.linalg.norm(PX[rp, pj] - PX[r, e]), pj) for pj in range(NP[rp]) if np.isfinite(cur[pj])]
            be_d[e] = min(dd) if dd else (np.inf, -1)
        for j in range(NP[r]):
            opts = [(be_d[e][0] + INT[r, e, j], e) for e in range(NP[r]) if e != j and np.isfinite(be_d[e][0])]
            if opts:
                v, e = min(opts); nxt[j] = v; bp[j] = (e, be_d[e][1])
        cur = nxt; back.append(bp)
    rl = order[m - 1]; jbest = min(range(NP[rl]), key=lambda j: cur[j] + np.linalg.norm(PX[rl, j] - PX[r0, x0]) if np.isfinite(cur[j]) else np.inf)
    exits = [0] * m; entries = [0] * m; j = jbest
    for t in range(m - 1, 0, -1):
        e, pj = back[t - 1][j]; exits[t] = j; entries[t] = e; j = pj
    exits[0] = j; entries[0] = x0
    out = []
    for t, ri in enumerate(order):
        idx = regions[ri]; P = xy[idx]; e = int(ports[ri][entries[t]]); x = int(ports[ri][exits[t]])
        loc = cell_path(P, e, x, path_rounds) if len(P) > 2 else np.arange(len(P)); out.extend(idx[loc].tolist())
    R = np.array(out, np.int64); assert len(R) == len(xy) and len(np.unique(R)) == len(xy)
    z = int(np.where(R == 0)[0][0]); return np.concatenate([R[z:], R[:z]]).astype(np.int32)


# --- version 3: region interior = greedy cycle on the sub-instance, cut open at the ports; macro order as in v1 ------------------
def greedy_cycle(P, K=8):
    """greedy-edge cycle on a small sub-instance (KNN, union-find, fragments joined by nearest ends)."""
    from domains.santa.baseline import greedy_tour
    n = len(P)
    if n <= 3: return np.arange(n)
    return greedy_tour(P, min(K, n - 1)).astype(np.int64)


def landmass_tour3(xy, K=10, ef=4.0, C=600, iters=300, log=None):
    regions, eps = regions_by_density(xy, K, ef, C); m = len(regions)
    cent = np.array([xy[r].mean(0) for r in regions]); Dm, ext = region_dist_matrix(xy, regions)
    order, mc = macro_order(Dm, cent, iters)
    c0 = next(i for i, r in enumerate(regions) if (r == 0).any()); z = int(np.where(order == c0)[0][0]); order = np.concatenate([order[z:], order[:z]])
    cycles = [greedy_cycle(xy[r]) for r in regions]
    if log: log(f"landmass3: eps={eps:.1f} regions={m} macro_cost={mc:.0f}")
    out = []; prev_exit = None
    for t, ri in enumerate(order):
        idx = regions[ri]; P = xy[idx]; cyc = cycles[ri]; L = len(cyc)
        if L <= 2: loc = cyc
        else:
            nxt = ext[order[(t + 1) % m]]
            dn = np.sqrt(((P[:, None, :] - nxt[None, :, :]) ** 2).sum(2)).min(1)          # distance of every city to the next region
            dp = ((P - prev_exit) ** 2).sum(1) ** 0.5 if prev_exit is not None else np.zeros(L)
            if prev_exit is None:                                                         # region with city 0: start at 0
                s = int(np.where(idx[cyc] == 0)[0][0]); loc = np.concatenate([cyc[s:], cyc[:s]])
                # direction: the end closer to the next region
                if dn[loc[1]] < dn[loc[-1]]: loc = np.concatenate([loc[:1], loc[1:][::-1]])
            else:
                # cut edge (cyc[e], cyc[e+1]): entry at cyc[e+1]... choose e minimizing d(prev, u) + d(v, next) in both directions
                u = cyc; v = np.roll(cyc, -1)
                c1 = dp[v] + dn[u]      # entry v, exit u (we go v -> ... -> u)
                c2 = dp[u] + dn[v]      # entry u, exit v (the reverse)
                e1 = int(np.argmin(c1)); e2 = int(np.argmin(c2))
                if c1[e1] <= c2[e2]: loc = np.roll(cyc, -(e1 + 1))
                else: loc = np.roll(cyc, -(e2 + 1))[::-1]
        out.extend(idx[loc].tolist()); prev_exit = xy[idx[loc[-1]]]
    R = np.array(out, np.int64); assert len(R) == len(xy) and len(np.unique(R)) == len(xy)
    z = int(np.where(R == 0)[0][0]); return np.concatenate([R[z:], R[:z]]).astype(np.int32)
