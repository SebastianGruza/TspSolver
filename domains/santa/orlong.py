"""Or-opt of LONG segments without distance limit (Sebastian's idea: "sew" a tour filament into the region where its
neighbors lie): segment S=[i, i+L) (L up to thousands) cut out and inserted after position k (k = position of the geometric neighbor of an end of S,
or just before it), optionally reversed. Delta WITH PENALTY in O(1): penalty step costs in prefix sums by residues of
position mod 10 (F: step from R[q] to R[q+1] as "non-prime u"; G: reversed step from R[q] to R[q-1]); shifting the block by s
changes the penalty grid → residue (9-s) mod 10. CPU numba prange: one thread per i, all L from the list × neighbors of both ends × 2
insertion sides × 2 orientations. Batch: best moves with disjoint position ranges [min, max], then recompute."""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from numba import njit, prange
sys.path.insert(0, "/home/seba/mf-santa")

LSET = np.array([3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 70, 100, 140, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 5000], np.int64)


@njit(cache=True)
def _dist(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]; return np.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def prefix(R, xy, prime):
    n = len(R); E = np.zeros(n + 1); F = np.zeros((10, n + 1)); G = np.zeros((10, n + 1)); npf = np.zeros(n, np.int64)
    for p in range(n):
        npf[p] = 1 if prime[R[p]] == 0 else 0
    for p in range(n):
        e = _dist(xy, R[p], R[(p + 1) % n]); E[p + 1] = E[p] + e
        for r in range(10):
            F[r, p + 1] = F[r, p]; G[r, p + 1] = G[r, p]
        if npf[p] == 1: F[p % 10, p + 1] += e
        if p >= 1 and npf[p] == 1: G[p % 10, p + 1] += _dist(xy, R[p], R[p - 1])
    return E, F, G, npf


@njit(cache=True, inline="always")
def _pen(p, npf_u):
    return 1 if (p % 10 == 9 and npf_u == 1) else 0


@njit(cache=True)
def move_delta(R, xy, n, E, F, G, npf, i, L, k, rev, PEN):
    """gain (old - new) of the move: S=[i,i+L) inserted after position k (k>=i+L or k<=i-2), rev=1 reversed. Returns -1e300 when not allowed."""
    if i < 1 or i + L >= n or k < 0 or k > n - 1: return -1e300   # i+L == n → R[i+L] out of bounds (segfault in numba)
    if k >= i - 1 and k <= i + L - 1: return -1e300
    Sf = R[i + L - 1] if rev else R[i]; Sl = R[i] if rev else R[i + L - 1]
    npSf = npf[i + L - 1] if rev else npf[i]; npSl = npf[i] if rev else npf[i + L - 1]
    dS = E[i + L - 1] - E[i]                     # internal length of S (without penalty)
    if k >= i + L:                                # forward: [.. i-1][i+L .. k][S][k+1 ..]
        old = (E[k + 1] - E[i - 1]) + PEN * (F[9, k + 1] - F[9, i - 1])
        new = 0.0
        e = _dist(xy, R[i - 1], R[i + L]); new += e * (1 + PEN * _pen(i - 1, npf[i - 1]))
        rM = (9 + L) % 10                        # middle shifted by -L
        new += (E[k] - E[i + L]) + PEN * (F[rM, k] - F[rM, i + L])
        e = _dist(xy, R[k], Sf); new += e * (1 + PEN * _pen(k - L, npf[k]))
        t = k - L + 1
        if rev:
            r = (t + i + L) % 10; new += dS + PEN * (G[r, i + L] - G[r, i + 1])
        else:
            s = t - i; r = (9 - s) % 10; new += dS + PEN * (F[r, i + L - 1] - F[r, i])
        e = _dist(xy, Sl, R[(k + 1) % n]); new += e * (1 + PEN * _pen(k, npSl))
        return old - new
    else:                                         # backward: [.. k][S][k+1 .. i-1][i+L ..]
        old = (E[i + L] - E[k]) + PEN * (F[9, i + L] - F[9, k])
        new = 0.0
        e = _dist(xy, R[k], Sf); new += e * (1 + PEN * _pen(k, npf[k]))
        t = k + 1
        if rev:
            r = (t + i + L) % 10; new += dS + PEN * (G[r, i + L] - G[r, i + 1])
        else:
            s = t - i; r = (9 - s) % 10; new += dS + PEN * (F[r, i + L - 1] - F[r, i])
        e = _dist(xy, Sl, R[k + 1]); new += e * (1 + PEN * _pen(k + L, npSl))
        rM = (9 - L) % 10                        # middle shifted by +L
        new += (E[i - 1] - E[k + 1]) + PEN * (F[rM, i - 1] - F[rM, k + 1])
        e = _dist(xy, R[i - 1], R[i + L]); new += e * (1 + PEN * _pen(i - 1 + L, npf[i - 1]))
        return old - new


@njit(cache=True, parallel=True)
def eval_all(R, pos, xy, neigh, K, n, E, F, G, npf, LSET, PEN, out_g, out_L, out_k, out_r):
    for i in prange(1, n - 1):
        bg = 1e-9; bL = 0; bk = -1; br = 0
        for li in range(len(LSET)):
            L = LSET[li]
            if i + L >= n: break
            for side in range(2):                 # 0: neighbors of the start of S, 1: neighbors of the end of S
                u = R[i] if side == 0 else R[i + L - 1]
                for kk in range(K):
                    c = neigh[u, kk]; pc = pos[c]
                    # 4 combinations: (k=pc, rev) / (k=pc-1, rev) — neighbor c is adjacent to end u after the move
                    for combo in range(2):
                        if side == 0:
                            k = pc if combo == 0 else pc - 1; rev = 0 if combo == 0 else 1
                        else:
                            k = pc if combo == 0 else pc - 1; rev = 1 if combo == 0 else 0
                        g = move_delta(R, xy, n, E, F, G, npf, i, L, k, rev, PEN)
                        if g > bg: bg = g; bL = L; bk = k; br = rev
        out_g[i] = bg if bk >= 0 else 0.0; out_L[i] = bL; out_k[i] = bk; out_r[i] = br


def apply_move(R, i, L, k, rev):
    S = R[i:i + L][::-1] if rev else R[i:i + L]
    if k >= i + L:
        return np.concatenate([R[:i], R[i + L:k + 1], S, R[k + 1:]])
    return np.concatenate([R[:k + 1], S, R[k + 1:i], R[i + L:]])


def orlong(R, xy, prime, neigh, rounds=10, max_apply=200, PEN=0.1, log=None, lset=LSET):
    from domains.santa.uncross import score_arr
    sc = lambda R_: score_arr(np.append(R_, 0), xy, prime, PEN)          # score on the phase's scale (pen = relaxed or full penalty)
    R = R.astype(np.int64).copy(); n = len(R); K = neigh.shape[1]; total = 0
    cur = sc(R)
    for rnd in range(rounds):
        pos = np.empty(n, np.int64); pos[R] = np.arange(n)
        E, F, G, npf = prefix(R, xy, prime)
        g = np.zeros(n); mL = np.zeros(n, np.int64); mk = np.zeros(n, np.int64); mr = np.zeros(n, np.int64)
        eval_all(R, pos, xy, neigh.astype(np.int64), K, n, E, F, G, npf, lset, PEN, g, mL, mk, mr)
        order = np.argsort(-g); used = np.zeros(n + 1, bool); applied = 0; gain_sum = 0.0
        moves = []
        for i in order:
            if g[i] <= 1e-9 or applied >= max_apply: break
            L, k, rev = int(mL[i]), int(mk[i]), int(mr[i]); lo = min(i - 1, k); hi = max(i + L, k + 1)
            if used[lo:hi + 1].any(): continue
            used[lo:hi + 1] = True; moves.append((float(g[i]), int(i), L, k, rev))
        # apply from the end of the tour to the start so that earlier positions do not shift later moves
        for gi, i, L, k, rev in sorted(moves, key=lambda m: -max(m[1] + m[2], m[3] + 1)):
            R = apply_move(R, i, L, k, rev); applied += 1; gain_sum += gi
        new = sc(R)
        if log: log(f"orlong {rnd}: candidates={int((g > 1e-9).sum())} applied={applied} gain_pred={gain_sum:.1f} real={cur - new:.1f} score={new:.1f}")
        if new > cur + 1e-6:
            raise RuntimeError(f"orlong: score increased {cur} -> {new}")
        total += applied; cur = new
        if applied == 0: break
    return R.astype(np.int32), cur, total


def main():
    from domains.santa import data as D
    from domains.santa.baseline import knn
    ap = argparse.ArgumentParser(); ap.add_argument("tour"); ap.add_argument("out"); ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1); P = D.read_tour(a.tour); R = P[:-1].astype(np.int64); n = len(R)
    neigh = knn(xy, a.K); s0 = D.score(P, xy, prime); print(f"start {s0:.1f}", flush=True)
    if a.test:
        pos = np.empty(n, np.int64); pos[R] = np.arange(n); E, F, G, npf = prefix(R, xy, prime)
        rng = np.random.default_rng(1); ok = 0; tot = 0
        for _ in range(400):
            i = int(rng.integers(1, n - 6000)); L = int(rng.choice(LSET)); k = int(rng.integers(0, n - 1)); rev = int(rng.integers(0, 2))
            g = move_delta(R, xy, n, E, F, G, npf, i, L, k, rev, 0.1)
            if g < -1e299: continue
            R2 = apply_move(R, i, L, k, rev); real = s0 - D.score(np.append(R2, 0), xy, prime); tot += 1
            if abs(real - g) < 1e-6: ok += 1
            else: print("MISMATCH", i, L, k, rev, round(g, 3), round(real, 3))
        print(f"exact {ok}/{tot}", flush=True); return
    t = time.time(); R2, s1, mv = orlong(R, xy, prime, neigh, rounds=a.rounds, log=lambda m: print(m, flush=True))
    P2 = np.append(R2, 0); D.validate(P2, n); assert abs(D.score(P2, xy, prime) - s1) < 1e-6
    D.write_submission(P2, a.out); print(f"DONE orlong: {s0:.1f} -> {s1:.1f} (gain {s0 - s1:.1f}, moves {mv}, {time.time() - t:.0f}s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()


# --- excursion merging (Sebastian 24.09: two long connections per island instead of one) -----------------------------------
def excursions(R, xy, nn1, tau=4.0, min_len=3, max_len=20000):
    """segments [i, i+L) between consecutive LONG edges (length > tau × distance to the 1-NN of both ends).
    Returns a list of (i, L) and the array of long-edge positions."""
    n = len(R); C = xy[R]; nxt = np.roll(C, -1, 0); d = np.sqrt(((nxt - C) ** 2).sum(1))
    thr = tau * np.maximum(nn1[R], nn1[np.roll(R, -1)]); long_pos = np.where(d > thr)[0]
    segs = []
    for ai in range(len(long_pos)):                        # ALL pairs of long edges (a<b, b-a ≤ max_len):
        a = long_pos[ai]                                   # an excursion may contain its own long edges inside (sparse region)
        for bi in range(ai + 1, len(long_pos)):
            b = long_pos[bi]
            if b - a > max_len: break
            i = a + 1; L = b - a
            if min_len <= L and i >= 1 and i + L < n: segs.append((int(i), int(L)))
    return segs, long_pos


@njit(cache=True, parallel=True)
def _best_k_all(R, xy, n, E, F, G, npf, i, L, PEN, out_g, out_k, out_r):
    """full scan over k for segment (i, L): each thread computes its own k (both orientations); result per k → reduction in Python."""
    for k in prange(0, n):
        bg = 0.0; br = 0
        for rev in range(2):
            g = move_delta(R, xy, n, E, F, G, npf, i, L, k, rev, PEN)
            if g > bg: bg = g; br = rev
        out_g[k] = bg; out_k[k] = k; out_r[k] = br


def excursion_merge(R, xy, prime, neigh, rounds=5, tau=4.0, PEN=0.1, log=None, max_len=20000, max_segs=400):
    """for every excursion S (segment between two long edges, also with long ones inside): cut it out, sew the landmass,
    insert after the BEST position k from a FULL scan (numba prange, O(1) per pair), both orientations; exact delta with penalty.
    Greedy moves with disjoint ranges; full recompute after a round."""
    from domains.santa.uncross import score_arr
    sc = lambda R_: score_arr(np.append(R_, 0), xy, prime, PEN)
    R = R.astype(np.int64).copy(); n = len(R); total = 0
    nn1 = np.sqrt(((xy[neigh[:, 0]] - xy) ** 2).sum(1)); cur = sc(R)
    out_g = np.zeros(n); out_k = np.zeros(n, np.int64); out_r = np.zeros(n, np.int64)
    for rnd in range(rounds):
        E, F, G, npf = prefix(R, xy, prime)
        segs, long_pos = excursions(R, xy, nn1, tau, max_len=max_len)
        if len(segs) > max_segs:                            # priority: segments enclosed by the LONGEST edges
            C = xy[R]; d = np.sqrt(((np.roll(C, -1, 0) - C) ** 2).sum(1))
            segs.sort(key=lambda s_: -(d[s_[0] - 1] + d[s_[0] + s_[1] - 1])); segs = segs[:max_segs]
        moves = []
        for i, L in segs:
            _best_k_all(R, xy, n, E, F, G, npf, i, L, PEN, out_g, out_k, out_r)
            k = int(np.argmax(out_g))
            if out_g[k] > 1e-9: moves.append((float(out_g[k]), i, L, k, int(out_r[k])))
        moves.sort(reverse=True); used = np.zeros(n + 1, bool); applied = 0; chosen = []
        for g, i, L, k, rev in moves:
            lo = min(i - 1, k); hi = max(i + L, k + 1)
            if used[lo:hi + 1].any(): continue
            used[lo:hi + 1] = True; chosen.append((g, i, L, k, rev))
        for g, i, L, k, rev in sorted(chosen, key=lambda m: -max(m[1] + m[2], m[3] + 1)):
            R = apply_move(R, i, L, k, rev); applied += 1
        new = sc(R)
        if log: log(f"excursion {rnd}: excursions={len(segs)} long={len(long_pos)} moves={applied} gain={cur - new:.1f} score={new:.1f}")
        if new > cur + 1e-6: raise RuntimeError(f"excursion_merge: score increased {cur} -> {new}")
        total += applied; cur = new
        if applied == 0: break
    return R.astype(np.int32), cur, total
