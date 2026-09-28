"""Exchange of cities/segments of EQUAL length between arbitrary (also distant) parts of the tour -- Sebastian's idea.
Swapping [i,i+L) <-> [j,j+L) shifts no other position, so the cost delta with penalty is only the 2 boundary edges
per segment + the steps inside the segments (L steps each) = O(L), with no limit on |i-j|. Candidates: for u=R[i] every
neighbour c from KNN(u) and the predecessor/successor of c determine j; L in {1..Lmax}; reversal of each segment: 4 variants.
One thread per position i; the CPU applies a batch of moves with disjoint windows [i-1,i+L] and [j-1,j+L]."""
from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda, njit


@cuda.jit(device=True, inline=True)
def _d(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]
    return math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True, inline=True)
def _stepc(xy, prime, u, v, p, pen):
    d = _d(xy, u, v)
    if (p % 10 == 9) and prime[u] == 0:
        d *= (1.0 + pen)
    return d


@cuda.jit(device=True, inline=True)
def _seg_city(R, i, L, rev, t):
    """city at position t (0..L-1) of the segment starting at i, possibly reversed."""
    return R[i + L - 1 - t] if rev else R[i + t]


@cuda.jit(device=True, inline=True)
def _cost_block(xy, prime, R, n, a, L, src_i, src_rev, pen):
    """cost of the steps covering a segment placed at positions a..a+L-1, with the cities of segment src_i (reversed if src_rev):
    step (a-1 -> a), inner steps, step (a+L-1 -> a+L). We assume a-1 >= 0 and a+L <= n-1."""
    total = 0.0
    prev = R[a - 1]
    for t in range(L):
        c = _seg_city(R, src_i, L, src_rev, t)
        total += _stepc(xy, prime, prev, c, a - 1 + t, pen)
        prev = c
    total += _stepc(xy, prime, prev, R[a + L], a + L - 1, pen)
    return total


@cuda.jit
def xswap_moves(xy, prime, R, pos, neigh, K, n, Lmax, pen, out_gain, out_j, out_L, out_rev):
    i = cuda.grid(1)
    if i <= 0 or i >= n:
        return
    best = 1e-9; bj = -1; bL = 0; brev = 0
    u = R[i]
    for kk in range(K):
        c = neigh[u, kk]
        pc = pos[c]
        for dj in range(-1, 2):
            j = pc + dj
            if j <= 0 or j >= n or j == i:
                continue
            for L in range(1, Lmax + 1):
                if i + L > n - 1 or j + L > n - 1:
                    continue
                if j < i + L and i < j + L:           # overlapping segments
                    continue
                old = _cost_block(xy, prime, R, n, i, L, i, 0, pen) + _cost_block(xy, prime, R, n, j, L, j, 0, pen)
                if abs(i - j) == L:                   # adjacent segments: the shared edge is counted twice in old -> compute directly
                    lo = i if i < j else j
                    old = 0.0
                    for p in range(lo - 1, lo + 2 * L):
                        old += _stepc(xy, prime, R[p], R[p + 1], p, pen)
                for rv in range(4):
                    ri = rv & 1; rj = rv >> 1
                    if abs(i - j) == L:
                        # adjacent: new sequence = [prev] segB(rev) segA(rev) [next] when i<j (segA=i, segB=j) etc.
                        lo = i if i < j else j; hi = j if i < j else i
                        first_src = hi; first_rev = (rj if hi == j else ri)
                        second_src = lo; second_rev = (ri if lo == i else rj)
                        total = 0.0; prev = R[lo - 1]; p = lo - 1
                        for t in range(L):
                            cc = _seg_city(R, first_src, L, first_rev, t); total += _stepc(xy, prime, prev, cc, p, pen); prev = cc; p += 1
                        for t in range(L):
                            cc = _seg_city(R, second_src, L, second_rev, t); total += _stepc(xy, prime, prev, cc, p, pen); prev = cc; p += 1
                        total += _stepc(xy, prime, prev, R[lo + 2 * L], p, pen)
                    else:
                        total = _cost_block(xy, prime, R, n, i, L, j, rj, pen) + _cost_block(xy, prime, R, n, j, L, i, ri, pen)
                    g = old - total
                    if g > best:
                        best = g; bj = j; bL = L; brev = rv
    out_gain[i] = best if bj >= 0 else 0.0
    out_j[i] = bj; out_L[i] = bL; out_rev[i] = brev


@njit(cache=True)
def apply_xswap_batch(R, order, gain, mj, mL, mrev, n):
    used = np.zeros(n, dtype=np.uint8); applied = 0
    for idx in range(order.shape[0]):
        i = order[idx]
        if gain[i] <= 0:
            break
        j = mj[i]; L = mL[i]; rv = mrev[i]
        ok = True
        for p in range(i - 1, i + L + 1):
            if used[p]: ok = False; break
        if ok:
            for p in range(j - 1, j + L + 1):
                if used[p]: ok = False; break
        if not ok:
            continue
        for p in range(i - 1, i + L + 1): used[p] = 1
        for p in range(j - 1, j + L + 1): used[p] = 1
        A = R[i:i + L].copy(); B = R[j:j + L].copy()
        if rv & 1: A = A[::-1].copy()
        if rv >> 1: B = B[::-1].copy()
        R[i:i + L] = B; R[j:j + L] = A
        applied += 1
    return applied


class GpuXSwap:
    def __init__(self, ls, Lmax=3, tpb=128):
        self.ls = ls; self.Lmax = Lmax; self.tpb = tpb; n = ls.n
        self.d_gain = cuda.device_array(n, np.float64); self.d_j = cuda.device_array(n, np.int32)
        self.d_L = cuda.device_array(n, np.int32); self.d_rev = cuda.device_array(n, np.int32)

    def eval(self, R, pen=0.1):
        ls = self.ls; n = ls.n
        pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
        ls.d_R.copy_to_device(R.astype(np.int32)); ls.d_pos.copy_to_device(pos)
        blocks = (n + self.tpb - 1) // self.tpb
        xswap_moves[blocks, self.tpb](ls.d_xy, ls.d_prime, ls.d_R, ls.d_pos, ls.d_neigh, ls.K, n, self.Lmax, float(pen),
                                      self.d_gain, self.d_j, self.d_L, self.d_rev)
        cuda.synchronize()
        return self.d_gain.copy_to_host(), self.d_j.copy_to_host(), self.d_L.copy_to_host(), self.d_rev.copy_to_host()

    def run(self, R, max_rounds=100, log=None, min_gain=1e-6, pen=0.1, time_limit=None):
        from domains.santa.gpu_ls import PEN, score_np
        n = self.ls.n; R = R.astype(np.int32).copy(); t0 = time.time()
        cur = score_np(R, self.ls.xy, self.ls.prime); hist = []; best_s = cur; best_R = R.copy()
        for rnd in range(max_rounds):
            gain, mj, mL, mrev = self.eval(R, pen)
            cand = int((gain > min_gain).sum())
            if cand == 0:
                break
            order = np.argsort(-gain).astype(np.int64)
            applied = apply_xswap_batch(R, order, gain, mj, mL, mrev, n)
            new = score_np(R, self.ls.xy, self.ls.prime)
            hist.append((rnd, cand, applied, cur - new, new, time.time() - t0))
            if log:
                log(f"xswap {rnd}: cand={cand} applied={applied} gain={cur - new:.1f} score={new:.1f} t={time.time() - t0:.0f}s")
            if pen >= PEN and new > cur + 1e-6:
                raise RuntimeError(f"xswap: score increased {cur}->{new}")
            if new < best_s: best_s = new; best_R = R.copy()
            if abs(cur - new) < min_gain * 10 and applied < 5:
                cur = new; break
            cur = new
            if time_limit and time.time() - t0 > time_limit: break
        return (best_R, best_s, hist) if pen < PEN else (R, cur, hist)
