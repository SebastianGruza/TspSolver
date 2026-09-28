"""Exact windows: the TOUR is split into consecutive fragments of k positions (ends frozen), each thread solves its fragment
OPTIMALLY by dynamic programming over subsets (Held-Karp on the m = k-2 inner cities), with the penalty computed
exactly -- positions inside the fragment are fixed, so it is known which steps are penalized. Fragments are disjoint -> all
improvements applied at once, without conflicts; a full pass = k split offsets.
Memory per thread: f[2^m][m] (float64) + parent[2^m][m] (int8) in global scratch; m <= 12 (k <= 14).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda

MAXM = 14


@cuda.jit(device=True, inline=True)
def _d(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]
    return math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True, inline=True)
def _step(xy, prime, u, v, p, pen):
    """cost of the step from city u (position p) to v."""
    d = _d(xy, u, v)
    if (p % 10 == 9) and prime[u] == 0:
        d *= (1.0 + pen)
    return d


@cuda.jit
def dp_windows(xy, prime, R, n, k, offset, pen, cnt, f, par, out_gain, out_order):
    w = cuda.grid(1)
    if w >= cnt:                             # threads outside the chunk (blocks*tpb > cnt) -- otherwise f[w] out of range (crash at k=16)
        return
    s = offset + w * k                       # position of the first (frozen) city of the fragment
    if s < 1 or s + k > n:                   # position 0 = the pole (excluded), the last fragment must fit
        out_gain[w] = 0.0
        return
    m = k - 2
    full = (1 << m) - 1
    first = R[s]; last = R[s + k - 1]
    # current cost of steps s .. s+k-2
    cur = 0.0
    for p in range(s, s + k - 1):
        cur += _step(xy, prime, R[p], R[p + 1], p, pen)
    F = f[w]; P = par[w]
    # base: first step from 'first' (position s) to city j (position s+1)
    for j in range(m):
        F[1 << j, j] = _step(xy, prime, first, R[s + 1 + j], s, pen)
        P[1 << j, j] = -1
    for mask in range(1, full + 1):
        cnt = 0
        mm = mask
        while mm:
            cnt += 1; mm &= mm - 1
        if cnt < 2:
            continue
        pos_i = s + cnt - 1                # cities of mask sit at s+1..s+cnt; j at s+cnt, predecessor i at s+cnt-1
        for j in range(m):
            if not (mask >> j) & 1:
                continue
            pm = mask ^ (1 << j)
            best = 1e300; bi = -1
            cj = R[s + 1 + j]
            for i in range(m):
                if not (pm >> i) & 1:
                    continue
                v = F[pm, i] + _step(xy, prime, R[s + 1 + i], cj, pos_i, pen)
                if v < best:
                    best = v; bi = i
            F[mask, j] = best; P[mask, j] = bi
    # closing: from city j (position s+m) to 'last' (position s+k-1)
    best = 1e300; bj = -1
    for j in range(m):
        v = F[full, j] + _step(xy, prime, R[s + 1 + j], last, s + m, pen)
        if v < best:
            best = v; bj = j
    gain = cur - best
    out_gain[w] = gain
    if gain > 1e-9:
        mask = full; j = bj
        for t in range(m - 1, -1, -1):
            out_order[w, t] = R[s + 1 + j]
            pj = P[mask, j]
            mask ^= (1 << j); j = pj
    else:
        out_gain[w] = 0.0


class GpuDP:
    def __init__(self, ls, k=12, max_windows=None):
        self.ls = ls; self.k = k; self.m = k - 2
        assert self.m <= MAXM
        if max_windows is None:                       # scratch budget ~1.5 GB: 2^m*m*9 B per window
            max_windows = max(256, int(1.5e9 // ((1 << self.m) * self.m * 9)))
        n = ls.n; self.nw = min(max_windows, n // k + 1)
        self.d_f = cuda.device_array((self.nw, 1 << self.m, self.m), np.float64)
        self.d_par = cuda.device_array((self.nw, 1 << self.m, self.m), np.int8)
        self.d_gain = cuda.device_array(self.nw, np.float64)
        self.d_order = cuda.device_array((self.nw, self.m), np.int32)

    def pass_offset(self, R, offset, pen=0.1, tpb=64):
        """One split offset; windows computed in chunks of nw (scratch). Returns (number of improvements, total gain)."""
        ls = self.ls; n = ls.n; k = self.k
        ls.d_R.copy_to_device(R.astype(np.int32))
        total_w = (n - offset) // k
        applied = 0; gsum = 0.0
        for w0 in range(0, total_w, self.nw):
            cnt = min(self.nw, total_w - w0)
            off = offset + w0 * k
            blocks = (cnt + tpb - 1) // tpb
            dp_windows[blocks, tpb](ls.d_xy, ls.d_prime, ls.d_R, n, k, off, float(pen), cnt, self.d_f, self.d_par,
                                    self.d_gain, self.d_order)
            cuda.synchronize()
            gain = self.d_gain.copy_to_host()[:cnt]; order = self.d_order.copy_to_host()[:cnt]
            idx = np.where(gain > 1e-9)[0]
            for w in idx:
                s = off + w * k
                R[s + 1:s + k - 1] = order[w]
            applied += len(idx); gsum += float(gain[idx].sum())
        return applied, gsum

    def run(self, R, sweeps=1, pen=0.1, log=None):
        from domains.santa.gpu_ls import score_np
        R = R.astype(np.int32).copy(); t0 = time.time(); hist = []
        cur = score_np(R, self.ls.xy, self.ls.prime)
        for sw in range(sweeps):
            for off in range(1, self.k + 1):
                ap, g = self.pass_offset(R, off, pen)
                new = score_np(R, self.ls.xy, self.ls.prime)
                hist.append((sw, off, ap, cur - new, new, time.time() - t0))
                if log:
                    log(f"dp k={self.k} off={off}: improved={ap} gain={cur - new:.1f} (dp:{g:.1f}) score={new:.1f} t={time.time() - t0:.0f}s")
                if pen >= 0.1 and new > cur + 1e-6:
                    raise RuntimeError(f"DP: score increased {cur}->{new}")
                cur = new
        return R, cur, hist
