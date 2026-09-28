"""Generalized xswap (Sebastian's idea, version 2): exchange of city u (position i) with city v (position j) between arbitrary
parts of the tour, but EACH is inserted at the BEST place within the +-D window of the other segment, not at the same position.
The lengths of both segments do not change, so the penalty grid shifts only inside two short ranges:
A: [min(i,i'+1)-1 .. max(i,i')+1], B analogously; delta O(D). Candidates: geometric neighbours c in KNN(u) determine
segment B (j' in {pos[c]-1, pos[c]}: u inserted right before/after c); v = any city in the +-D window around pos[c]; v
inserted after i' in the +-D window around i. One thread per position i; the CPU applies a batch of moves with disjoint windows."""
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
def _newc(R, i, ip, w, p):
    """city at position p after removing R[i] and inserting w after position ip (within the window)."""
    if ip >= i:
        if p < i or p > ip:
            return R[p]
        if p == ip:
            return w
        return R[p + 1]
    else:
        if p <= ip or p > i:
            return R[p]
        if p == ip + 1:
            return w
        return R[p - 1]


@cuda.jit(device=True)
def _win_cost(xy, prime, R, i, ip, w, lo, hi, pen):
    """cost of steps lo..hi-1 after the move (remove R[i], insert w after ip); lo..hi covers the whole touched range + borders."""
    c = 0.0
    prev = _newc(R, i, ip, w, lo)
    for p in range(lo + 1, hi + 1):
        nxt = _newc(R, i, ip, w, p)
        c += _stepc(xy, prime, prev, nxt, p - 1, pen)
        prev = nxt
    return c


@cuda.jit(device=True)
def _old_cost(xy, prime, R, lo, hi, pen):
    c = 0.0
    for p in range(lo, hi):
        c += _stepc(xy, prime, R[p], R[p + 1], p, pen)
    return c


@cuda.jit
def xswap2_moves(xy, prime, R, pos, neigh, K, n, D, pen, out_gain, out_j, out_ip, out_jp):
    i = cuda.grid(1)
    if i <= 0 or i >= n - 1:
        return
    best = 1e-9; bj = -1; bip = 0; bjp = 0
    u = R[i]
    for kk in range(K):
        c = neigh[u, kk]; pc = pos[c]
        for jp in range(pc - 1, pc + 1):                      # u inserted after jp: right before c or right after c
            if jp < 1 or jp >= n - 1:
                continue
            for j in range(pc - D, pc + D + 1):               # v = R[j] removed from segment B
                if j < 1 or j >= n - 1:
                    continue
                v = R[j]
                # window B: remove j, insert u after jp
                lob = (j if j < jp + 1 else jp + 1) - 1; hib = (j if j > jp else jp) + 1
                if lob < 0 or hib > n - 1:
                    continue
                for ip in range(i - D, i + D + 1):            # v inserted after ip in segment A
                    if ip < 1 or ip >= n - 1:
                        continue
                    loa = (i if i < ip + 1 else ip + 1) - 1; hia = (i if i > ip else ip) + 1
                    if loa < 0 or hia > n - 1:
                        continue
                    if not (hia < lob or hib < loa):          # windows must be disjoint (including borders)
                        continue
                    old = _old_cost(xy, prime, R, loa, hia, pen) + _old_cost(xy, prime, R, lob, hib, pen)
                    new = _win_cost(xy, prime, R, i, ip, v, loa, hia, pen) + _win_cost(xy, prime, R, j, jp, u, lob, hib, pen)
                    g = old - new
                    if g > best:
                        best = g; bj = j; bip = ip; bjp = jp
    out_gain[i] = best if bj >= 0 else 0.0
    out_j[i] = bj; out_ip[i] = bip; out_jp[i] = bjp


@njit(cache=True)
def _rebuild(R, T, i, ip, w, lo, hi):
    for p in range(lo, hi + 1):
        if ip >= i:
            if p < i or p > ip: T[p - lo] = R[p]
            elif p == ip: T[p - lo] = w
            else: T[p - lo] = R[p + 1]
        else:
            if p <= ip or p > i: T[p - lo] = R[p]
            elif p == ip + 1: T[p - lo] = w
            else: T[p - lo] = R[p - 1]


@njit(cache=True)
def apply_xswap2_batch(R, order, gain, mj, mip, mjp, n, D):
    used = np.zeros(n, dtype=np.uint8); applied = 0
    TA = np.empty(2 * D + 4, np.int32); TB = np.empty(2 * D + 4, np.int32)
    for idx in range(order.shape[0]):
        i = order[idx]
        if gain[i] <= 0:
            break
        j = mj[i]; ip = mip[i]; jp = mjp[i]
        loa = min(i, ip + 1) - 1; hia = max(i, ip) + 1
        lob = min(j, jp + 1) - 1; hib = max(j, jp) + 1
        ok = True
        for p in range(loa, hia + 1):
            if used[p]: ok = False; break
        if ok:
            for p in range(lob, hib + 1):
                if used[p]: ok = False; break
        if not ok:
            continue
        for p in range(loa, hia + 1): used[p] = 1
        for p in range(lob, hib + 1): used[p] = 1
        u = R[i]; v = R[j]
        _rebuild(R, TA, i, ip, v, loa, hia); _rebuild(R, TB, j, jp, u, lob, hib)
        R[loa:hia + 1] = TA[:hia - loa + 1]; R[lob:hib + 1] = TB[:hib - lob + 1]
        applied += 1
    return applied


class GpuXSwap2:
    def __init__(self, ls, D=5, tpb=128):
        self.ls = ls; self.D = D; self.tpb = tpb; n = ls.n
        self.d_gain = cuda.device_array(n, np.float64); self.d_j = cuda.device_array(n, np.int32)
        self.d_ip = cuda.device_array(n, np.int32); self.d_jp = cuda.device_array(n, np.int32)

    def eval(self, R, pen=0.1):
        ls = self.ls; n = ls.n
        pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
        ls.d_R.copy_to_device(R.astype(np.int32)); ls.d_pos.copy_to_device(pos)
        blocks = (n + self.tpb - 1) // self.tpb
        xswap2_moves[blocks, self.tpb](ls.d_xy, ls.d_prime, ls.d_R, ls.d_pos, ls.d_neigh, ls.K, n, self.D, float(pen),
                                       self.d_gain, self.d_j, self.d_ip, self.d_jp)
        cuda.synchronize()
        return self.d_gain.copy_to_host(), self.d_j.copy_to_host(), self.d_ip.copy_to_host(), self.d_jp.copy_to_host()

    def run(self, R, max_rounds=100, log=None, min_gain=1e-6, pen=0.1, time_limit=None):
        from domains.santa.gpu_ls import PEN, score_np
        n = self.ls.n; R = R.astype(np.int32).copy(); t0 = time.time()
        cur = score_np(R, self.ls.xy, self.ls.prime); hist = []; best_s = cur; best_R = R.copy()
        for rnd in range(max_rounds):
            gain, mj, mip, mjp = self.eval(R, pen)
            cand = int((gain > min_gain).sum())
            if cand == 0:
                break
            order = np.argsort(-gain).astype(np.int64)
            applied = apply_xswap2_batch(R, order, gain, mj, mip, mjp, n, self.D)
            new = score_np(R, self.ls.xy, self.ls.prime)
            hist.append((rnd, cand, applied, cur - new, new, time.time() - t0))
            if log:
                log(f"xswap2 {rnd}: cand={cand} applied={applied} gain={cur - new:.1f} score={new:.1f} t={time.time() - t0:.0f}s")
            if pen >= PEN and new > cur + 1e-6:
                raise RuntimeError(f"xswap2: score increased {cur}->{new}")
            if new < best_s: best_s = new; best_R = R.copy()
            if applied < 3:
                cur = new; break
            cur = new
            if time_limit and time.time() - t0 > time_limit: break
        return (best_R, best_s, hist) if pen < PEN else (R, cur, hist)
