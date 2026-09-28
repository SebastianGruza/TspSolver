"""Parallel local search aware of the "every 10th step" penalty on the GPU (Numba CUDA).

Architecture for n~200k: ONE tour, one thread per position i. Each thread looks for the best move for its position
(Or-opt of a segment L in {1,2,3} to a KNN neighbour, with/without reversal; 2-opt = reversal of a segment up to a KNN
neighbour), computing the EXACT cost change with penalties (cities change position inside the move window). Moves are
limited to a position window W so that evaluation costs O(W). Then the CPU greedily picks non-conflicting moves (disjoint
position windows) and applies them as a batch; loop until no improvement. Penalty: step k (1..n) from position k-1 to k
costs x1.1 when k%10==0 and the city at position k-1 is not prime -- i.e. positions p with p%10==9 decide.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda, njit

PEN = 0.1


@cuda.jit(device=True, inline=True)
def dist(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]
    return math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True, inline=True)
def step_cost(xy, prime, R, n, p, pen):
    """cost of the step from position p to p+1 (mod n), with penalty."""
    a = R[p]; q = p + 1
    if q == n:
        q = 0
    d = dist(xy, a, R[q])
    if (p % 10 == 9) and prime[a] == 0:
        d *= (1.0 + pen)
    return d


@cuda.jit(device=True, inline=True)
def seg_cost_after_oropt(xy, prime, R, n, i, L, j, rev, lo, hi, pen):
    """Cost of steps from positions lo..hi-1 (on the new tour) after Or-opt: segment [i,i+L) removed and inserted after position j.
    The new order on the window [lo,hi] is built virtually without copying: function new_city(pos)."""
    total = 0.0
    prev_city = -1
    for p in range(lo, hi + 1):
        # city at position p in the NEW tour
        if j > i:
            # segment moves right: positions i..j-L get the cities from i+L..j, then the segment at j-L+1..j
            if p < i or p > j:
                c = R[p]
            elif p <= j - L:
                c = R[p + L]
            else:
                k = p - (j - L + 1)
                c = R[i + L - 1 - k] if rev else R[i + k]
        else:
            # j < i: segment goes left, to positions j+1..j+L; cities j+1..i-1 shift right by L
            if p <= j or p >= i + L:
                c = R[p]
            elif p <= j + L:
                k = p - (j + 1)
                c = R[i + L - 1 - k] if rev else R[i + k]
            else:
                c = R[p - L]
        if prev_city >= 0:
            d = dist(xy, prev_city, c)
            if ((p - 1) % 10 == 9) and prime[prev_city] == 0:
                d *= (1.0 + pen)
            total += d
        prev_city = c
    return total


@cuda.jit(device=True, inline=True)
def seg_cost_current(xy, prime, R, n, lo, hi, pen):
    total = 0.0
    for p in range(lo, hi):
        total += step_cost(xy, prime, R, n, p, pen)
    return total


@cuda.jit
def eval_moves(xy, prime, R, pos, neigh, K, n, W, Lmax, pen, Lblk, out_gain, out_j, out_L, out_rev, out_type):
    i = cuda.grid(1)
    if i <= 0 or i >= n:            # position 0 = the pole, never moved
        return
    best = 1e-9; bj = -1; bL = 0; brev = 0; btype = 0
    # --- Or-opt: segment [i, i+L) -> after position j (j = position of a neighbour of city R[i] or R[i+L-1]) ---
    nL = Lmax + Lblk // 10
    for li in range(nL):
        L = li + 1 if li < Lmax else (li - Lmax + 1) * 10       # 1..Lmax, then 10, 20, ..., Lblk
        if i + L > n:
            continue
        for side in range(2):
            a = R[i] if side == 0 else R[i + L - 1]
            for kk in range(K):
                c = neigh[a, kk]
                j = pos[c]
                if j >= i and j < i + L:
                    continue
                if j < 0:
                    continue
                dd = j - i
                if dd < 0:
                    dd = -dd
                if dd > W:
                    continue
                # insertion after j (j==0 allowed: after the pole), also the "before j" variant = after j-1
                for before in range(2):
                    jj = j - before
                    if jj < 0:
                        continue
                    if jj >= i and jj < i + L:
                        continue
                    if jj == i - 1:
                        continue      # no-op
                    for rev in range(2):
                        if jj > i:
                            lo = i - 1; hi = jj + 1
                        else:
                            lo = jj; hi = i + L
                        if lo < 0:
                            lo = 0
                        if hi > n - 1:
                            hi = n - 1
                        old = seg_cost_current(xy, prime, R, n, lo, hi, pen)
                        new = seg_cost_after_oropt(xy, prime, R, n, i, L, jj, rev, lo, hi, pen)
                        g = old - new
                        if g > best:
                            best = g; bj = jj; bL = L; brev = rev; btype = 1
    # --- 2-opt: reverse [i+1 .. j], j = position of neighbour c of city a=R[i], j>i, j-i<=W ---
    a = R[i]
    if i + 1 < n:
        for kk in range(K):
            c = neigh[a, kk]
            j = pos[c]
            if j <= i + 1 or j - i > W or j >= n:
                continue
            lo = i; hi = j + 1
            if hi > n - 1:
                hi = n - 1
            old = seg_cost_current(xy, prime, R, n, lo, hi, pen)
            # new tour on the window: positions i+1..j reversed
            total = 0.0; prev_city = -1
            for p in range(lo, hi + 1):
                if p <= i or p > j:
                    cc = R[p]
                else:
                    cc = R[j - (p - (i + 1))]
                if prev_city >= 0:
                    d = dist(xy, prev_city, cc)
                    if ((p - 1) % 10 == 9) and prime[prev_city] == 0:
                        d *= (1.0 + pen)
                    total += d
                prev_city = cc
            g = old - total
            if g > best:
                best = g; bj = j; bL = 0; brev = 1; btype = 2
    # --- swap: exchange the cities at positions i and j (KNN neighbour of city a, also neighbours of the predecessor/successor) ---
    for src in range(3):
        if src == 0:
            a0 = R[i]
        elif src == 1:
            a0 = R[i - 1] if i - 1 >= 0 else R[i]
        else:
            a0 = R[i + 1] if i + 1 < n else R[i]
        for kk in range(K):
            c = neigh[a0, kk]
            j = pos[c]
            if j <= 0 or j == i:
                continue
            dd = j - i
            if dd < 0:
                dd = -dd
            if dd > W:
                continue
            lo = i if i < j else j
            hi = i if i > j else j
            lo -= 1; hi += 1
            if lo < 0:
                lo = 0
            if hi > n - 1:
                hi = n - 1
            old = seg_cost_current(xy, prime, R, n, lo, hi, pen)
            total = 0.0; prev_city = -1
            for p in range(lo, hi + 1):
                if p == i:
                    cc = R[j]
                elif p == j:
                    cc = R[i]
                else:
                    cc = R[p]
                if prev_city >= 0:
                    d = dist(xy, prev_city, cc)
                    if ((p - 1) % 10 == 9) and prime[prev_city] == 0:
                        d *= (1.0 + pen)
                    total += d
                prev_city = cc
            g = old - total
            if g > best:
                best = g; bj = j; bL = 0; brev = 0; btype = 3
    # --- prime-slot fill: penalty position (i%10==9) holding a non-prime; insert a prime from position q (|q-i|<=Dp) AT position i ---
    if (i % 10 == 9) and prime[R[i]] == 0:
        Dp = 10
        for q in range(i - Dp, i + Dp + 1):
            if q <= 0 or q >= n or q == i:
                continue
            if prime[R[q]] == 0:
                continue
            # Or-opt L=1: segment [q,q+1) -> after jj; lands at position i when jj=i (q<i) or jj=i-1 (q>i)
            jj = i if q < i else i - 1
            if jj == q - 1:
                continue
            if jj > q:
                lo = q - 1; hi = jj + 1
            else:
                lo = jj; hi = q + 1
            if lo < 0:
                lo = 0
            if hi > n - 1:
                hi = n - 1
            old = seg_cost_current(xy, prime, R, n, lo, hi, pen)
            new = seg_cost_after_oropt(xy, prime, R, n, q, 1, jj, 0, lo, hi, pen)
            g = old - new
            if g > best:
                best = g; bj = q; bL = 1; brev = 0; btype = 4
    out_gain[i] = best if bj >= 0 else 0.0
    out_j[i] = bj; out_L[i] = bL; out_rev[i] = brev; out_type[i] = btype


@njit(cache=True)
def apply_batch(R, order, gain, mj, mL, mrev, mtype, n, W):
    """Greedy: moves in order of decreasing gain, skip when the position window collides with one already used. Returns the count."""
    used = np.zeros(n, dtype=np.uint8)
    applied = 0
    for idx in range(order.shape[0]):
        i = order[idx]
        if gain[i] <= 0:
            break
        j = mj[i]; L = mL[i]; t = mtype[i]
        if t == 4:                                   # prime-slot fill = Or-opt L=1 of the city from position q=j to position i
            q = j; jj = i if q < i else i - 1
            lo = min(q - 1, jj); hi = max(q + 1, jj + 1)
        elif t == 1:
            lo = min(i - 1, j); hi = max(i + L, j + 1)
        elif t == 3:
            lo = min(i, j) - 1; hi = max(i, j) + 1
        else:
            lo = i; hi = j + 1
        if lo < 0: lo = 0
        if hi > n - 1: hi = n - 1
        ok = True
        for p in range(lo, hi + 1):
            if used[p]:
                ok = False; break
        if not ok:
            continue
        for p in range(lo, hi + 1):
            used[p] = 1
        if t == 4:
            q = j; jj = i if q < i else i - 1
            city = R[q]
            if jj > q:
                R[q:jj] = R[q + 1:jj + 1]; R[jj] = city
            else:
                R[jj + 2:q + 1] = R[jj + 1:q]; R[jj + 1] = city
            applied += 1
            continue
        if t == 1:
            seg = R[i:i + L].copy()
            if mrev[i]:
                seg = seg[::-1].copy()
            if j > i:
                R[i:j - L + 1] = R[i + L:j + 1]
                R[j - L + 1:j + 1] = seg
            else:
                R[j + 1 + L:i + L] = R[j + 1:i]
                R[j + 1:j + 1 + L] = seg
        elif t == 3:
            tmp = R[i]; R[i] = R[j]; R[j] = tmp
        else:
            R[i + 1:j + 1] = R[i + 1:j + 1][::-1].copy()
        applied += 1
    return applied


def score_np(R, xy, prime):
    p = np.concatenate([R, R[:1]])
    d = np.sqrt(((xy[p[1:]] - xy[p[:-1]]) ** 2).sum(1))
    k = np.arange(1, len(p)); pen = (k % 10 == 0) & (prime[p[:-1]] == 0)
    return float(d.sum() + PEN * d[pen].sum())


class GpuLS:
    def __init__(self, xy, prime, neigh, W=1000, Lmax=3, tpb=128, Lblk=30):
        self.n = len(xy); self.W = W; self.Lmax = Lmax; self.tpb = tpb; self.Lblk = Lblk
        self.d_xy = cuda.to_device(np.ascontiguousarray(xy, dtype=np.float64))
        self.d_prime = cuda.to_device(prime.astype(np.uint8))
        self.d_neigh = cuda.to_device(neigh.astype(np.int32)); self.K = neigh.shape[1]; self.neigh = np.asarray(neigh)
        n = self.n
        self.d_gain = cuda.device_array(n, np.float64); self.d_j = cuda.device_array(n, np.int32)
        self.d_L = cuda.device_array(n, np.int32); self.d_rev = cuda.device_array(n, np.int32); self.d_type = cuda.device_array(n, np.int32)
        self.d_R = cuda.device_array(n, np.int32); self.d_pos = cuda.device_array(n, np.int32)
        self.xy = xy; self.prime = prime

    def run(self, R, max_rounds=10_000, log=None, min_gain=1e-6, time_limit=None, pen=PEN):
        """pen < PEN = search with a weakened penalty (relaxation); the returned score and "cur" are ALWAYS at the full price PEN."""
        n = self.n; R = R.astype(np.int32).copy(); t0 = time.time()
        blocks = (n + self.tpb - 1) // self.tpb
        cur = score_np(R, self.xy, self.prime); hist = []; best_s = cur; best_R = R.copy()
        for rnd in range(max_rounds):
            pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
            self.d_R.copy_to_device(R); self.d_pos.copy_to_device(pos)
            eval_moves[blocks, self.tpb](self.d_xy, self.d_prime, self.d_R, self.d_pos, self.d_neigh, self.K, n, self.W,
                                         self.Lmax, float(pen), self.Lblk, self.d_gain, self.d_j, self.d_L, self.d_rev, self.d_type)
            cuda.synchronize()
            gain = self.d_gain.copy_to_host(); mj = self.d_j.copy_to_host(); mL = self.d_L.copy_to_host()
            mrev = self.d_rev.copy_to_host(); mtype = self.d_type.copy_to_host()
            cand = int((gain > min_gain).sum())
            if cand == 0:
                break
            order = np.argsort(-gain).astype(np.int64)
            R_prev = R.copy()
            applied = apply_batch(R, order, gain, mj, mL, mrev, mtype, n, self.W)
            new = score_np(R, self.xy, self.prime)
            if pen >= PEN and new > cur + 1e-6:
                # drift: batch with a conflict (rare after big external changes) -> revert and apply ever smaller prefixes
                k = max(1, applied // 8); ok = False
                while k >= 1:
                    R[:] = R_prev; applied = apply_batch(R, order[:k], gain, mj, mL, mrev, mtype, n, self.W)
                    new = score_np(R, self.xy, self.prime)
                    if new <= cur + 1e-6: ok = True; break
                    k //= 2
                if log:
                    i0 = int(order[0]); R[:] = R_prev; apply_batch(R, order[:1], gain, mj, mL, mrev, mtype, n, self.W)
                    d1 = cur - score_np(R, self.xy, self.prime); R[:] = R_prev
                    log(f"round {rnd}: batch DRIFT reverted (prefix {k}, ok={ok}); top move i={i0} type={int(mtype[i0])} j={int(mj[i0])} L={int(mL[i0])} rev={int(mrev[i0])} gain_eval={gain[i0]:.2f} gain_real={d1:.2f}")
                    if ok: apply_batch(R, order[:k], gain, mj, mL, mrev, mtype, n, self.W); new = score_np(R, self.xy, self.prime)
                if not ok:
                    R[:] = R_prev; new = cur; break
            hist.append((rnd, cand, applied, cur - new, new, time.time() - t0))
            if log:
                log(f"round {rnd}: cand={cand} applied={applied} gain={cur - new:.1f} score={new:.1f} t={time.time() - t0:.0f}s")
            if new < best_s:
                best_s = new; best_R = R.copy()
            if abs(cur - new) < min_gain * 10 and applied < 5:
                cur = new; break
            cur = new
            if time_limit and time.time() - t0 > time_limit:
                break
        if pen < PEN:                                   # relaxation: return the best tour AT THE FULL PRICE
            return best_R, best_s, hist
        return R, cur, hist
