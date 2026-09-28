"""Lin-Kernighan chain on the GPU, aware of the "every 10th step" penalty.

LK variant as a sequence of 2-opt moves from a fixed t1 (Johnson-McGeoch): one thread per position a (t1 = R[a]), free end
at position a+1. Step i: for t3 from the KNN of the free end (position c, a+1 < c <= a+W) reverse segment [a+1..c] -> removes
(t1,cur),(t3,nxt), adds (t1,t3),(cur,nxt). LK criterion (on plain distances): G_i = G_{i-1} + d(t1,cur) - d(cur,t3) > 0.
The real cost change WITH PENALTY is computed exactly on the position range [a, c+1] before/after the reversal (positions in
the window do not shift, they only reverse -- the penalty depends on position, so we compute on a copy). Depth <= DEPTH,
branching BR[i] best (by G_i) candidates at level i. The best chain prefix (max sum of gains) is remembered.
Each thread has its own copy of the tour window (scratch[i, 0..W+1]) in global memory.
Result: out_gain[a], out_depth[a], out_c[a, 0..DEPTH-1] (cut positions in order). The CPU applies batches of chains with
disjoint windows [a, a+max(c)+1] as successive reversals.
"""
from __future__ import annotations

import math

import numpy as np
from numba import cuda, njit

DEPTH = 8                        # maximum depth (array size); actual = depth_max argument
DEPTH1 = 9
BR0, BR1, BR2 = 5, 3, 2          # branching at levels 0,1,2; deeper 1


@cuda.jit(device=True, inline=True)
def _d(xy, a, b):
    dx = xy[a, 0] - xy[b, 0]; dy = xy[a, 1] - xy[b, 1]
    return math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True, inline=True)
def _range_cost(xy, prime, S, base, klo, khi, dirn, pen):
    """cost of steps between consecutive elements of the copy S[klo..khi] (copy indices). Element S[k] sits at global
    position base + dirn*k. Step between S[k] and S[k+1]: for dirn=+1 it is the step from position base+k (penalty by S[k]);
    for dirn=-1 it is the step from position base-k-1 (penalty by S[k+1])."""
    total = 0.0
    for k in range(klo, khi):
        a = S[k]; b = S[k + 1]
        d = _d(xy, a, b)
        if dirn > 0:
            p = base + k; src = a
        else:
            p = base - k - 1; src = b
        if (p % 10 == 9) and prime[src] == 0:
            d *= (1.0 + pen)
        total += d
    return total


@cuda.jit(device=True, inline=True)
def _reverse(S, l, r):
    while l < r:
        t = S[l]; S[l] = S[r]; S[r] = t
        l += 1; r -= 1


@cuda.jit
def _rot_left(S, l, r, L):
    """S[l..r] := S[l+L..r] + S[l..l+L-1] (three reversals)."""
    _reverse(S, l, r); _reverse(S, l, r - L); _reverse(S, r - L + 1, r)


@cuda.jit(device=True, inline=True)
def _rot_right(S, l, r, L):
    _reverse(S, l, r); _reverse(S, l, l + L - 1); _reverse(S, l + L, r)


@cuda.jit(device=True, inline=True)
def _apply_step(S, mt, c, L):
    """step in the copy: mt=0 2-opt (reverse S[1..c-1]); mt=1 or-opt: S[1..L] after S[c] (rotate S[1..c] left by L);
    mt=2 reversed or-opt: S[1..L] reversed before S[c] (S[1..c-1] := S[L+1..c-1] + S[L..1])."""
    if mt == 0:
        _reverse(S, 1, c - 1)
    elif mt == 1:
        _rot_left(S, 1, c, L)
    else:
        _reverse(S, 1, c - 1); _reverse(S, 1, c - 1 - L)


@cuda.jit(device=True, inline=True)
def _undo_step(S, mt, c, L):
    if mt == 0:
        _reverse(S, 1, c - 1)
    elif mt == 1:
        _rot_right(S, 1, c, L)
    else:
        _reverse(S, 1, c - 1 - L); _reverse(S, 1, c - 1)


@cuda.jit
def lk_moves(xy, prime, R, pos, neigh, K, n, W, pen, dirn, depth_max, br0, br1, br2, oropt, scratch, out_gain, out_depth, out_c, out_m):
    a = cuda.grid(1)
    if a <= 0 or a >= n:
        return
    base = a                                   # window copy: S[k] = R[a + dirn*k], k = 0..wlen
    if dirn > 0:
        wlen = W + 1
        if a + wlen > n - 1:
            wlen = n - 1 - a
    else:
        wlen = W + 1
        if a - wlen < 1:
            wlen = a - 1                       # position 0 (the pole) is not part of the window
    if wlen < 3:
        out_gain[a] = 0.0; out_depth[a] = 0
        return
    S = scratch[a]
    for k in range(wlen + 1):
        S[k] = R[base + dirn * k]
    t1 = S[0]
    # search state: a stack of levels with candidate counters (DFS with branching BR)
    # level candidates: K neighbours of the free end, we take up to BR_l best by G (simple: scan K in KNN order,
    # take the first BR_l satisfying the criterion -- KNN is already sorted by distance, so these are ~the best G)
    best_gain = 1e-9; best_depth = 0
    for k in range(DEPTH):
        out_c[a, k] = -1; out_m[a, k] = 0
    # iterative DFS: per-level arrays
    cut = cuda.local.array(DEPTH, dtype=np.int32)
    Gl = cuda.local.array(DEPTH1, dtype=np.float64)
    gain_l = cuda.local.array(DEPTH1, dtype=np.float64)
    kidx = cuda.local.array(DEPTH, dtype=np.int32)      # KNN index currently considered at the level
    taken = cuda.local.array(DEPTH, dtype=np.int32)     # how many candidates already examined at the level
    mtl = cuda.local.array(DEPTH, dtype=np.int32)       # step type at the level (0 2-opt, 1 or, 2 reversed or)
    mll = cuda.local.array(DEPTH, dtype=np.int32)       # L of the or-opt segment
    Gl[0] = 0.0; gain_l[0] = 0.0
    level = 0; kidx[0] = 0; taken[0] = 0
    while True:
        if level < 0:
            break
        br = br0 if level == 0 else (br1 if level == 1 else (br2 if level == 2 else 1))
        found = False
        # look for the next candidate at this level
        while kidx[level] < K and taken[level] < br:
            cur = S[1]                                   # free end (position a+1)
            t3 = neigh[cur, kidx[level]]
            kidx[level] += 1
            c = (pos[t3] - base) * dirn                  # index in the copy (if nothing was reversed)
            if c <= 2 or c > wlen:                       # t4 = S[c-1] must be != cur (c-1 >= 2)
                continue
            # after reversals in the copy the position of t3 may have changed -> find it in the copy (level>0)
            if level > 0:
                cc = -1
                for q in range(3, wlen + 1):
                    if S[q] == t3:
                        cc = q; break
                if cc < 0:
                    continue
                c = cc
            g2 = Gl[level] + _d(xy, t1, cur) - _d(xy, cur, t3)
            # step options for this t3: 2-opt (always) + or-opt L=1..3 in 2 variants (when oropt) -- chosen by exact gain
            bopt_g = -1e300; bopt_mt = -1; bopt_L = 0; bopt_G = 0.0; bopt_hi = c
            if g2 > 0.0:
                if pen == 0.0:                               # no penalty: O(1) delta
                    before = _d(xy, S[0], S[1]) + _d(xy, S[c - 1], S[c]); after = _d(xy, S[0], S[c - 1]) + _d(xy, S[1], S[c])
                else:
                    before = _range_cost(xy, prime, S, base, 0, c, dirn, pen)
                    _reverse(S, 1, c - 1)
                    after = _range_cost(xy, prime, S, base, 0, c, dirn, pen)
                    _reverse(S, 1, c - 1)
                bopt_g = before - after; bopt_mt = 0; bopt_L = 0; bopt_G = g2; bopt_hi = c
            if oropt != 0:
                for L in range(1, 4):
                    if c < L + 2:
                        break
                    sL = S[L]; sL1 = S[L + 1]
                    # variant 1: S[1..L] after S[c]; added (cur,t3),(S_L,S_{c+1}),(t1,S_{L+1}); removed (t1,cur),(S_L,S_{L+1}),(S_c,S_{c+1})
                    if c + 1 <= wlen:
                        g = g2 + _d(xy, sL, sL1) - _d(xy, sL, S[c + 1])
                        if g > 0.0:
                            if pen == 0.0:
                                before = _d(xy, S[0], S[1]) + _d(xy, sL, sL1) + _d(xy, S[c], S[c + 1])
                                after = _d(xy, S[0], sL1) + _d(xy, S[c], S[1]) + _d(xy, sL, S[c + 1])
                            else:
                                before = _range_cost(xy, prime, S, base, 0, c + 1, dirn, pen)
                                _rot_left(S, 1, c, L)
                                after = _range_cost(xy, prime, S, base, 0, c + 1, dirn, pen)
                                _rot_right(S, 1, c, L)
                            if before - after > bopt_g:
                                bopt_g = before - after; bopt_mt = 1; bopt_L = L; bopt_G = g; bopt_hi = c + 1
                    # variant 2: S[L..1] before S[c]; added (S_1,S_c),(S_{c-1},S_L),(t1,S_{L+1}); removed (t1,cur),(S_L,S_{L+1}),(S_{c-1},S_c)
                    g = g2 + _d(xy, sL, sL1) - _d(xy, S[c - 1], sL)
                    if g > 0.0:
                        if pen == 0.0:
                            before = _d(xy, S[0], S[1]) + _d(xy, sL, sL1) + _d(xy, S[c - 1], S[c])
                            after = _d(xy, S[0], sL1) + _d(xy, S[c - 1], sL) + _d(xy, S[1], S[c])
                        else:
                            before = _range_cost(xy, prime, S, base, 0, c, dirn, pen)
                            _reverse(S, 1, c - 1); _reverse(S, 1, c - 1 - L)
                            after = _range_cost(xy, prime, S, base, 0, c, dirn, pen)
                            _reverse(S, 1, c - 1 - L); _reverse(S, 1, c - 1)
                        if before - after > bopt_g:
                            bopt_g = before - after; bopt_mt = 2; bopt_L = L; bopt_G = g; bopt_hi = c
            if bopt_mt < 0:
                continue
            taken[level] += 1
            _apply_step(S, bopt_mt, c, bopt_L)
            gtot = gain_l[level] + bopt_g
            cut[level] = c; mtl[level] = bopt_mt; mll[level] = bopt_L
            Gl[level + 1] = bopt_G
            gain_l[level + 1] = gtot
            if gtot > best_gain:
                best_gain = gtot; best_depth = level + 1
                for k in range(level + 1):
                    out_c[a, k] = cut[k]; out_m[a, k] = mtl[k] * 4 + mll[k]
            found = True
            break
        if found and level + 1 < depth_max:
            level += 1; kidx[level] = 0; taken[level] = 0
            continue
        if found:
            # leaf: undo the last step and keep searching at this level
            _undo_step(S, mtl[level], cut[level], mll[level])
            continue
        # level exhausted: undo the step of the level above and go back
        level -= 1
        if level >= 0:
            _undo_step(S, mtl[level], cut[level], mll[level])
    out_gain[a] = best_gain if best_depth > 0 else 0.0
    out_depth[a] = best_depth


@njit(cache=True)
def _rev_pos(R, a, dr, l, r):
    """reverse copy elements with indices l..r -> positions a+dr*l .. a+dr*r."""
    x = a + dr * l; y = a + dr * r
    if x > y:
        x, y = y, x
    while x < y:
        t = R[x]; R[x] = R[y]; R[y] = t
        x += 1; y -= 1


@njit(cache=True)
def apply_lk_batch(R, order, gain, depth, cuts, mvs, dirs, n):
    """Greedy: chains by gain; cuts[a,k] = index c in the copy, mvs[a,k] = type*4+L, dirs[a] = +-1.
    Global windows [a, a+dir*(max c+1)] disjoint; application = successive copy steps mapped onto positions."""
    used = np.zeros(n, dtype=np.uint8)
    applied = 0
    for idx in range(order.shape[0]):
        a = order[idx]
        if gain[a] <= 0:
            break
        d = depth[a]; dr = dirs[a]
        mx = 0
        for k in range(d):
            if cuts[a, k] + 1 > mx:
                mx = cuts[a, k] + 1
        lo = a; hi = a + dr * mx
        if hi < lo:
            lo, hi = hi, lo
        if lo < 0 or hi > n - 1:
            continue
        ok = True
        for p in range(lo, hi + 1):
            if used[p]:
                ok = False; break
        if not ok:
            continue
        for p in range(lo, hi + 1):
            used[p] = 1
        for k in range(d):
            c = cuts[a, k]; mt = mvs[a, k] // 4; L = mvs[a, k] % 4
            if mt == 0:
                _rev_pos(R, a, dr, 1, c - 1)
            elif mt == 1:
                _rev_pos(R, a, dr, 1, c); _rev_pos(R, a, dr, 1, c - L); _rev_pos(R, a, dr, c - L + 1, c)
            else:
                _rev_pos(R, a, dr, 1, c - 1); _rev_pos(R, a, dr, 1, c - 1 - L)
        applied += 1
    return applied


class GpuLK:
    """Rounds of LK chains on a single tour: eval on the GPU (lk_moves), batch on the CPU (apply_lk_batch)."""

    def __init__(self, ls, W=1000, tpb=64, depth=5, br=(5, 3, 2), oropt=False):
        self.ls = ls; self.W = W; self.tpb = tpb; n = ls.n; self.depth = min(depth, DEPTH); self.br = br; self.oropt = 1 if oropt else 0
        self.d_m = cuda.device_array((n, DEPTH), np.int32)
        self.d_scratch = cuda.device_array((n, W + 2), np.int32)
        self.d_gain = cuda.device_array(n, np.float64); self.d_depth = cuda.device_array(n, np.int32)
        self.d_c = cuda.device_array((n, DEPTH), np.int32)

    def eval(self, R, pen=0.1):
        """Two passes (window to the right and to the left of a); per position the better chain. Returns gain, depth, cuts, dirs."""
        ls = self.ls; n = ls.n
        pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
        ls.d_R.copy_to_device(R.astype(np.int32)); ls.d_pos.copy_to_device(pos)
        blocks = (n + self.tpb - 1) // self.tpb
        res = []
        for dirn in (1, -1):
            lk_moves[blocks, self.tpb](ls.d_xy, ls.d_prime, ls.d_R, ls.d_pos, ls.d_neigh, ls.K, n, self.W, float(pen),
                                       dirn, self.depth, self.br[0], self.br[1], self.br[2], self.oropt,
                                       self.d_scratch, self.d_gain, self.d_depth, self.d_c, self.d_m)
            cuda.synchronize()
            res.append((self.d_gain.copy_to_host(), self.d_depth.copy_to_host(), self.d_c.copy_to_host(), self.d_m.copy_to_host()))
        g1, d1, c1, m1 = res[0]; g2, d2, c2, m2 = res[1]
        use2 = g2 > g1
        gain = np.where(use2, g2, g1); depth = np.where(use2, d2, d1)
        cuts = np.where(use2[:, None], c2, c1); mvs = np.where(use2[:, None], m2, m1); dirs = np.where(use2, -1, 1).astype(np.int32)
        return gain, depth, cuts, mvs, dirs

    def run(self, R, max_rounds=1000, log=None, min_gain=1e-6, time_limit=None, pen=0.1):
        import time
        from domains.santa.gpu_ls import PEN, score_np
        n = self.ls.n; R = R.astype(np.int32).copy(); t0 = time.time()
        cur = score_np(R, self.ls.xy, self.ls.prime); hist = []; best_s = cur; best_R = R.copy()
        for rnd in range(max_rounds):
            gain, depth, cuts, mvs, dirs = self.eval(R, pen)
            cand = int((gain > min_gain).sum())
            if cand == 0:
                break
            order = np.argsort(-gain).astype(np.int64)
            applied = apply_lk_batch(R, order, gain, depth, cuts, mvs, dirs, n)
            new = score_np(R, self.ls.xy, self.ls.prime)
            hist.append((rnd, cand, applied, cur - new, new, time.time() - t0))
            if log:
                dd = depth[gain > min_gain]
                log(f"lk {rnd}: cand={cand} applied={applied} gain={cur - new:.1f} score={new:.1f} "
                    f"depth_mean={dd.mean():.2f} t={time.time() - t0:.0f}s")
            if pen >= PEN and new > cur + 1e-6:
                raise RuntimeError(f"score increased {cur}->{new}: bug in LK apply/eval")
            if new < best_s:
                best_s = new; best_R = R.copy()
            if abs(cur - new) < min_gain * 10 and applied < 5:
                cur = new; break
            cur = new
            if time_limit and time.time() - t0 > time_limit:
                break
        if pen < PEN:
            return best_R, best_s, hist
        return R, cur, hist
