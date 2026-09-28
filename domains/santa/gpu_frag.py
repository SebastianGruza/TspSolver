"""Form 2 (Sebastian's idea: "windows of ~100 with a heuristic"): the tour is split into fragments of F positions with frozen
ends; ONE GPU BLOCK per fragment, fragment in shared memory, inside a mini-ILS: double-bridge kick (inside the
fragment) -> 2-opt + Or-opt (L<=3, both directions) to a local optimum with the penalty computed EXACTLY (fragment positions
are fixed: absolute position = s + t) -> acceptance "better than best" (return to best). 128 threads evaluate moves in
parallel (thread = position i / segment a), thread 0 reduces, all threads build the new order through a virtual index.
Fragments are disjoint -> improvements applied at once; subsequent passes with a shifted split (offset).
Cost: O(F) per move evaluation (recomputing the touched range), ~kicks x rounds x F^2/2 evaluations per fragment.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda

F = 128          # default fragment length = number of threads per block (128 or 256)
ORW = 32         # Or-opt window inside the fragment (|a-b| <= ORW)
XW = 16          # segment-exchange window (Or-3opt) |a-b| <= XW


@cuda.jit(device=True, inline=True)
def _dd(xs, ys, u, v):
    dx = xs[u] - xs[v]; dy = ys[u] - ys[v]
    return math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True, inline=True)
def _stp(xs, ys, pr, u, v, p, pen):
    dx = xs[u] - xs[v]; dy = ys[u] - ys[v]
    d = math.sqrt(dx * dx + dy * dy)
    if (p % 10 == 9) and pr[u] == 0:
        d *= (1.0 + pen)
    return d


@cuda.jit(device=True)
def _slot_after(S, mt, m0, m1, m2, m3, t):
    """slot at position t AFTER the move (mt,m0..m3): 0 = 2-opt(i,j): reverse [i+1..j]; 1 = Or-opt(a,L,b,rev): segment [a,a+L)
    after position b; 2 = double-bridge(p,q,r)."""
    if mt == 0:
        i = m0; j = m1
        if t > i and t <= j:
            return S[i + j + 1 - t]
        return S[t]
    if mt == 1:
        a = m0; L = m1; b = m2; rev = m3
        if b >= a + L:
            if t < a or t > b:
                return S[t]
            if t < a + (b - a + 1 - L):
                return S[t + L]
            k = t - (b + 1 - L)
            return S[a + L - 1 - k] if rev else S[a + k]
        else:
            if t <= b or t >= a + L:
                return S[t]
            k = t - (b + 1)
            if k < L:
                return S[a + L - 1 - k] if rev else S[a + k]
            return S[t - L]
    if mt == 2:
        p = m0; q = m1; r = m2
        if t < p or t >= r:
            return S[t]
        if t < p + (r - q):
            return S[q + (t - p)]
        return S[p + (t - (p + r - q))]
    if mt == 3:
        # exchange of segments A=[a,a+L1) and B=[b,b+L2), a<b, rev bit0 = A, bit1 = B (Or-3opt inside the fragment)
        a = m0; L1 = m1 & 7; b = m2; L2 = (m1 >> 3) & 7; rev = m3
        if t < a or t >= b + L2:
            return S[t]
        if t < a + L2:
            k = t - a
            return S[b + L2 - 1 - k] if (rev & 2) else S[b + k]
        if t < b + L2 - L1:
            return S[t - L2 + L1]
        k = t - (b + L2 - L1)
        return S[a + L1 - 1 - k] if (rev & 1) else S[a + k]
    # mt == 4: reversal of [p..q] (a kick that shifts the penalty grid)
    p = m0; q = m1
    if t < p or t > q:
        return S[t]
    return S[p + q - t]


@cuda.jit(device=True)
def _span_new(S, xs, ys, pr, s0, lo, hi, mt, m0, m1, m2, m3, pen):
    """cost of steps lo..hi-1 after the move (position t -> t+1)."""
    c = 0.0
    prev = _slot_after(S, mt, m0, m1, m2, m3, lo)
    for t in range(lo + 1, hi + 1):
        nxt = _slot_after(S, mt, m0, m1, m2, m3, t)
        c += _stp(xs, ys, pr, prev, nxt, s0 + t - 1, pen)
        prev = nxt
    return c


@cuda.jit(device=True)
def _span_old(S, xs, ys, pr, s0, lo, hi, pen):
    c = 0.0
    for t in range(lo, hi):
        c += _stp(xs, ys, pr, S[t], S[t + 1], s0 + t, pen)
    return c


def make_kernel(F, ORW, XW):
  @cuda.jit
  def frag_ils(xy, prime, R, n, offset, pen, kicks, max_rounds, seed, use_x, nfrag, out_gain, out_order):
      w = cuda.blockIdx.x; tid = cuda.threadIdx.x
      fr = w % nfrag; cp = w // nfrag                  # fragment copies with a different seed (small n: fill the GPU)
      s0 = offset + fr * F
      S = cuda.shared.array(F, np.int32); T = cuda.shared.array(F, np.int32); B = cuda.shared.array(F, np.int32)
      xs = cuda.shared.array(F, np.float64); ys = cuda.shared.array(F, np.float64); pr = cuda.shared.array(F, np.uint8)
      g = cuda.shared.array(F, np.float64); mv = cuda.shared.array((F, 5), np.int32)
      ctl = cuda.shared.array(8, np.int32)      # 0: type of move to apply, 1..4: parameters, 5: continue flag, 6: rng
      val = cuda.shared.array(4, np.float64)    # 0: cur, 1: best
      if s0 < 1 or s0 + F > n:
          if tid == 0:
              out_gain[w] = 0.0
          return
      c = R[s0 + tid]
      S[tid] = tid; B[tid] = tid; xs[tid] = xy[c, 0]; ys[tid] = xy[c, 1]; pr[tid] = prime[c]
      if tid == 0:
          ctl[6] = (seed * 2654435761 + fr * 40503 + cp * 7919 + 12345) & 0x7fffffff
          if ctl[6] == 0: ctl[6] = 1
      cuda.syncthreads()
      if tid == 0:
          val[0] = _span_old(S, xs, ys, pr, s0, 0, F - 1, pen); val[1] = val[0]
      cuda.syncthreads()
      init = val[1]
      for kick in range(kicks + 1):
          if kick > 0:
              # kick: double-bridge p<q<r in [1, F-1); thread 0 draws (xorshift32), all threads build T
              if tid == 0:
                  x = ctl[6]
                  x ^= (x << 13) & 0x7fffffff; x ^= (x >> 17); x ^= (x << 5) & 0x7fffffff
                  p = 1 + x % (F - 4); x ^= (x << 13) & 0x7fffffff; x ^= (x >> 17); x ^= (x << 5) & 0x7fffffff
                  q = p + 1 + x % (F - 2 - p); x ^= (x << 13) & 0x7fffffff; x ^= (x >> 17); x ^= (x << 5) & 0x7fffffff
                  r = q + 1 + x % (F - 1 - q) if q < F - 2 else F - 1
                  ctl[6] = x
                  if kick % 2 == 1:
                      ctl[0] = 2; ctl[1] = p; ctl[2] = q; ctl[3] = r; ctl[4] = 0
                  else:                                   # reversal of [p..q]: shifts the penalty grid inside the segment
                      ctl[0] = 4; ctl[1] = p; ctl[2] = q; ctl[3] = 0; ctl[4] = 0
              cuda.syncthreads()
              T[tid] = _slot_after(S, ctl[0], ctl[1], ctl[2], ctl[3], ctl[4], tid)
              cuda.syncthreads()
              S[tid] = T[tid]
              cuda.syncthreads()
              if tid == 0:
                  val[0] = _span_old(S, xs, ys, pr, s0, 0, F - 1, pen)
              cuda.syncthreads()
          # local search: best move per round
          for rnd in range(max_rounds):
              bg = 1e-7; bt = -1; b0 = 0; b1 = 0; b2 = 0; b3 = 0
              i = tid
              if i <= F - 3:                                          # 2-opt: reverse [i+1..j], j <= F-2 (ends frozen)
                  if pen == 0.0:                                          # no penalty: O(1) delta (4 distances)
                      si = S[i]; si1 = S[i + 1]; dab = _dd(xs, ys, si, si1)
                      for j in range(i + 2, F - 1):
                          sj = S[j]; sj1 = S[j + 1]
                          gg = dab + _dd(xs, ys, sj, sj1) - _dd(xs, ys, si, sj) - _dd(xs, ys, si1, sj1)
                          if gg > bg:
                              bg = gg; bt = 0; b0 = i; b1 = j; b2 = 0; b3 = 0
                  else:
                      for j in range(i + 2, F - 1):
                          old = _span_old(S, xs, ys, pr, s0, i, j + 1, pen)
                          new = _span_new(S, xs, ys, pr, s0, i, j + 1, 0, i, j, 0, 0, pen)
                          if old - new > bg:
                              bg = old - new; bt = 0; b0 = i; b1 = j; b2 = 0; b3 = 0
              a = tid
              if a >= 1:
                  for L in range(1, 4):
                      if a + L > F - 1:
                          break
                      lo_b = a - ORW if a - ORW > 0 else 0
                      hi_b = a + L - 1 + ORW if a + L - 1 + ORW < F - 2 else F - 2
                      for b in range(lo_b, hi_b + 1):
                          if b >= a - 1 and b <= a + L - 1:
                              continue
                          if pen == 0.0:                                  # no penalty: O(1) delta
                              p0 = S[a - 1]; p1 = S[a]; p2 = S[a + L - 1]; p3 = S[a + L]; q0 = S[b]; q1 = S[b + 1]
                              rem = _dd(xs, ys, p0, p1) + _dd(xs, ys, p2, p3) - _dd(xs, ys, p0, p3); base = _dd(xs, ys, q0, q1)
                              g0 = rem - (_dd(xs, ys, q0, p1) + _dd(xs, ys, p2, q1) - base)
                              g1 = rem - (_dd(xs, ys, q0, p2) + _dd(xs, ys, p1, q1) - base)
                              if g0 > bg:
                                  bg = g0; bt = 1; b0 = a; b1 = L; b2 = b; b3 = 0
                              if g1 > bg:
                                  bg = g1; bt = 1; b0 = a; b1 = L; b2 = b; b3 = 1
                              continue
                          if b >= a + L:
                              lo = a - 1; hi = b + 1
                          else:
                              lo = b; hi = a + L
                          old = _span_old(S, xs, ys, pr, s0, lo, hi, pen)
                          for rev in range(2):
                              new = _span_new(S, xs, ys, pr, s0, lo, hi, 1, a, L, b, rev, pen)
                              if old - new > bg:
                                  bg = old - new; bt = 1; b0 = a; b1 = L; b2 = b; b3 = rev
              if use_x and a >= 1:                                # exchange of segments A=[a,a+L1) with B=[b,b+L2), b > a+L1-1
                  for L1 in range(1, 4):
                      if a + L1 > F - 1:
                          break
                      hi_b = a + L1 + XW if a + L1 + XW < F - 1 else F - 2
                      for b in range(a + L1, hi_b + 1):
                          for L2 in range(1, 4):
                              if b + L2 > F - 1:
                                  break
                              if L1 == L2 and b == a + L1:
                                  pass
                              lo = a - 1; hi = b + L2
                              old = _span_old(S, xs, ys, pr, s0, lo, hi, pen)
                              m1 = L1 | (L2 << 3)
                              for rev in range(4):
                                  new = _span_new(S, xs, ys, pr, s0, lo, hi, 3, a, m1, b, rev, pen)
                                  if old - new > bg:
                                      bg = old - new; bt = 3; b0 = a; b1 = m1; b2 = b; b3 = rev
              g[tid] = bg if bt >= 0 else 0.0
              mv[tid, 0] = bt; mv[tid, 1] = b0; mv[tid, 2] = b1; mv[tid, 3] = b2; mv[tid, 4] = b3
              cuda.syncthreads()
              if tid == 0:
                  bi = -1; bb = 1e-7
                  for t in range(F):
                      if g[t] > bb:
                          bb = g[t]; bi = t
                  if bi < 0:
                      ctl[5] = 0
                  else:
                      ctl[5] = 1; ctl[0] = mv[bi, 0]; ctl[1] = mv[bi, 1]; ctl[2] = mv[bi, 2]; ctl[3] = mv[bi, 3]; ctl[4] = mv[bi, 4]
                      val[0] -= bb
              cuda.syncthreads()
              if ctl[5] == 0:
                  break
              T[tid] = _slot_after(S, ctl[0], ctl[1], ctl[2], ctl[3], ctl[4], tid)
              cuda.syncthreads()
              S[tid] = T[tid]
              cuda.syncthreads()
          # acceptance: better than best -> new best; otherwise return to best
          if tid == 0:
              val[0] = _span_old(S, xs, ys, pr, s0, 0, F - 1, pen)   # exact, no drift of running sums
          cuda.syncthreads()
          if val[0] < val[1] - 1e-9:
              if tid == 0:
                  val[1] = val[0]
              B[tid] = S[tid]
          else:
              S[tid] = B[tid]
          cuda.syncthreads()
      if tid == 0:
          out_gain[w] = init - val[1]
      out_order[w, tid] = R[s0 + B[tid]]


  return frag_ils


_KERNELS = {}


class GpuFrag:
    def __init__(self, ls, kicks=30, max_rounds=60, seed=1, F=128, use_x=False, orw=None, xw=None, copies=1):
        self.ls = ls; self.kicks = kicks; self.max_rounds = max_rounds; self.seed = seed; self.F = F; self.use_x = use_x; self.copies = max(1, copies)
        key = (F, orw or ORW * F // 128, xw or XW)
        if key not in _KERNELS:
            _KERNELS[key] = make_kernel(*key)
        self.kernel = _KERNELS[key]
        n = ls.n; self.nw = (n // F + 1) * self.copies
        self.d_gain = cuda.device_array(self.nw, np.float64)
        self.d_order = cuda.device_array((self.nw, F), np.int32)

    def pass_offset(self, R, offset, pen=0.1):
        ls = self.ls; n = ls.n
        ls.d_R.copy_to_device(R.astype(np.int32))
        F = self.F; nfrag = (n - offset) // F; nb = nfrag * self.copies
        if nfrag <= 0: return 0, 0.0
        self.kernel[nb, F](ls.d_xy, ls.d_prime, ls.d_R, n, offset, float(pen), self.kicks, self.max_rounds, self.seed,
                           self.use_x, nfrag, self.d_gain, self.d_order)
        cuda.synchronize()
        gain = self.d_gain.copy_to_host()[:nb].reshape(self.copies, nfrag); order = self.d_order.copy_to_host()[:nb].reshape(self.copies, nfrag, F)
        best_cp = gain.argmax(0); gain = gain[best_cp, np.arange(nfrag)]; order = order[best_cp, np.arange(nfrag)]
        idx = np.where(gain > 1e-9)[0]
        for w in idx:
            s = offset + w * F
            R[s:s + F] = order[w]
        self.seed += 1
        return len(idx), float(gain[idx].sum())

    def run(self, R, sweeps=1, pen=0.1, log=None, offsets=None):
        F = self.F
        if offsets is None:
            offsets = (1, 1 + F // 2, 1 + F // 4, 1 + 3 * F // 4)
        from domains.santa.gpu_ls import score_np
        R = R.astype(np.int32).copy(); t0 = time.time(); hist = []
        cur = score_np(R, self.ls.xy, self.ls.prime)
        for sw in range(sweeps):
            for off in offsets:
                ap, g = self.pass_offset(R, off, pen)
                new = score_np(R, self.ls.xy, self.ls.prime)
                hist.append((sw, off, ap, cur - new, new, time.time() - t0))
                if log:
                    log(f"frag F={F} k={self.kicks}{'x' if self.use_x else ''} off={off}: improved={ap} gain={cur - new:.1f} (gpu:{g:.1f}) score={new:.1f} t={time.time() - t0:.0f}s")
                if pen >= 0.1 and new > cur + 1e-6:
                    raise RuntimeError(f"frag: score increased {cur}->{new}")
                cur = new
        return R, cur, hist
