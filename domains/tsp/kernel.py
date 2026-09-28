"""Population-based memetic GA on GPU (Numba, cooperative groups).

Extends ILS with a population structure as in the original:
- pm INDIVIDUALS per thread (island); OX crossover between them + elitist selection;
- COLONIES: T threads in C colonies; migration within a colony (adoption of the best
  neighbor) -- race-free: migrant buffer + two grid.sync phases;
- local memetic: 2-opt <-> Or-opt on neighbor lists; double-bridge on the weakest.

Tours flat: R[M, n], M=T*pm; thread gid owns individuals [gid*pm .. gid*pm+pm).
P/scratch/existed/states -- per thread (gid), reused sequentially across individuals.
"""
import os
import numpy as np
from numba import cuda, int32
from engine.gpu import rnd01


@cuda.jit(device=True, inline=True)
def route_len(D, R, ri, n):
    s = 0
    for i in range(n - 1):
        s += D[R[ri, i], R[ri, i + 1]]
    s += D[R[ri, n - 1], R[ri, 0]]
    return s


@cuda.jit(device=True, inline=True)
def two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, max_sweeps):
    for _sweep in range(max_sweeps):
        improved = False
        for i in range(n - 1):
            a = R[ri, i]; b = R[ri, i + 1]; dab = D[a, b]
            for kk in range(Knl):
                c = neigh[a, kk]; dac = D[a, c]
                if dac >= dab:
                    break
                j = P[pi, c]
                if j <= i:
                    continue
                jn = j + 1 if j + 1 < n else 0
                d = R[ri, jn]
                if dab + D[c, d] - dac - D[b, d] > 0:
                    lo = i + 1; hi = j
                    while lo < hi:
                        cl = R[ri, lo]; ch = R[ri, hi]
                        R[ri, lo] = ch; R[ri, hi] = cl
                        P[pi, ch] = lo; P[pi, cl] = hi
                        lo += 1; hi -= 1
                    improved = True
                    b = R[ri, i + 1]; dab = D[a, b]
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def relocate_city(R, P, ri, pi, n, src, after):
    city = R[ri, src]
    if after > src:
        for k in range(src, after):
            R[ri, k] = R[ri, k + 1]; P[pi, R[ri, k]] = k
        R[ri, after] = city; P[pi, city] = after
    else:
        for k in range(src, after + 1, -1):
            R[ri, k] = R[ri, k - 1]; P[pi, R[ri, k]] = k
        R[ri, after + 1] = city; P[pi, city] = after + 1


@cuda.jit(device=True, inline=True)
def or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, max_sweeps):
    for _sweep in range(max_sweeps):
        improved = False
        for i in range(n):
            a = R[ri, i]
            ip = i - 1 if i > 0 else n - 1
            isn = i + 1 if i + 1 < n else 0
            p = R[ri, ip]; s = R[ri, isn]
            remove_gain = D[p, a] + D[a, s] - D[p, s]
            if remove_gain <= 0:
                continue
            for kk in range(Knl):
                c = neigh[a, kk]; jc = P[pi, c]
                if jc == i or jc == ip:
                    continue
                jd = jc + 1 if jc + 1 < n else 0
                d = R[ri, jd]
                if (D[c, a] + D[a, d] - D[c, d]) - remove_gain < 0:
                    relocate_city(R, P, ri, pi, n, i, jc)
                    improved = True
                    break
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def relocate_segment(R, P, ri, pi, n, i, L, after, rev):
    """Move segment [i, i+L) right AFTER position 'after' (outside the segment). rev=1 => reversed. L<=16."""
    seg = cuda.local.array(16, int32)
    for t in range(L):
        seg[t] = R[ri, i + t]
    if rev == 1:
        lo = 0; hi = L - 1
        while lo < hi:
            tmp = seg[lo]; seg[lo] = seg[hi]; seg[hi] = tmp
            lo += 1; hi -= 1
    if after >= i + L:                              # insertion after the segment: shift [i+L..after] left by L
        k = i; j = i + L
        while j <= after:
            R[ri, k] = R[ri, j]; P[pi, R[ri, k]] = k
            k += 1; j += 1
        for t in range(L):
            R[ri, k + t] = seg[t]; P[pi, seg[t]] = k + t
    else:                                           # after < i-1: shift [after+1..i-1] right by L
        k = i + L - 1; j = i - 1
        while j > after:
            R[ri, k] = R[ri, j]; P[pi, R[ri, k]] = k
            k -= 1; j -= 1
        for t in range(L):
            R[ri, after + 1 + t] = seg[t]; P[pi, seg[t]] = after + 1 + t


@cuda.jit(device=True, inline=True)
def or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, L, max_sweeps):
    """Or-opt of an L-city segment, KNN-guided: forward next to KNN(a), reversed next to KNN(z). O(n*K)/sweep."""
    for _sweep in range(max_sweeps):
        improved = False
        for i in range(1, n - L):                   # segment [i,i+L); prev=i-1, next=i+L
            a = R[ri, i]; z = R[ri, i + L - 1]
            ip = i - 1
            prev = R[ri, ip]; nxt = R[ri, i + L]
            remove_gain = D[prev, a] + D[z, nxt] - D[prev, nxt]
            if remove_gain <= 0:
                continue
            best_delta = 0; best_after = -1; best_rev = 0
            for kk in range(Knl):                   # forward: c close to a (new edge c-a)
                c = neigh[a, kk]; jc = P[pi, c]
                if ip <= jc <= i + L - 1:            # c==prev or inside the segment => skip
                    continue
                d = R[ri, jc + 1 if jc + 1 < n else 0]
                delta = (D[c, a] + D[z, d] - D[c, d]) - remove_gain
                if delta < best_delta:
                    best_delta = delta; best_after = jc; best_rev = 0
            for kk in range(Knl):                   # reversed: c close to z (new edge c-z)
                c = neigh[z, kk]; jc = P[pi, c]
                if ip <= jc <= i + L - 1:
                    continue
                d = R[ri, jc + 1 if jc + 1 < n else 0]
                delta = (D[c, z] + D[a, d] - D[c, d]) - remove_gain
                if delta < best_delta:
                    best_delta = delta; best_after = jc; best_rev = 1
            if best_after >= 0:
                relocate_segment(R, P, ri, pi, n, i, L, best_after, best_rev)
                improved = True
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def swap_neighbor(D, neigh, Knl, R, P, ri, pi, n, n_long, max_sweeps):
    """Vertex swap TARGETED at long edges (from Java: mutVertexSwap, improved): city to swap =
    endpoint of one of the n_long longest edges; swap with a close neighbor (KNN). 4 edges, no rev.
    O(n) to find the long ones + O(n_long*K) for the swaps. n_long<=16."""
    bp = cuda.local.array(16, int32)                     # positions of the top-n_long longest edges
    bl = cuda.local.array(16, int32)
    for _sweep in range(max_sweeps):
        for t in range(n_long):                          # find the n_long longest edges (i -> i+1)
            bp[t] = -1; bl[t] = -1
        for i in range(n):
            j = i + 1 if i + 1 < n else 0
            d = D[R[ri, i], R[ri, j]]
            t = n_long - 1
            if d > bl[t]:
                bl[t] = d; bp[t] = i
                while t > 0 and bl[t] > bl[t - 1]:
                    tl = bl[t]; bl[t] = bl[t - 1]; bl[t - 1] = tl
                    tp = bp[t]; bp[t] = bp[t - 1]; bp[t - 1] = tp
                    t -= 1
        improved = False
        for t in range(n_long):                          # swap a long-edge city with a close neighbor
            i = bp[t]
            if i <= 0 or i >= n - 1:                      # edges/wrap => skip
                continue
            a = R[ri, i]; ap = R[ri, i - 1]; an = R[ri, i + 1]
            best_delta = 0; best_j = -1
            for kk in range(Knl):
                c = neigh[a, kk]; jj = P[pi, c]
                if jj <= i + 1 and jj >= i - 1:           # same/adjacent pos. => skip
                    continue
                if jj == 0 or jj == n - 1:
                    continue
                cp = R[ri, jj - 1]; cn = R[ri, jj + 1]
                old = D[ap, a] + D[a, an] + D[cp, c] + D[c, cn]
                new = D[ap, c] + D[c, an] + D[cp, a] + D[a, cn]
                delta = new - old
                if delta < best_delta:
                    best_delta = delta; best_j = jj
            if best_j >= 0:
                jj = best_j; c = R[ri, jj]
                R[ri, i] = c; P[pi, c] = i
                R[ri, jj] = a; P[pi, a] = jj
                improved = True
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def or_opt_seg_var(D, neigh, Knl, R, P, ri, pi, n, max_sweeps):
    """Relocation of LONGER segments (from Java: mutSegmentRelocation, variable length) -- len 5 and 10,
    covers the medium/long gap beyond the rigid Or-2/Or-3. KNN-guided (via or_opt_seg)."""
    or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 5, max_sweeps)
    or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 10, max_sweeps)


@cuda.jit(device=True, inline=True)
def three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, max_sweeps):
    """3-opt neighbor-guided (type from the original: reverse [i+1..j] + [j+1..k],
    new edges a-c, b-e, d-f). c in neighbors(a), e in neighbors(b), pruning on a-c."""
    for _sweep in range(max_sweeps):
        improved = False
        for i in range(n - 2):
            a = R[ri, i]; b = R[ri, i + 1]; dab = D[a, b]
            moved = False
            for k1 in range(Knl):
                c = neigh[a, k1]; dac = D[a, c]
                if dac >= dab:
                    break
                j = P[pi, c]
                if j <= i or j >= n - 1:
                    continue
                d = R[ri, j + 1]
                base = dab + D[c, d] - dac        # (dab+dcd+def) - (dac+dbe+ddf), constant part
                for k2 in range(Knl):
                    e = neigh[b, k2]; k = P[pi, e]
                    if k <= j or k >= n - 1:
                        continue
                    f = R[ri, k + 1]
                    if base + D[e, f] - D[b, e] - D[d, f] > 0:
                        lo = i + 1; hi = j
                        while lo < hi:
                            cl = R[ri, lo]; ch = R[ri, hi]
                            R[ri, lo] = ch; R[ri, hi] = cl
                            P[pi, ch] = lo; P[pi, cl] = hi
                            lo += 1; hi -= 1
                        lo = j + 1; hi = k
                        while lo < hi:
                            cl = R[ri, lo]; ch = R[ri, hi]
                            R[ri, lo] = ch; R[ri, hi] = cl
                            P[pi, ch] = lo; P[pi, cl] = hi
                            lo += 1; hi -= 1
                        improved = True; moved = True
                        break
                if moved:
                    break
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, n_long, max_sweeps):
    """Targeted repair: find ~n_long longest edges of the tour and insert into each
    gap (u,v) the best-fitting city from KNN(u) U KNN(v) when it lowers the objective. Correctness:
    I keep the edge CITIES (not positions) + check that the edge still exists after moves."""
    le_u = cuda.local.array(16, int32)
    le_v = cuda.local.array(16, int32)
    le_len = cuda.local.array(16, int32)
    if n_long > 16:
        n_long = 16
    for _sweep in range(max_sweeps):
        cnt = 0                                       # 1. top n_long longest edges
        for i in range(n):
            iu = R[ri, i]; iv = R[ri, i + 1 if i + 1 < n else 0]
            el = D[iu, iv]
            if cnt < n_long:
                le_u[cnt] = iu; le_v[cnt] = iv; le_len[cnt] = el; cnt += 1
            else:
                mi = 0; mv = le_len[0]
                for t in range(1, n_long):
                    if le_len[t] < mv:
                        mv = le_len[t]; mi = t
                if el > mv:
                    le_u[mi] = iu; le_v[mi] = iv; le_len[mi] = el
        applied = False                               # 2. insert the best city into each gap
        for t in range(cnt):
            u = le_u[t]; v = le_v[t]
            p = P[pi, u]
            pv = p + 1 if p + 1 < n else 0
            if R[ri, pv] != v:                        # edge already changed earlier -> skip
                continue
            duv = D[u, v]
            best_delta = 0; best_c = -1
            for src in range(2):
                anchor = u if src == 0 else v
                for kk in range(Knl):
                    c = neigh[anchor, kk]
                    if c == u or c == v:
                        continue
                    jc = P[pi, c]
                    jp = jc - 1 if jc > 0 else n - 1
                    jn = jc + 1 if jc + 1 < n else 0
                    gain = D[R[ri, jp], c] + D[c, R[ri, jn]] - D[R[ri, jp], R[ri, jn]]
                    delta = (D[u, c] + D[c, v] - duv) - gain
                    if delta < best_delta:
                        best_delta = delta; best_c = jc
            if best_c >= 0:
                relocate_city(R, P, ri, pi, n, best_c, P[pi, u])   # insert c right after u
                applied = True
        if not applied:
            break


@cuda.jit(device=True, inline=True)
def local_search(D, neigh, Knl, R, P, ri, pi, n, sweeps, ls_mode, le_nlong, opt_on, ops):
    for i in range(n):
        P[pi, R[ri, i]] = i
    if ls_mode == 0:                                   # A: classic 2-opt + Or-1 + 3-opt
        for _r in range(2):
            two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
    elif ls_mode == 1:                                 # B: relocations WITHOUT 2/3-opt
        for _r in range(3):
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # Or-1
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)      # Or-2 (+reversed)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)      # Or-3 (+reversed)
            if le_nlong > 0:                                           # long-edge gated by phase
                long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
    elif ls_mode == 2:                                 # C: hybrid -- classic from the start, relocations Or-2/3+long-edge TOGETHER from opt_start
        for _r in range(2):                            # skeleton: 2-opt + Or-1 + 3-opt (untangling)
            two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # Or-1
        three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)        # Or-1
        if opt_on == 1:                                # relocations join TOGETHER in the late phase (endgame)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)      # Or-2 (+reversed)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)      # Or-3 (+reversed)
            if le_nlong > 0:
                long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
    elif ls_mode == 3:                                 # D: ultra-fast -- only 2-opt + Or-1
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
    else:                                              # E (ls_mode==9): BITMASK -- fine-grained ladder
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)   # base: 2-opt
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # base: Or-1
        if ops & 4:                                                # +Or-2
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)
        if ops & 8:                                                # +Or-3
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)
        if ops & 16:                                               # +3-opt
            three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        if (ops & 32) and le_nlong > 0:                            # +long-edge
            long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
        if ops & 64:                                               # +vertex swap targeted at long edges
            swap_neighbor(D, neigh, Knl, R, P, ri, pi, n, 10, sweeps)
        if ops & 128:                                              # +long segments 5/10 (from Java)
            or_opt_seg_var(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        if ops != 0:                                               # cleanup when an operator was added
            two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)


@cuda.jit(device=True, inline=True)
def double_bridge(R, S, ri, si, n, states, sti):
    a = 1 + int(rnd01(states, sti) * (n - 3))
    b = a + 1 + int(rnd01(states, sti) * (n - a - 2))
    c = b + 1 + int(rnd01(states, sti) * (n - b - 1))
    if b <= a:
        b = a + 1
    if c <= b:
        c = b + 1
    if c >= n:
        c = n - 1
    idx = 0
    for i in range(0, a):
        S[si, idx] = R[ri, i]; idx += 1
    for i in range(b, c):
        S[si, idx] = R[ri, i]; idx += 1
    for i in range(a, b):
        S[si, idx] = R[ri, i]; idx += 1
    for i in range(c, n):
        S[si, idx] = R[ri, i]; idx += 1
    for i in range(n):
        R[ri, i] = S[si, i]


@cuda.jit(device=True, inline=True)
def ox_crossover(R, rp1, rp2, CH, rc, existed, pi, n, states, sti):
    """Order Crossover: segment [cut,cut+L) from parent1, the rest in parent2 order."""
    cut = int(rnd01(states, sti) * n)
    L = int(rnd01(states, sti) * 0.4 * n) + int(0.3 * n) + 1
    if L >= n:
        L = n - 1
    for i in range(n):
        existed[pi, i] = 0
    for k in range(L):
        pos = (cut + k) % n
        city = R[rp1, pos]
        CH[rc, pos] = city
        existed[pi, city] = 1
    fillpos = (cut + L) % n
    for k in range(n):
        pos2 = (cut + L + k) % n
        city = R[rp2, pos2]
        if existed[pi, city] == 0:
            CH[rc, fillpos] = city
            existed[pi, city] = 1
            fillpos = (fillpos + 1) % n


@cuda.jit(device=True, inline=True)
def ox_longedge(D, R, rp1, rp2, CH, rc, existed, pi, n, states, sti, topk):
    """Targeted OX: the segment seam (cut) lands RIGHT AFTER one of the topk longest edges
    of parent1 => the worst edge is torn out and rebuilt from parent2. topk<=4."""
    bp = cuda.local.array(16, int32)              # positions of the top-k longest edges (max 16)
    bl = cuda.local.array(16, int32)              # their lengths (D int32)
    for t in range(topk):
        bp[t] = -1
        bl[t] = -1
    for i in range(n):                            # one scan + insert-sort for k<=4 (descending)
        j = i + 1 if i + 1 < n else 0
        d = D[R[rp1, i], R[rp1, j]]
        t = topk - 1
        if d > bl[t]:
            bl[t] = d
            bp[t] = i
            while t > 0 and bl[t] > bl[t - 1]:
                tl = bl[t]; bl[t] = bl[t - 1]; bl[t - 1] = tl
                tp = bp[t]; bp[t] = bp[t - 1]; bp[t - 1] = tp
                t -= 1
    pick = int(rnd01(states, sti) * topk)         # randomly one of the top-k (diversification)
    if pick >= topk:
        pick = topk - 1
    cut = (bp[pick] + 1) % n                       # segment start AFTER the long edge

    L = int(rnd01(states, sti) * 0.4 * n) + int(0.3 * n) + 1
    if L >= n:
        L = n - 1
    for i in range(n):
        existed[pi, i] = 0
    for k in range(L):
        pos = (cut + k) % n
        city = R[rp1, pos]
        CH[rc, pos] = city
        existed[pi, city] = 1
    fillpos = (cut + L) % n
    for k in range(n):
        pos2 = (cut + L + k) % n
        city = R[rp2, pos2]
        if existed[pi, city] == 0:
            CH[rc, fillpos] = city
            existed[pi, city] = 1
            fillpos = (fillpos + 1) % n


@cuda.jit(device=True, inline=True)
def greedy_edge(D, neigh, Knl, R, rp1, rp2, CH, rc, P, p1i, S, p2i, existed, ei, n, states, sti):
    """Greedy edge recombination: from the current city take the shortest AVAILABLE parental
    edge (<=4 candidates = neighbors in p1/p2), fallback = nearest free city via KNN.
    Neighbors computed on the fly from the position tables of both parents => zero adjacency materialization."""
    for i in range(n):
        P[p1i, R[rp1, i]] = i                      # city position in parent1
        S[p2i, R[rp2, i]] = i                      # city position in parent2
        existed[ei, i] = 0
    cur = R[rp1, int(rnd01(states, sti) * n) % n]  # random start (diversification)
    CH[rc, 0] = cur
    existed[ei, cur] = 1
    for step in range(1, n):
        pp = P[p1i, cur]
        a = R[rp1, pp - 1] if pp > 0 else R[rp1, n - 1]
        b = R[rp1, pp + 1] if pp < n - 1 else R[rp1, 0]
        qq = S[p2i, cur]
        cc = R[rp2, qq - 1] if qq > 0 else R[rp2, n - 1]
        dd = R[rp2, qq + 1] if qq < n - 1 else R[rp2, 0]
        best = -1
        bestd = 0
        if existed[ei, a] == 0:
            best = a; bestd = D[cur, a]
        if existed[ei, b] == 0:
            dc = D[cur, b]
            if best == -1 or dc < bestd:
                best = b; bestd = dc
        if existed[ei, cc] == 0:
            dc = D[cur, cc]
            if best == -1 or dc < bestd:
                best = cc; bestd = dc
        if existed[ei, dd] == 0:
            dc = D[cur, dd]
            if best == -1 or dc < bestd:
                best = dd; bestd = dc
        if best == -1:                             # fallback: nearest free city via KNN
            t = 0
            while t < Knl:
                cand = neigh[cur, t]
                if existed[ei, cand] == 0:
                    best = cand
                    t = Knl
                else:
                    t += 1
            if best == -1:                         # KNN exhausted -> linear scan (rare)
                j = 0
                while j < n:
                    if existed[ei, j] == 0:
                        best = j
                        j = n
                    else:
                        j += 1
        CH[rc, step] = best
        existed[ei, best] = 1
        cur = best


@cuda.jit(device=True, inline=True)
def eff2(v, a, dup, use_tabu, tabu_age):
    """Effective length for SELECTION: age penalty (tabu) + length DUPLICATE penalty
    (uniqueness within the colony -- the more copies of this length, the easier to displace)."""
    r = v
    if use_tabu == 1 and a >= tabu_age:
        r += v // 250
    if dup > 1:                                   # length-recurrence: maintaining uniqueness
        m = dup - 1
        if m > 8:
            m = 8
        r += (v // 250) * m
    return r


@cuda.jit(device=True, inline=True)
def find_worst(rlen, age, hcount, col, nbucket, base, pm, use_tabu, tabu_age,
               gbest_val, gbstall, best_tabu):
    """Index + effective length of the WEAKEST (age + duplicate + INCUMBENT penalty).
    The incumbent (rlen==island gbest) holding best longer than best_tabu (1 chunk) gets a penalty
    x1.004 (as in Aparapi) -- self-regulating: it becomes the 'worst' only when the population has converged
    to <0.4% of best (then a shake is desired); with diversity the best stays."""
    worst = 0; wl = -1
    for w in range(pm):
        dw = hcount[col, rlen[base + w] % nbucket]
        le = eff2(rlen[base + w], age[base + w], dw, use_tabu, tabu_age)
        if use_tabu == 1 and rlen[base + w] == gbest_val and gbstall > best_tabu:
            le = int(le * 1.004)                      # simple multiplicative penalty (as in Aparapi) -- self-regulating
        if le > wl:
            wl = le; worst = w
    return worst, wl


@cuda.jit(device=True, inline=True)
def sched_k(gge, total, k1, k2, k3):
    """Adaptive K -- smooth double smoothstep (cubic ease 3x^2-2x^3): k1 -> k2 (plateau)
    -> k3. Transitions: 0.05-0.25 (k1->k2), 0.75-0.95 (k2->k3). k1/k2/k3 from n (solve_ga)."""
    if total <= 0:
        return k3
    t = gge / total
    x1 = (t - 0.05) / 0.20                            # smoothstep 0.05..0.25
    if x1 < 0.0:
        x1 = 0.0
    elif x1 > 1.0:
        x1 = 1.0
    s1 = x1 * x1 * (3.0 - 2.0 * x1)
    x2 = (t - 0.75) / 0.20                            # smoothstep 0.75..0.95
    if x2 < 0.0:
        x2 = 0.0
    elif x2 > 1.0:
        x2 = 1.0
    s2 = x2 * x2 * (3.0 - 2.0 * x2)
    return int(k1 + (k2 - k1) * s1 + (k3 - k2) * s2 + 0.5)


@cuda.jit
def evolve_ga(D, neigh, k1, k2, k3, R, CH, migrant, P, scratch, existed, states, rlen, ml,
              age, gbest, gbest_route, hcount, nbucket, use_uniq,
              n, T, pm, C, grid_epochs, sweeps, migrate_every,
              use_merge, merge_len, use_tabu, tabu_age, resume, epoch_offset, total_epochs,
              ls_mode, le_start, opt_start, ox_mode, ox_topk, force_merge, ops, gbstall, best_tabu):
    g = cuda.cg.this_grid()
    gid = cuda.grid(1)
    if resume == 0:                               # init only in the 1st chunk (resume=0)
        if gid < T:
            base = gid * pm
            t0 = float(epoch_offset) / total_epochs
            le_init = 15 if (t0 >= le_start) else 0
            opt_init = 1 if (t0 >= opt_start) else 0
            for e in range(pm):                   # one-off initial LS (init = small K = k1)
                local_search(D, neigh, k1, R, P, base + e, gid, n, sweeps, ls_mode, le_init, opt_init, ops)
                rlen[base + e] = route_len(D, R, base + e, n)
                age[base + e] = 0
            bi = 0; bl = rlen[base]               # best-ever of the island (immune to tabu)
            for e in range(1, pm):
                if rlen[base + e] < bl:
                    bl = rlen[base + e]; bi = e
            gbest[gid] = bl
            gbstall[gid] = 0                      # incumbent stall counter (epochs without island best improvement)
            for i in range(n):
                gbest_route[gid, i] = R[base + bi, i]
        g.sync()
    colsize = T // C
    for ge in range(grid_epochs):
        gge = epoch_offset + ge                   # global epoch index (for merge windows)
        active_k = sched_k(gge, total_epochs, k1, k2, k3)   # adaptive K (3 thresholds, depending on n)
        tphase = float(gge) / total_epochs
        le_nlong = 15 if (tphase >= le_start) else 0     # long-edge: phase gate
        opt_on = 1 if (tphase >= opt_start) else 0       # 2/3-opt (hybrid): phase gate
        if use_uniq == 1:                         # length histogram per colony (uniqueness)
            idx = gid
            while idx < C * nbucket:
                hcount[idx // nbucket, idx % nbucket] = 0
                idx += T
            g.sync()
            if gid < T:
                bb = gid * pm; col = gid // colsize
                for e in range(pm):
                    cuda.atomic.add(hcount, (col, rlen[bb + e] % nbucket), 1)
            g.sync()
        if gid < T:
            base = gid * pm
            colid = gid // colsize
            gbstall[gid] += 1                     # +1 epoch; reset on best improvement (below)
            for e in range(pm):                   # aging of individuals
                age[base + e] += 1
            # OX + child LS + steady-state (the child displaces the effectively weakest)
            for e in range(pm):
                e2 = e + 1 if e + 1 < pm else 0
                if ox_mode == 3:                       # greedy-edge recombination
                    greedy_edge(D, neigh, active_k, R, base + e, base + e2, CH, base + e,
                                P, gid, scratch, gid, existed, gid, n, states, gid)
                else:
                    use_le_ox = 0                      # ox_mode: 0=classic, 1=long-edge, 2=mix 50/50
                    if ox_mode == 1:
                        use_le_ox = 1
                    elif ox_mode == 2 and rnd01(states, gid) < 0.5:
                        use_le_ox = 1
                    if use_le_ox == 1:
                        ox_longedge(D, R, base + e, base + e2, CH, base + e, existed, gid, n, states, gid, ox_topk)
                    else:
                        ox_crossover(R, base + e, base + e2, CH, base + e, existed, gid, n, states, gid)
                local_search(D, neigh, active_k, CH, P, base + e, gid, n, sweeps, ls_mode, le_nlong, opt_on, ops)
                clen = route_len(D, CH, base + e, n)
                worst, wl = find_worst(rlen, age, hcount, colid, nbucket, base, pm,
                                       use_tabu, tabu_age, gbest[gid], gbstall[gid], best_tabu)
                if clen < wl:
                    for i in range(n):
                        R[base + worst, i] = CH[base + e, i]
                    rlen[base + worst] = clen; age[base + worst] = 0
                    if clen < gbest[gid]:
                        gbest[gid] = clen; gbstall[gid] = 0
                        for i in range(n):
                            gbest_route[gid, i] = CH[base + e, i]
            # perturbation of the effectively weakest + LS
            worst, wl = find_worst(rlen, age, hcount, colid, nbucket, base, pm,
                                   use_tabu, tabu_age, gbest[gid], gbstall[gid], best_tabu)
            double_bridge(R, scratch, base + worst, gid, n, states, gid)
            local_search(D, neigh, active_k, R, P, base + worst, gid, n, sweeps, ls_mode, le_nlong, opt_on, ops)
            rlen[base + worst] = route_len(D, R, base + worst, n)
            age[base + worst] = 0
            if rlen[base + worst] < gbest[gid]:
                gbest[gid] = rlen[base + worst]; gbstall[gid] = 0
                for i in range(n):
                    gbest_route[gid, i] = R[base + worst, i]
        g.sync()
        # --- migration: phase 1 collect the migrant (reads others', stable) ---
        # gge (global) not ge (local) -- otherwise in chunks <migrate_every migration would NEVER fire.
        # force_merge forces do_mig => the "merge" candidate in a trial actually merges.
        do_mig = ((gge + 1) % migrate_every == 0) or (force_merge == 1)
        merge_now = False
        if force_merge == 1:                          # LADDER: merge forced by the controller (whole chunk)
            merge_now = True
        elif use_merge == 1:                          # 4 merge windows around budget thresholds (globally)
            c0 = total_epochs // 4; c1 = total_epochs // 2
            c2 = (3 * total_epochs) // 4; c3 = (9 * total_epochs) // 10
            if (c0 <= gge < c0 + merge_len) or (c1 <= gge < c1 + merge_len) or \
               (c2 <= gge < c2 + merge_len) or (c3 <= gge < c3 + merge_len):
                merge_now = True
        if gid < T and do_mig:
            if merge_now:                             # MERGE: global migration (mixes colonies)
                lo = 0; span = T
            else:                                     # intra-colony (re-divergence)
                lo = (gid // colsize) * colsize; span = colsize
            other = lo + int(rnd01(states, gid) * span)
            obase = other * pm
            ob = 0; obl = rlen[obase]
            for e in range(1, pm):
                if rlen[obase + e] < obl:
                    obl = rlen[obase + e]; ob = e
            for i in range(n):
                migrant[gid, i] = R[obase + ob, i]
            ml[gid] = obl
        g.sync()
        # --- phase 2: apply (writes only its own) ---
        if gid < T and do_mig:
            base = gid * pm
            worst, wl = find_worst(rlen, age, hcount, gid // colsize, nbucket, base, pm,
                                   use_tabu, tabu_age, gbest[gid], gbstall[gid], best_tabu)
            if ml[gid] < wl:
                for i in range(n):
                    R[base + worst, i] = migrant[gid, i]
                rlen[base + worst] = ml[gid]; age[base + worst] = 0
                if ml[gid] < gbest[gid]:
                    gbest[gid] = ml[gid]; gbstall[gid] = 0
                    for i in range(n):
                        gbest_route[gid, i] = migrant[gid, i]
        g.sync()


def knn(D, Knl):
    n = D.shape[0]
    neigh = np.empty((n, Knl), np.int32)
    for c in range(n):
        row = D[c]
        idx = np.argpartition(row, Knl + 1)[:Knl + 1]
        idx = idx[idx != c]
        idx = idx[np.argsort(row[idx])][:Knl]
        neigh[c] = idx
    return neigh


def solve_ga(D, T=2048, pm=4, C=4, grid_epochs=200, sweeps=50,
             migrate_every=10, tpb=128, seed=1, use_merge=0, merge_len=20,
             use_tabu=0, tabu_age=30, chunk=0, verbose=False, tag="", init_mode="kicks",
             use_uniq=0, k1=0, k2=0, k3=0, ls_mode=0, le_start=0.0, opt_start=0.0, ox_mode=0,
             ox_topk=10, metrics=False, ladder=False, decel_rho=0.4, decel_win=2, d_min=2,
             merge_period=2, d_max=4, discovery=False, trial_chunks=2, db_path="", best_tabu=5,
             coords=None, polish_cb=None, polish_k=16, polish_mode="best"):
    # ls_mode: 0 = classic (2-opt+Or-1+3-opt), 1 = relocations (Or-1/2/3+long-edge, no 2/3-opt),
    #          2 = hybrid (classic 2/3-opt from the start + relocations Or-2/3+long-edge TOGETHER from opt_start).
    # le_start: budget fraction from which long-edge turns on inside the relocation block (0.0=from block entry).
    # opt_start: budget fraction from which relocations Or-2/3+long-edge join the classic set (0.6 = after 60%).
    # ox_mode: crossover -- 0 = classic OX, 1 = OX targeted at long edges, 2 = mix 50/50,
    #          3 = greedy-edge recombination (edge-preserving, shortest parental edge + KNN fallback).
    # ox_topk: how many longest edges to draw the seam from in ox_longedge (max 16, default 10).
    # ladder: FINE-GRAINED escalation controller -- list LADDER_MOVES (one atomic change/step:
    #   K+delta, +operator, ox, merge). Apply the next move on deceleration: recent drop (window decel_win)
    #   < decel_rho * peak rate of the step; force after d_max chunks without improvement. After moves are exhausted =
    #   regular merge (every merge_period). ls_mode=9 (ops bitmask). For responsiveness: decel_win=1, d_min=1.
    # discovery: instead of the fixed LADDER_MOVES, on every escalation a ROLLOUT -- state snapshot, trial of each
    #   candidate (trial_chunks) from the same start+RNG, log (state,move,burst,dt,reward=burst/dt) to SQLite
    #   (db_path, table ladder_trials), commit the winner. Offline aggregation of tuples -> statistically best
    #   order (per instance features). Cost ~number_of_candidates x trial_chunks extra/escalation.
    # Adaptive K depending on n (3 budget thresholds): by default (k*<=0) computed from n:
    #   k1=max(8,n//600) until 20% | k2=max(12,n//300) until 60% | k3=max(15,n//200) until the end.
    # Can be overridden by passing k1/k2/k3 (e.g. constant: k1=k2=k3=20).
    # High T = fills the GPU (at n<~1000 it is almost free, strongly improves quality);
    # for large n the LS is more expensive per island, so the GPU saturates earlier.
    from domains.tsp.io import nn_tour
    n = D.shape[0]
    M = T * pm
    nn = nn_tour(D, 0).astype(np.int32)
    k1 = max(8, n // 600) if k1 <= 0 else k1         # K depending on n (3 budget thresholds)
    k2 = max(12, n // 300) if k2 <= 0 else k2
    k3 = max(15, n // 200) if k3 <= 0 else k3
    kmax = max(k1, k2, k3)
    # FINE-GRAINED LADDER: list of atomic moves -- EXACTLY ONE small change per deceleration step.
    #   ("K",v)=set K | ("op",bit)=add operator | ("ox",m)=crossover | ("merge",)=one-off merge.
    rb3 = round(max(15, n // 200) * 1.2)             # upper K (k3+20%)
    KMAX_L = max(16, rb3)
    OP_OR2 = 4; OP_OR3 = 8; OP_3OPT = 16; OP_LE = 32  # operator bits (2-opt+Or-1 = base, always)
    LADDER_MOVES = [                                 # high-value first: operators (big burst) before K-bumps
        ("op", OP_3OPT), ("K", 12), ("op", OP_OR2), ("K", 16), ("op", OP_OR3),
        ("K", 20), ("merge",), ("op", OP_LE), ("K", 24), ("merge",),
        ("ox", 2), ("K", 28), ("merge",), ("ox", 3), ("K", KMAX_L), ("merge",),
    ]
    if ladder:
        kmax = max(kmax, KMAX_L)
    neigh = knn(D, kmax)
    rng = np.random.default_rng(seed)
    R = np.empty((M, n), np.int32)
    if init_mode == "random":                       # OLD: 1 NN + the rest random permutations
        for gid in range(T):
            R[gid * pm] = nn
            for e in range(1, pm):
                R[gid * pm + e] = rng.permutation(n).astype(np.int32)
    else:                                           # NEW: all = NN + 0..5 double-bridge
        for m in range(M):                          # diverse GOOD tours (crucial for large n)
            t = nn.copy()
            for _ in range(int(rng.integers(0, 6))):
                p = np.sort(rng.integers(1, n - 1, 3))
                a, b, c = int(p[0]), int(p[1]), int(p[2])
                if a < b < c:
                    t = np.concatenate([t[:a], t[b:c], t[a:b], t[c:]])
            R[m] = t
    states = rng.integers(-2**31, 2**31 - 1, size=(T, 5), dtype=np.int32)
    states[states == 0] = 1
    d_D = cuda.to_device(D); d_neigh = cuda.to_device(neigh)
    d_R = cuda.to_device(R); d_CH = cuda.device_array_like(R)
    d_mig = cuda.device_array((T, n), np.int32)
    d_P = cuda.device_array((T, n), np.int32)
    d_scr = cuda.device_array((T, n), np.int32)
    d_ex = cuda.device_array((T, n), np.int32)
    d_st = cuda.to_device(states)
    d_rl = cuda.device_array(M, np.int32); d_ml = cuda.device_array(T, np.int32)
    d_age = cuda.device_array(M, np.int32)
    d_gbest = cuda.device_array(T, np.int32)
    d_gbr = cuda.device_array((T, n), np.int32)   # best-ever route per island (immune to tabu)
    d_gbstall = cuda.to_device(np.zeros(T, np.int32))   # incumbent stall (epochs without island best improvement)
    NBUCKET = 1 << 16                             # hash-histogram of lengths per colony
    d_hcount = cuda.to_device(np.zeros((C, NBUCKET), np.int32))
    blocks = (T + tpb - 1) // tpb
    if discovery:                                 # DISCOVERY: candidate rollout + SQLite log (offline order search)
        import sqlite3
        snap_R = cuda.device_array_like(d_R); snap_rl = cuda.device_array_like(d_rl)
        snap_age = cuda.device_array_like(d_age); snap_gb = cuda.device_array_like(d_gbest)
        snap_gbr = cuda.device_array_like(d_gbr); snap_st = cuda.device_array_like(d_st)
        snap_gbstall = cuda.device_array_like(d_gbstall)

        def _snap():                              # device state snapshot (fair start for every candidate)
            snap_R.copy_to_device(d_R); snap_rl.copy_to_device(d_rl); snap_age.copy_to_device(d_age)
            snap_gb.copy_to_device(d_gbest); snap_gbr.copy_to_device(d_gbr); snap_st.copy_to_device(d_st)
            snap_gbstall.copy_to_device(d_gbstall)

        def _restore():
            d_R.copy_to_device(snap_R); d_rl.copy_to_device(snap_rl); d_age.copy_to_device(snap_age)
            d_gbest.copy_to_device(snap_gb); d_gbr.copy_to_device(snap_gbr); d_st.copy_to_device(snap_st)
            d_gbstall.copy_to_device(snap_gbstall)

        def _cands(ops, K, ox):                   # available atomic moves in the given state
            cc = [("none", 0, "stay")]            # NULL-ACTION: baseline "stay as is" (do not escalate)
            for bit, nm in ((16, "3opt"), (4, "Or2"), (8, "Or3"), (32, "LE"),
                            (64, "swap"), (128, "segvar")):     # +operators from Java: swap, long segments
                if not (ops & bit):
                    cc.append(("op", bit, nm))
            if K < KMAX_L:
                cc.append(("K", min(K + 4, KMAX_L), "K+4"))
            if ox == 0:
                cc.append(("ox", 2, "ox_mix"))
            elif ox == 2:
                cc.append(("ox", 3, "ox_greedy"))
            cc.append(("merge", 0, "merge"))
            return cc

        def _apply(cand, K, ops, ox):
            mg = 0
            if cand[0] == "op":
                ops |= cand[1]
            elif cand[0] == "K":
                K = cand[1]
            elif cand[0] == "ox":
                ox = cand[1]
            elif cand[0] == "merge":
                mg = 1
            # "none" => no change (stay with the current config)
            return K, ops, ox, mg
        nn1 = D[np.arange(n), neigh[:, 0]].astype(np.float64)     # nearest-neighbor distribution
        mu = nn1.mean(); sd = nn1.std() + 1e-9
        feat_nn = float(mu); feat_cv = float(sd / mu)
        feat_skew = float((((nn1 - mu) / sd) ** 3).mean())        # skewness of the edge distribution
        feat_kurt = float((((nn1 - mu) / sd) ** 4).mean() - 3.0)  # kurtosis
        farr = D[np.arange(n), neigh[:, -1]].astype(np.float64)   # farthest in KNN => scale span
        feat_far = float(farr.mean() / mu)
        # --- SPATIAL features of the points (numpy, no sklearn) ---
        feat_clark = feat_gridcv = feat_aspect = feat_tight = 0.0
        if coords is not None:
            xy = np.asarray(coords, dtype=np.float64)
            xx = xy[:, 0]; yy = xy[:, 1]
            w = xx.max() - xx.min() + 1e-9; h = yy.max() - yy.min() + 1e-9
            feat_clark = float(mu / (0.5 * np.sqrt(w * h / n) + 1e-9))   # Clark-Evans: <1 clustered, >1 dispersed
            feat_aspect = float(max(w, h) / min(w, h))                   # elongation of the cloud
            g = max(2, int(np.sqrt(n) / 2))                             # density heterogeneity (g x g grid)
            Hh, _, _ = np.histogram2d(xx, yy, bins=g)
            cnt = Hh.flatten()
            feat_gridcv = float(cnt.std() / (cnt.mean() + 1e-9))
            feat_tight = float((nn1 < 0.5 * mu).mean())                 # fraction of tight pairs (cluster proxy)
        dbcon = sqlite3.connect(db_path); dbcur = dbcon.cursor()
        dbcur.execute("CREATE TABLE IF NOT EXISTS ladder_trials (instance TEXT, n INT, "
                      "feat_nn REAL, feat_cv REAL, feat_skew REAL, feat_kurt REAL, feat_far REAL, "
                      "feat_clark REAL, feat_gridcv REAL, feat_aspect REAL, feat_tight REAL, "
                      "seed INT, step INT, state_ops INT, state_K INT, state_ox INT, "
                      "phase_uniq REAL, phase_cv REAL, phase_gbstall REAL, phase_gbstall_max REAL, cum_t REAL, "
                      "candidate TEXT, burst INT, dt REAL, reward REAL, best INT)")
        dbcur.execute("CREATE TABLE IF NOT EXISTS ladder_runs (instance TEXT, n INT, "
                      "feat_nn REAL, feat_cv REAL, feat_skew REAL, feat_kurt REAL, feat_far REAL, "
                      "feat_clark REAL, feat_gridcv REAL, feat_aspect REAL, feat_tight REAL, "
                      "seed INT, terminated_by TEXT, final_step INT, final_best INT, total_time REAL)")
        dbcur.execute("CREATE TABLE IF NOT EXISTS ladder_states (instance TEXT, seed INT, step INT, "
                      "n INT, gbest INT, route BLOB, colony_routes BLOB)")   # SOURCE for GNN: best tour + colony bests
    import time
    t0 = time.time()
    step = grid_epochs if chunk <= 0 else chunk    # chunk<=0 => single launch (no preview)
    if ladder and step > 5:                        # the ladder needs fine-grained decisions
        step = 5
    done = 0
    prev_best = None; r_early = None; stagn = 0     # metrics state (convergence tracking)
    mp = 0; cur_K = 8; cur_ops = 0; cur_ox = 0; pending_merge = 0   # ladder state: move pointer + operators
    merge_cd = 0                                    # merge cooldown: after a merge 2 escalations without merge (the 3rd allows it again)
    es_best = 1 << 62; noimp = 0; stopped_early = False   # early-stop: 3 consecutive chunks without best improvement
    dwell = 0; chunk_idx = 0; rh = []; rung_peak_r = 0.0; stagn_l = 0  # rh=best history within the STEP, peak, stagnation
    if metrics and not ladder:
        print(f"    [{tag}] ep  best  rel/ep%  r_norm  spread%  cv%  uniq%  stagn  s", flush=True)
    if ladder:
        print(f"    [{tag}] ep  m  K  op  ox  best  r_norm  uniq%  mrg  s", flush=True)
    while done < grid_epochs:
        cs = min(step, grid_epochs - done)
        if ladder:                                 # chunk parameters from the current ladder state
            kk1 = kk2 = kk3 = cur_K
            kls = 9; kox = cur_ox; kuq = 1; kum = 0; kops = cur_ops
            kle = 0.0; kopt = 0.0                  # phase gates OFF -- the moves steer
            kfm = pending_merge; pending_merge = 0               # one-off merge (the "merge" move)
            if mp >= len(LADDER_MOVES) and (chunk_idx % merge_period == 0):   # terminal: regular merge
                kfm = 1
        else:
            kk1, kk2, kk3 = k1, k2, k3
            kls = ls_mode; kox = ox_mode; kuq = use_uniq; kum = use_merge
            kle = le_start; kopt = opt_start; kfm = 0; kops = 0
        evolve_ga[blocks, tpb](d_D, d_neigh, kk1, kk2, kk3, d_R, d_CH, d_mig, d_P, d_scr, d_ex,
                               d_st, d_rl, d_ml, d_age, d_gbest, d_gbr,
                               d_hcount, NBUCKET, kuq, n, T, pm, C,
                               cs, sweeps, migrate_every, kum, merge_len,
                               use_tabu, tabu_age, 0 if done == 0 else 1, done, grid_epochs,
                               kls, kle, kopt, kox, ox_topk, kfm, kops, d_gbstall, best_tabu)
        cuda.synchronize()
        done += cs
        if polish_cb is not None:                  # HYBRID (25.09): descent with our LK/or-opt on the best islands after a chunk
            gb_h = d_gbest.copy_to_host(); top = np.argsort(gb_h)[:polish_k]
            gbr_h = d_gbr.copy_to_host(); rl_h = d_rl.copy_to_host(); age_h = d_age.copy_to_host(); gbs_h = d_gbstall.copy_to_host()
            R_h = d_R.copy_to_host(); changed = 0; gain_sum = 0
            injected = 0
            for gid in top:
                base = gid * pm; worst = base + int(np.argmax(rl_h[base:base + pm]))
                if polish_mode == "worst": r0 = R_h[worst].astype(np.int64)          # fresh material (OX child)
                else: r0 = gbr_h[gid].astype(np.int64)                                # best (kick: perturbation in the callback)
                r1 = polish_cb(r0, polish_mode)
                if r1 is None: continue
                L1 = int(D[r1[:-1], r1[1:]].sum() + D[r1[-1], r1[0]])
                if L1 < int(rl_h[worst]):                                             # inject if better than the weakest of the island
                    R_h[worst] = r1.astype(np.int32); rl_h[worst] = L1; age_h[worst] = 0; injected += 1
                    if L1 < int(gb_h[gid]):
                        gain_sum += int(gb_h[gid]) - L1; gb_h[gid] = L1; gbr_h[gid] = r1.astype(np.int32); gbs_h[gid] = 0; changed += 1
            if injected:
                d_R.copy_to_device(R_h); d_rl.copy_to_device(rl_h); d_age.copy_to_device(age_h)
            if changed:
                d_gbest.copy_to_device(gb_h); d_gbr.copy_to_device(gbr_h); d_gbstall.copy_to_device(gbs_h)
            if verbose or ladder:
                print(f"    [{tag}] polish[{polish_mode}]: islands={len(top)} injected={injected} new_best={changed} gain={gain_sum}", flush=True)
            if False:
                pass
        if ladder:                                 # --- escalation CONTROLLER (deceleration recent/peak) + log ---
            bsf = int(d_gbest.copy_to_host().min())
            uniqp = (d_hcount.copy_to_host() > 0).sum(axis=1).mean() / (M // C) * 100.0
            improved = (len(rh) >= 1 and bsf < rh[-1])
            rh.append(bsf)
            if len(rh) > decel_win:                 # recent drop within a window of decel_win chunks (best decreasing => >=0)
                r = float(rh[-1 - decel_win] - rh[-1])
            else:
                r = None
            if r is not None and r > rung_peak_r:   # peak rate of THIS rung
                rung_peak_r = r
            r_norm = (r / rung_peak_r) if (r is not None and rung_peak_r > 0) else 1.0
            stagn_l = 0 if improved else stagn_l + 1
            merged = kfm
            decel = (r is not None and dwell >= d_min and rung_peak_r > 0 and r_norm < decel_rho)
            escalate = decel or (stagn_l >= d_max)               # deceleration or anti-deadlock
            if escalate and discovery:                           # ROLLOUT: try every candidate, log burst/time
                rl_now = d_rl.copy_to_host()                      # PHASE metrics at the decision point
                ph_cv = float(rl_now.std() / (rl_now.mean() + 1e-9) * 100.0)
                gbs_h = d_gbstall.copy_to_host()
                ph_gbstall = float(gbs_h.mean())                  # mean stall
                ph_gbstall_max = float(gbs_h.max())               # whether ANY island has an active tabu penalty (>best_tabu)
                ph_uniq = float(uniqp); cum_t = time.time() - t0
                _snap()
                try:                                  # SOURCE for GNN: best-route + bests of the C colonies (per decision, BEFORE the rollout)
                    gb_h = d_gbest.copy_to_host(); gbr_h = d_gbr.copy_to_host()
                    colsize = T // C
                    st_route = gbr_h[int(gb_h.argmin())].astype(np.int32)
                    st_cols = np.stack([gbr_h[c * colsize + int(gb_h[c * colsize:(c + 1) * colsize].argmin())]
                                        for c in range(C)]).astype(np.int32)
                    dbcur.execute("INSERT INTO ladder_states VALUES (?,?,?,?,?,?,?)",
                                  (tag, seed, mp, n, int(bsf), st_route.tobytes(), st_cols.tobytes()))
                except Exception:
                    pass
                cands = _cands(cur_ops, cur_K, cur_ox)
                if merge_cd > 0:                          # cooldown: 2 escalations after a merge without trying merge
                    cands = [cc for cc in cands if cc[0] != "merge"]
                    merge_cd -= 1
                best_rew = -1e18; best_cand = None
                ops_full = (cur_ops == 252)                      # 4|8|16|32|64|128 -- all operators added
                for cand in cands:
                    _restore()                                   # every candidate starts from the same state+RNG
                    tK, tops, tox, tmg = _apply(cand, cur_K, cur_ops, cur_ox)
                    tc0 = time.time()
                    evolve_ga[blocks, tpb](d_D, d_neigh, tK, tK, tK, d_R, d_CH, d_mig, d_P, d_scr, d_ex,
                                           d_st, d_rl, d_ml, d_age, d_gbest, d_gbr,
                                           d_hcount, NBUCKET, 1, n, T, pm, C,
                                           trial_chunks, sweeps, migrate_every, 0, merge_len,
                                           use_tabu, tabu_age, 1, done, grid_epochs,
                                           9, 0.0, 0.0, tox, ox_topk, tmg, tops, d_gbstall, best_tabu)
                    cuda.synchronize()
                    tb = int(d_gbest.copy_to_host().min())
                    burst = bsf - tb; dtc = time.time() - tc0
                    rew = burst / dtc if dtc > 0 else 0.0         # reward = burst / TIME (objective quality/time)
                    dbcur.execute("INSERT INTO ladder_trials VALUES "
                                  "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (tag, n, feat_nn, feat_cv, feat_skew, feat_kurt, feat_far,
                                   feat_clark, feat_gridcv, feat_aspect, feat_tight,
                                   seed, mp, cur_ops, cur_K, cur_ox,
                                   ph_uniq, ph_cv, ph_gbstall, ph_gbstall_max, cum_t,
                                   cand[2], int(burst), dtc, rew, int(bsf)))
                    # "stay" cannot win before all operators are added (log, but do not end the escalation)
                    if (cand[0] != "none" or ops_full) and rew > best_rew:
                        best_rew = rew; best_cand = cand
                dbcon.commit(); _restore()
                if best_cand is not None:                        # commit the winner permanently
                    cur_K, cur_ops, cur_ox, mg = _apply(best_cand, cur_K, cur_ops, cur_ox)
                    if mg:
                        pending_merge = 1
                        merge_cd = 2                              # after merge: the next 2 escalations without merge
                mp += 1
                dwell = 0; stagn_l = 0; rh = [bsf]; rung_peak_r = 0.0
            elif escalate and mp < len(LADDER_MOVES):            # FIXED: next move from LADDER_MOVES
                mv = LADDER_MOVES[mp]; mp += 1                    # APPLY one atomic move
                if mv[0] == "K":
                    cur_K = min(mv[1], KMAX_L)
                elif mv[0] == "op":
                    cur_ops |= mv[1]
                elif mv[0] == "ox":
                    cur_ox = mv[1]
                else:                                            # "merge" -- one-off in the next chunk
                    pending_merge = 1
                dwell = 0; stagn_l = 0; rh = [bsf]; rung_peak_r = 0.0   # reset the step window
            else:
                dwell += 1
            chunk_idx += 1
            if bsf < es_best:                       # early-stop: counter of chunks without best improvement
                es_best = bsf; noimp = 0
            else:
                noimp += 1
            print(f"    [{tag}] {done:3d} m{mp:2d} K{cur_K:2d} op{cur_ops:2d} x{cur_ox}  {bsf}  "
                  f"{r_norm:4.2f}  {uniqp:4.0f}  {merged}  {time.time()-t0:.0f}", flush=True)
            # stop only when: 3 chunks without improvement AND tabu exhausted AND (discovery) all operators added
            tabu_done = (use_tabu == 0) or (int(d_gbstall.copy_to_host().max()) > best_tabu + 16)
            ops_terminal = (not discovery) or (cur_ops == 252)   # do not finish before the ladder tries 3-opt/LE/swap/segvar
            if noimp >= 3 and tabu_done and ops_terminal:
                stopped_early = True
                break
        elif metrics:                              # rich metrics table every chunk (including the last)
            bsf = int(d_gbest.copy_to_host().min())
            rl_now = d_rl.copy_to_host()               # lengths of the WHOLE population (M individuals)
            mean_p = float(rl_now.mean()); std_p = float(rl_now.std())
            spread = (mean_p / bsf - 1.0) * 100.0      # population spread above best (decreases on convergence)
            cv = std_p / mean_p * 100.0                # coeff. of variation (diversity in objective space)
            distinct = (d_hcount.copy_to_host() > 0).sum(axis=1)
            uniqp = distinct.mean() / (M // C) * 100.0
            if prev_best is None:
                rel_ep = 0.0; r_norm = 1.0
            else:
                rel_ep = (prev_best - bsf) / prev_best / cs * 100.0   # relative improvement per epoch [%]
                if r_early is None and rel_ep > 0:
                    r_early = rel_ep
                r_norm = (rel_ep / r_early) if r_early else 0.0
                stagn = 0 if bsf < prev_best else stagn + 1           # chunks without improvement
            prev_best = bsf
            print(f"    [{tag}] {done:3d}  {bsf}  {rel_ep:6.3f}  {r_norm:5.2f}  "
                  f"{spread:6.2f}  {cv:5.2f}  {uniqp:4.0f}  {stagn:3d}  {time.time()-t0:.0f}",
                  flush=True)
        elif verbose and done < grid_epochs:       # preview of best-so-far + uniqueness after a chunk
            bsf = int(d_gbest.copy_to_host().min())
            uq = ""
            if use_uniq == 1:
                hc = d_hcount.copy_to_host()
                distinct = (hc > 0).sum(axis=1)        # unique lengths per colony
                uq = f" uniq={distinct.mean() / (M // C) * 100:.0f}%"
            print(f"    [{tag}] {done}/{grid_epochs} epochs  best={bsf}{uq}  [{time.time()-t0:.0f}s]",
                  flush=True)
    if discovery:                                  # trajectory summary (credit assignment for DP)
        fbest = int(d_gbest.copy_to_host().min())
        dbcur.execute("INSERT INTO ladder_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (tag, n, feat_nn, feat_cv, feat_skew, feat_kurt, feat_far,
                       feat_clark, feat_gridcv, feat_aspect, feat_tight, seed,
                       "early_stop" if stopped_early else "cap", mp, fbest, time.time() - t0))
        dbcon.commit(); dbcon.close()
    dt = time.time() - t0
    rl = d_gbest.copy_to_host()               # result from best-ever (immune to tabu)
    bi = int(rl.argmin())
    bestt = d_gbr.copy_to_host()[bi]
    perm_ok = sorted(bestt.tolist()) == list(range(n))
    solve_ga.last_tour = np.asarray(bestt, dtype=np.int64).copy()     # best tour (for the playoff/report in solve.py)
    return int(rl.min()), perm_ok, dt


OPT = {"berlin52": 7542, "kroA100": 21282, "pcb3038": 137694,
       "gr431": 171414, "pcb442": 50778, "pr1002": 259045}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from domains.tsp.io import load_tsplib, load_txd, dist_matrix
    path = sys.argv[1] if len(sys.argv) > 1 else "instances/gr431.tsp"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    ge = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    pm = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    C = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    merge = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    tabu = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    name = os.path.basename(path).split(".")[0]
    if path.endswith(".txd"):
        coords, _ = load_txd(path); ewt = "EUC_2D"
    else:
        coords, _, ewt = load_tsplib(path)
    D = dist_matrix(coords, ewt)
    best, perm_ok, dt = solve_ga(D, T=T, pm=pm, C=C, grid_epochs=ge,
                                 use_merge=merge, use_tabu=tabu)
    opt = OPT.get(name)
    gap = f"{(best/opt-1)*100:.3f}%" if opt else "?"
    print(f"{name}: n={D.shape[0]} T={T} pm={pm} C={C} ge={ge} merge={merge} tabu={tabu} "
          f"best={best} opt={opt} gap={gap} perm_ok={perm_ok} [{dt:.1f}s]")
