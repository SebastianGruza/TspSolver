"""Populacyjny memetyk GA na GPU (Numba, cooperative groups).

Rozszerza ILS o strukturę populacji jak w oryginale:
- pm OSOBNIKÓW na wątek (wyspa); OX crossover między nimi + selekcja elitarna;
- KOLONIE: T wątków w C koloniach; migracja w obrębie kolonii (adopcja best
  sąsiada) — race-free: bufor migranta + dwie fazy grid.sync;
- lokalny memetyk: 2-opt ⇄ Or-opt na listach sąsiadów; double-bridge na najsłabszym.

Trasy płasko: R[M, n], M=T*pm; wątek gid ma osobniki [gid*pm .. gid*pm+pm).
P/scratch/existed/states — per wątek (gid), reużywane sekwencyjnie po osobnikach.
"""
import os
import numpy as np
from numba import cuda, int32
from prng_dev import rnd01


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
    """Przenieś segment [i, i+L) tuż ZA pozycję 'after' (poza segmentem). rev=1 => odwrócony. L<=3."""
    seg = cuda.local.array(3, int32)
    for t in range(L):
        seg[t] = R[ri, i + t]
    if rev == 1:
        lo = 0; hi = L - 1
        while lo < hi:
            tmp = seg[lo]; seg[lo] = seg[hi]; seg[hi] = tmp
            lo += 1; hi -= 1
    if after >= i + L:                              # wstawka za segmentem: przesuń [i+L..after] w lewo o L
        k = i; j = i + L
        while j <= after:
            R[ri, k] = R[ri, j]; P[pi, R[ri, k]] = k
            k += 1; j += 1
        for t in range(L):
            R[ri, k + t] = seg[t]; P[pi, seg[t]] = k + t
    else:                                           # after < i-1: przesuń [after+1..i-1] w prawo o L
        k = i + L - 1; j = i - 1
        while j > after:
            R[ri, k] = R[ri, j]; P[pi, R[ri, k]] = k
            k -= 1; j -= 1
        for t in range(L):
            R[ri, after + 1 + t] = seg[t]; P[pi, seg[t]] = after + 1 + t


@cuda.jit(device=True, inline=True)
def or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, L, max_sweeps):
    """Or-opt segmentu L miast, KNN-guided: forward obok KNN(a), reversed obok KNN(z). O(n·K)/sweep."""
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
            for kk in range(Knl):                   # forward: c blisko a (nowa krawędź c-a)
                c = neigh[a, kk]; jc = P[pi, c]
                if ip <= jc <= i + L - 1:            # c==prev lub wewnątrz segmentu => pomiń
                    continue
                d = R[ri, jc + 1 if jc + 1 < n else 0]
                delta = (D[c, a] + D[z, d] - D[c, d]) - remove_gain
                if delta < best_delta:
                    best_delta = delta; best_after = jc; best_rev = 0
            for kk in range(Knl):                   # reversed: c blisko z (nowa krawędź c-z)
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
def three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, max_sweeps):
    """3-opt neighbor-guided (typ z oryginału: reverse [i+1..j] + [j+1..k],
    nowe krawędzie a-c, b-e, d-f). c∈sąsiedzi(a), e∈sąsiedzi(b), pruning a-c."""
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
                base = dab + D[c, d] - dac        # (dab+dcd+def) - (dac+dbe+ddf), część stała
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
    """Targetowana naprawa: znajdź ~n_long najdłuższych krawędzi trasy i wstaw w każdą
    lukę (u,v) najlepiej pasujące miasto z KNN(u)∪KNN(v), gdy obniża cel. Poprawność:
    trzymam MIASTA krawędzi (nie pozycje) + sprawdzam czy krawędź istnieje po ruchach."""
    le_u = cuda.local.array(16, int32)
    le_v = cuda.local.array(16, int32)
    le_len = cuda.local.array(16, int32)
    if n_long > 16:
        n_long = 16
    for _sweep in range(max_sweeps):
        cnt = 0                                       # 1. top n_long najdłuższych krawędzi
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
        applied = False                               # 2. wstaw najlepsze miasto w każdą lukę
        for t in range(cnt):
            u = le_u[t]; v = le_v[t]
            p = P[pi, u]
            pv = p + 1 if p + 1 < n else 0
            if R[ri, pv] != v:                        # krawędź już zmieniona wcześniej -> pomiń
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
                relocate_city(R, P, ri, pi, n, best_c, P[pi, u])   # wstaw c tuż za u
                applied = True
        if not applied:
            break


@cuda.jit(device=True, inline=True)
def local_search(D, neigh, Knl, R, P, ri, pi, n, sweeps):
    for i in range(n):
        P[pi, R[ri, i]] = i
    for _r in range(2):
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)        # Or-1
        or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)          # Or-2 (+reversed)
        or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)          # Or-3 (+reversed)
    long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, 15, 3)        # napraw najgorsze krawędzie
    three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
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
    """Order Crossover: segment [cut,cut+L) z rodzica1, reszta w kolejności rodzica2."""
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
def eff2(v, a, dup, use_tabu, tabu_age):
    """Efektywna długość do SELEKCJI: kara wieku (tabu) + kara DUPLIKATU długości
    (unikalność w kolonii — im więcej kopii tej długości, tym łatwiej wyprzeć)."""
    r = v
    if use_tabu == 1 and a >= tabu_age:
        r += v // 250
    if dup > 1:                                   # length-recurrence: utrzymanie unikalności
        m = dup - 1
        if m > 8:
            m = 8
        r += (v // 250) * m
    return r


@cuda.jit(device=True, inline=True)
def find_worst(rlen, age, hcount, col, nbucket, base, pm, use_tabu, tabu_age):
    """Indeks + efektywna długość NAJSŁABSZEGO (z karą wieku i duplikatu długości)."""
    d0 = hcount[col, rlen[base] % nbucket]
    worst = 0
    wl = eff2(rlen[base], age[base], d0, use_tabu, tabu_age)
    for w in range(1, pm):
        dw = hcount[col, rlen[base + w] % nbucket]
        le = eff2(rlen[base + w], age[base + w], dw, use_tabu, tabu_age)
        if le > wl:
            wl = le; worst = w
    return worst, wl


@cuda.jit(device=True, inline=True)
def sched_k(gge, total, k1, k2, k3):
    """Adaptacyjne K — płynny podwójny smoothstep (cubic ease 3x²−2x³): k1 → k2 (plateau)
    → k3. Przejścia: 0.05–0.25 (k1→k2), 0.75–0.95 (k2→k3). k1/k2/k3 z n (solve_ga)."""
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
              use_merge, merge_len, use_tabu, tabu_age, resume, epoch_offset, total_epochs):
    g = cuda.cg.this_grid()
    gid = cuda.grid(1)
    if resume == 0:                               # init tylko w 1. kawałku (resume=0)
        if gid < T:
            base = gid * pm
            for e in range(pm):                   # jednorazowy LS startu (init = małe K = k1)
                local_search(D, neigh, k1, R, P, base + e, gid, n, sweeps)
                rlen[base + e] = route_len(D, R, base + e, n)
                age[base + e] = 0
            bi = 0; bl = rlen[base]               # best-ever wyspy (odporny na tabu)
            for e in range(1, pm):
                if rlen[base + e] < bl:
                    bl = rlen[base + e]; bi = e
            gbest[gid] = bl
            for i in range(n):
                gbest_route[gid, i] = R[base + bi, i]
        g.sync()
    colsize = T // C
    for ge in range(grid_epochs):
        gge = epoch_offset + ge                   # globalny indeks epoki (dla okien merge)
        active_k = sched_k(gge, total_epochs, k1, k2, k3)   # adaptacyjne K (3 progi, zależne od n)
        if use_uniq == 1:                         # histogram długości per kolonia (unikalność)
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
            for e in range(pm):                   # postarzanie osobników
                age[base + e] += 1
            # OX + LS dziecka + steady-state (dziecko wypiera efektywnie najsłabszego)
            for e in range(pm):
                e2 = e + 1 if e + 1 < pm else 0
                ox_crossover(R, base + e, base + e2, CH, base + e, existed, gid, n, states, gid)
                local_search(D, neigh, active_k, CH, P, base + e, gid, n, sweeps)
                clen = route_len(D, CH, base + e, n)
                worst, wl = find_worst(rlen, age, hcount, colid, nbucket, base, pm,
                                       use_tabu, tabu_age)
                if clen < wl:
                    for i in range(n):
                        R[base + worst, i] = CH[base + e, i]
                    rlen[base + worst] = clen; age[base + worst] = 0
                    if clen < gbest[gid]:
                        gbest[gid] = clen
                        for i in range(n):
                            gbest_route[gid, i] = CH[base + e, i]
            # perturbacja efektywnie najsłabszego + LS
            worst, wl = find_worst(rlen, age, hcount, colid, nbucket, base, pm,
                                   use_tabu, tabu_age)
            double_bridge(R, scratch, base + worst, gid, n, states, gid)
            local_search(D, neigh, active_k, R, P, base + worst, gid, n, sweeps)
            rlen[base + worst] = route_len(D, R, base + worst, n)
            age[base + worst] = 0
            if rlen[base + worst] < gbest[gid]:
                gbest[gid] = rlen[base + worst]
                for i in range(n):
                    gbest_route[gid, i] = R[base + worst, i]
        g.sync()
        # --- migracja: faza 1 zbierz migranta (czyta cudze, stabilne) ---
        do_mig = (ge + 1) % migrate_every == 0
        merge_now = False
        if use_merge == 1:                            # 4 okna merge wokół progów budżetu (globalnie)
            c0 = total_epochs // 4; c1 = total_epochs // 2
            c2 = (3 * total_epochs) // 4; c3 = (9 * total_epochs) // 10
            if (c0 <= gge < c0 + merge_len) or (c1 <= gge < c1 + merge_len) or \
               (c2 <= gge < c2 + merge_len) or (c3 <= gge < c3 + merge_len):
                merge_now = True
        if gid < T and do_mig:
            if merge_now:                             # MERGE: migracja globalna (miesza kolonie)
                lo = 0; span = T
            else:                                     # intra-colony (re-dywergencja)
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
        # --- faza 2: zastosuj (pisze tylko swoje) ---
        if gid < T and do_mig:
            base = gid * pm
            worst, wl = find_worst(rlen, age, hcount, gid // colsize, nbucket, base, pm,
                                   use_tabu, tabu_age)
            if ml[gid] < wl:
                for i in range(n):
                    R[base + worst, i] = migrant[gid, i]
                rlen[base + worst] = ml[gid]; age[base + worst] = 0
                if ml[gid] < gbest[gid]:
                    gbest[gid] = ml[gid]
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
             use_uniq=0, k1=0, k2=0, k3=0):
    # Adaptacyjne K zależne od n (3 progi budżetu): domyślnie (k*<=0) liczone z n:
    #   k1=max(5,n//600) do 20% | k2=max(10,n//300) do 60% | k3=max(15,n//200) do końca.
    # Można nadpisać podając k1/k2/k3 (np. stałe: k1=k2=k3=20).
    # T wysokie = wypełnia GPU (przy n<~1000 to niemal darmowe, mocno poprawia jakość);
    # dla dużych n LS jest droższy per wyspa, więc GPU nasyca się wcześniej.
    from tsp_io import nn_tour
    n = D.shape[0]
    M = T * pm
    nn = nn_tour(D, 0).astype(np.int32)
    k1 = max(5, n // 600) if k1 <= 0 else k1         # K zależne od n (3 progi budżetu)
    k2 = max(10, n // 300) if k2 <= 0 else k2
    k3 = max(15, n // 200) if k3 <= 0 else k3
    kmax = max(k1, k2, k3)
    neigh = knn(D, kmax)
    rng = np.random.default_rng(seed)
    R = np.empty((M, n), np.int32)
    if init_mode == "random":                       # STARY: 1 NN + reszta losowe permutacje
        for gid in range(T):
            R[gid * pm] = nn
            for e in range(1, pm):
                R[gid * pm + e] = rng.permutation(n).astype(np.int32)
    else:                                           # NOWY: wszystkie = NN + 0..5 double-bridge
        for m in range(M):                          # różnorodne DOBRE trasy (kluczowe dla dużych n)
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
    d_gbr = cuda.device_array((T, n), np.int32)   # best-ever route per wyspa (odporny na tabu)
    NBUCKET = 1 << 16                             # hash-histogram długości per kolonia
    d_hcount = cuda.to_device(np.zeros((C, NBUCKET), np.int32))
    blocks = (T + tpb - 1) // tpb
    import time
    t0 = time.time()
    step = grid_epochs if chunk <= 0 else chunk    # chunk<=0 => jeden launch (bez podglądu)
    done = 0
    while done < grid_epochs:
        cs = min(step, grid_epochs - done)
        evolve_ga[blocks, tpb](d_D, d_neigh, k1, k2, k3, d_R, d_CH, d_mig, d_P, d_scr, d_ex,
                               d_st, d_rl, d_ml, d_age, d_gbest, d_gbr,
                               d_hcount, NBUCKET, use_uniq, n, T, pm, C,
                               cs, sweeps, migrate_every, use_merge, merge_len,
                               use_tabu, tabu_age, 0 if done == 0 else 1, done, grid_epochs)
        cuda.synchronize()
        done += cs
        if verbose and done < grid_epochs:         # podgląd best-so-far + unikalność po kawałku
            bsf = int(d_gbest.copy_to_host().min())
            uq = ""
            if use_uniq == 1:
                hc = d_hcount.copy_to_host()
                distinct = (hc > 0).sum(axis=1)        # unikalnych długości per kolonia
                uq = f" uniq={distinct.mean() / (M // C) * 100:.0f}%"
            print(f"    [{tag}] {done}/{grid_epochs} epok  best={bsf}{uq}  [{time.time()-t0:.0f}s]",
                  flush=True)
    dt = time.time() - t0
    rl = d_gbest.copy_to_host()               # wynik z best-ever (odporny na tabu)
    bi = int(rl.argmin())
    bestt = d_gbr.copy_to_host()[bi]
    perm_ok = sorted(bestt.tolist()) == list(range(n))
    return int(rl.min()), perm_ok, dt


OPT = {"berlin52": 7542, "kroA100": 21282, "pcb3038": 137694,
       "gr431": 171414, "pcb442": 50778, "pr1002": 259045}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from tsp_io import load_tsplib, load_txd, dist_matrix
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
