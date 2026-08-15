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
    """Przenieś segment [i, i+L) tuż ZA pozycję 'after' (poza segmentem). rev=1 => odwrócony. L<=16."""
    seg = cuda.local.array(16, int32)
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
def swap_neighbor(D, neigh, Knl, R, P, ri, pi, n, n_long, max_sweeps):
    """Vertex swap CELOWANY w długie krawędzie (z Javy: mutVertexSwap, ulepszone): miasto do zamiany =
    koniec jednej z n_long najdłuższych krawędzi; zamień z bliskim sąsiadem (KNN). 4 krawędzie, bez rev.
    O(n) na znalezienie długich + O(n_long·K) na swapy. n_long<=16."""
    bp = cuda.local.array(16, int32)                     # pozycje top-n_long najdłuższych krawędzi
    bl = cuda.local.array(16, int32)
    for _sweep in range(max_sweeps):
        for t in range(n_long):                          # znajdź n_long najdłuższych krawędzi (i -> i+1)
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
        for t in range(n_long):                          # swap miasta z długiej krawędzi z bliskim sąsiadem
            i = bp[t]
            if i <= 0 or i >= n - 1:                      # brzegi/wrap => pomiń
                continue
            a = R[ri, i]; ap = R[ri, i - 1]; an = R[ri, i + 1]
            best_delta = 0; best_j = -1
            for kk in range(Knl):
                c = neigh[a, kk]; jj = P[pi, c]
                if jj <= i + 1 and jj >= i - 1:           # ta sama/sąsiednia poz. => pomiń
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
    """Relokacja segmentów DŁUŻSZYCH (z Javy: mutSegmentRelocation, zmienna długość) — len 5 i 10,
    pokrywa lukę medium/long ponad sztywne Or-2/Or-3. KNN-guided (przez or_opt_seg)."""
    or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 5, max_sweeps)
    or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 10, max_sweeps)


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
def local_search(D, neigh, Knl, R, P, ri, pi, n, sweeps, ls_mode, le_nlong, opt_on, ops):
    for i in range(n):
        P[pi, R[ri, i]] = i
    if ls_mode == 0:                                   # A: klasyczny 2-opt + Or-1 + 3-opt
        for _r in range(2):
            two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
    elif ls_mode == 1:                                 # B: relokacje BEZ 2/3-opt
        for _r in range(3):
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # Or-1
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)      # Or-2 (+reversed)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)      # Or-3 (+reversed)
            if le_nlong > 0:                                           # long-edge bramkowany fazą
                long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
    elif ls_mode == 2:                                 # C: hybryda — klasyk od startu, relokacje Or-2/3+long-edge RAZEM od opt_start
        for _r in range(2):                            # szkielet: 2-opt + Or-1 + 3-opt (rozplatanie)
            two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
            or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # Or-1
        three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)        # Or-1
        if opt_on == 1:                                # relokacje dołączają RAZEM w późnej fazie (endgame)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)      # Or-2 (+reversed)
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)      # Or-3 (+reversed)
            if le_nlong > 0:
                long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
    elif ls_mode == 3:                                 # D: ultra-szybki — tylko 2-opt + Or-1
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
    else:                                              # E (ls_mode==9): BITMASKA — drabina drobnoziarnista
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)   # baza: 2-opt
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)    # baza: Or-1
        if ops & 4:                                                # +Or-2
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 2, sweeps)
        if ops & 8:                                                # +Or-3
            or_opt_seg(D, neigh, Knl, R, P, ri, pi, n, 3, sweeps)
        if ops & 16:                                               # +3-opt
            three_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        if (ops & 32) and le_nlong > 0:                            # +long-edge
            long_edge_relocation(D, neigh, Knl, R, P, ri, pi, n, le_nlong, 2)
        if ops & 64:                                               # +vertex swap celowany w długie krawędzie
            swap_neighbor(D, neigh, Knl, R, P, ri, pi, n, 10, sweeps)
        if ops & 128:                                              # +segmenty długie 5/10 (z Javy)
            or_opt_seg_var(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        if ops != 0:                                               # cleanup gdy dołożono operator
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
def ox_longedge(D, R, rp1, rp2, CH, rc, existed, pi, n, states, sti, topk):
    """OX celowany: szew segmentu (cut) ląduje TUŻ ZA jedną z topk najdłuższych krawędzi
    rodzica1 => najgorsza krawędź jest rywana i odbudowywana z rodzica2. topk<=4."""
    bp = cuda.local.array(16, int32)              # pozycje top-k najdłuższych krawędzi (max 16)
    bl = cuda.local.array(16, int32)              # ich długości (D int32)
    for t in range(topk):
        bp[t] = -1
        bl[t] = -1
    for i in range(n):                            # jeden skan + insert-sort na k<=4 (malejąco)
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
    pick = int(rnd01(states, sti) * topk)         # losowo jedna z top-k (dywersyfikacja)
    if pick >= topk:
        pick = topk - 1
    cut = (bp[pick] + 1) % n                       # start segmentu ZA długą krawędzią

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
    """Greedy edge recombination: z bieżącego miasta bierz najkrótszą DOSTĘPNĄ krawędź
    rodzicielską (<=4 kandydatów = sąsiedzi w p1/p2), fallback = najbliższy wolny przez KNN.
    Sąsiedzi liczeni w locie z tablic pozycji obu rodziców => zero materializacji adjacency."""
    for i in range(n):
        P[p1i, R[rp1, i]] = i                      # pozycja miasta w rodzicu1
        S[p2i, R[rp2, i]] = i                      # pozycja miasta w rodzicu2
        existed[ei, i] = 0
    cur = R[rp1, int(rnd01(states, sti) * n) % n]  # losowy start (dywersyfikacja)
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
        if best == -1:                             # fallback: najbliższy wolny przez KNN
            t = 0
            while t < Knl:
                cand = neigh[cur, t]
                if existed[ei, cand] == 0:
                    best = cand
                    t = Knl
                else:
                    t += 1
            if best == -1:                         # KNN wyczerpany -> skan liniowy (rzadkie)
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
def find_worst(rlen, age, hcount, col, nbucket, base, pm, use_tabu, tabu_age,
               gbest_val, gbstall, best_tabu):
    """Indeks + efektywna długość NAJSŁABSZEGO (kara wieku + duplikatu + INKUMBENTA).
    Inkumbent (rlen==gbest wyspy) trzymający best dłużej niż best_tabu (1 chunk) dostaje karę
    ×1.004 (jak Aparapi) — samoregulującą: staje się 'najgorszym' tylko gdy populacja zbiegła
    do <0.4% od best (wtedy potrząśnięcie pożądane), przy różnorodności best zostaje."""
    worst = 0; wl = -1
    for w in range(pm):
        dw = hcount[col, rlen[base + w] % nbucket]
        le = eff2(rlen[base + w], age[base + w], dw, use_tabu, tabu_age)
        if use_tabu == 1 and rlen[base + w] == gbest_val and gbstall > best_tabu:
            le = int(le * 1.004)                      # prosta kara multiplikatywna (jak Aparapi) — samoregulująca
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
              use_merge, merge_len, use_tabu, tabu_age, resume, epoch_offset, total_epochs,
              ls_mode, le_start, opt_start, ox_mode, ox_topk, force_merge, ops, gbstall, best_tabu):
    g = cuda.cg.this_grid()
    gid = cuda.grid(1)
    if resume == 0:                               # init tylko w 1. kawałku (resume=0)
        if gid < T:
            base = gid * pm
            t0 = float(epoch_offset) / total_epochs
            le_init = 15 if (t0 >= le_start) else 0
            opt_init = 1 if (t0 >= opt_start) else 0
            for e in range(pm):                   # jednorazowy LS startu (init = małe K = k1)
                local_search(D, neigh, k1, R, P, base + e, gid, n, sweeps, ls_mode, le_init, opt_init, ops)
                rlen[base + e] = route_len(D, R, base + e, n)
                age[base + e] = 0
            bi = 0; bl = rlen[base]               # best-ever wyspy (odporny na tabu)
            for e in range(1, pm):
                if rlen[base + e] < bl:
                    bl = rlen[base + e]; bi = e
            gbest[gid] = bl
            gbstall[gid] = 0                      # licznik stall inkumbenta (epoki bez poprawy best wyspy)
            for i in range(n):
                gbest_route[gid, i] = R[base + bi, i]
        g.sync()
    colsize = T // C
    for ge in range(grid_epochs):
        gge = epoch_offset + ge                   # globalny indeks epoki (dla okien merge)
        active_k = sched_k(gge, total_epochs, k1, k2, k3)   # adaptacyjne K (3 progi, zależne od n)
        tphase = float(gge) / total_epochs
        le_nlong = 15 if (tphase >= le_start) else 0     # long-edge: bramka fazy
        opt_on = 1 if (tphase >= opt_start) else 0       # 2/3-opt (hybryda): bramka fazy
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
            gbstall[gid] += 1                     # +1 epoka; reset przy poprawie best (niżej)
            for e in range(pm):                   # postarzanie osobników
                age[base + e] += 1
            # OX + LS dziecka + steady-state (dziecko wypiera efektywnie najsłabszego)
            for e in range(pm):
                e2 = e + 1 if e + 1 < pm else 0
                if ox_mode == 3:                       # greedy-edge recombination
                    greedy_edge(D, neigh, active_k, R, base + e, base + e2, CH, base + e,
                                P, gid, scratch, gid, existed, gid, n, states, gid)
                else:
                    use_le_ox = 0                      # ox_mode: 0=klasyk, 1=long-edge, 2=mix 50/50
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
            # perturbacja efektywnie najsłabszego + LS
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
        # --- migracja: faza 1 zbierz migranta (czyta cudze, stabilne) ---
        # gge (globalne) nie ge (lokalne) — inaczej w chunkach <migrate_every migracja NIGDY nie fire.
        # force_merge wymusza do_mig => kandydat "merge" w trialu faktycznie merge'uje.
        do_mig = ((gge + 1) % migrate_every == 0) or (force_merge == 1)
        merge_now = False
        if force_merge == 1:                          # DRABINA: merge wymuszony przez kontroler (cały chunk)
            merge_now = True
        elif use_merge == 1:                          # 4 okna merge wokół progów budżetu (globalnie)
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
             coords=None):
    # ls_mode: 0 = klasyczny (2-opt+Or-1+3-opt), 1 = relokacje (Or-1/2/3+long-edge, bez 2/3-opt),
    #          2 = hybryda (klasyk 2/3-opt od startu + relokacje Or-2/3+long-edge RAZEM od opt_start).
    # le_start: ułamek budżetu od którego włącza się long-edge wewnątrz bloku relokacji (0.0=od wejścia bloku).
    # opt_start: ułamek budżetu od którego relokacje Or-2/3+long-edge dołączają do klasyka (0.6 = po 60%).
    # ox_mode: krzyżowanie — 0 = klasyczny OX, 1 = OX celowany w długie krawędzie, 2 = mix 50/50,
    #          3 = greedy-edge recombination (edge-preserving, najkrótsza krawędź rodzicielska + KNN fallback).
    # ox_topk: z ilu najdłuższych krawędzi losować szew w ox_longedge (max 16, domyślnie 10).
    # ladder: kontroler eskalacji DROBNOZIARNISTY — lista LADDER_MOVES (jedna atomowa zmiana/krok:
    #   K+Δ, +operator, ox, merge). Aplikuj następny ruch gdy deceleracja: świeży spadek (okno decel_win)
    #   < decel_rho · szczytowe tempo kroku; force po d_max chunkach bez poprawy. Po wyczerpaniu ruchów =
    #   merge regularny (co merge_period). ls_mode=9 (bitmaska ops). Dla responsywności: decel_win=1, d_min=1.
    # discovery: zamiast stałej LADDER_MOVES, na każdej eskalacji ROLLOUT — snapshot stanu, próba każdego
    #   kandydata (trial_chunks) z tego samego startu+RNG, log (stan,ruch,burst,dt,reward=burst/dt) do SQLite
    #   (db_path, tabela ladder_trials), commit zwycięzcy. Offline agregacja krotek -> statystycznie najlepsza
    #   kolejność (per cechy instancji). Koszt ~liczba_kandydatów × trial_chunks ekstra/eskalację.
    # Adaptacyjne K zależne od n (3 progi budżetu): domyślnie (k*<=0) liczone z n:
    #   k1=max(8,n//600) do 20% | k2=max(12,n//300) do 60% | k3=max(15,n//200) do końca.
    # Można nadpisać podając k1/k2/k3 (np. stałe: k1=k2=k3=20).
    # T wysokie = wypełnia GPU (przy n<~1000 to niemal darmowe, mocno poprawia jakość);
    # dla dużych n LS jest droższy per wyspa, więc GPU nasyca się wcześniej.
    from tsp_io import nn_tour
    n = D.shape[0]
    M = T * pm
    nn = nn_tour(D, 0).astype(np.int32)
    k1 = max(8, n // 600) if k1 <= 0 else k1         # K zależne od n (3 progi budżetu)
    k2 = max(12, n // 300) if k2 <= 0 else k2
    k3 = max(15, n // 200) if k3 <= 0 else k3
    kmax = max(k1, k2, k3)
    # DRABINA DROBNOZIARNISTA: lista atomowych ruchów — DOKŁADNIE JEDNA mała zmiana na krok deceleracji.
    #   ("K",v)=ustaw K | ("op",bit)=dołóż operator | ("ox",m)=krzyżowanie | ("merge",)=jednorazowy merge.
    rb3 = round(max(15, n // 200) * 1.2)             # górne K (k3+20%)
    KMAX_L = max(16, rb3)
    OP_OR2 = 4; OP_OR3 = 8; OP_3OPT = 16; OP_LE = 32  # bity operatorów (2-opt+Or-1 = baza, zawsze)
    LADDER_MOVES = [                                 # high-value first: operatory (duży burst) przed K-bumpami
        ("op", OP_3OPT), ("K", 12), ("op", OP_OR2), ("K", 16), ("op", OP_OR3),
        ("K", 20), ("merge",), ("op", OP_LE), ("K", 24), ("merge",),
        ("ox", 2), ("K", 28), ("merge",), ("ox", 3), ("K", KMAX_L), ("merge",),
    ]
    if ladder:
        kmax = max(kmax, KMAX_L)
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
    d_gbstall = cuda.to_device(np.zeros(T, np.int32))   # stall inkumbenta (epoki bez poprawy best wyspy)
    NBUCKET = 1 << 16                             # hash-histogram długości per kolonia
    d_hcount = cuda.to_device(np.zeros((C, NBUCKET), np.int32))
    blocks = (T + tpb - 1) // tpb
    if discovery:                                 # DISCOVERY: rollout kandydatów + log SQLite (offline order search)
        import sqlite3
        snap_R = cuda.device_array_like(d_R); snap_rl = cuda.device_array_like(d_rl)
        snap_age = cuda.device_array_like(d_age); snap_gb = cuda.device_array_like(d_gbest)
        snap_gbr = cuda.device_array_like(d_gbr); snap_st = cuda.device_array_like(d_st)
        snap_gbstall = cuda.device_array_like(d_gbstall)

        def _snap():                              # snapshot stanu device (fair start dla każdego kandydata)
            snap_R.copy_to_device(d_R); snap_rl.copy_to_device(d_rl); snap_age.copy_to_device(d_age)
            snap_gb.copy_to_device(d_gbest); snap_gbr.copy_to_device(d_gbr); snap_st.copy_to_device(d_st)
            snap_gbstall.copy_to_device(d_gbstall)

        def _restore():
            d_R.copy_to_device(snap_R); d_rl.copy_to_device(snap_rl); d_age.copy_to_device(snap_age)
            d_gbest.copy_to_device(snap_gb); d_gbr.copy_to_device(snap_gbr); d_st.copy_to_device(snap_st)
            d_gbstall.copy_to_device(snap_gbstall)

        def _cands(ops, K, ox):                   # dostępne atomowe ruchy w danym stanie
            cc = [("none", 0, "stay")]            # NULL-ACTION: baseline "zostań jak jest" (nie eskaluj)
            for bit, nm in ((16, "3opt"), (4, "Or2"), (8, "Or3"), (32, "LE"),
                            (64, "swap"), (128, "segvar")):     # +operatory z Javy: swap, długie segmenty
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
            # "none" => bez zmian (zostań przy bieżącym configu)
            return K, ops, ox, mg
        nn1 = D[np.arange(n), neigh[:, 0]].astype(np.float64)     # rozkład najbliższego sąsiada
        mu = nn1.mean(); sd = nn1.std() + 1e-9
        feat_nn = float(mu); feat_cv = float(sd / mu)
        feat_skew = float((((nn1 - mu) / sd) ** 3).mean())        # skośność rozkładu krawędzi
        feat_kurt = float((((nn1 - mu) / sd) ** 4).mean() - 3.0)  # kurtoza
        farr = D[np.arange(n), neigh[:, -1]].astype(np.float64)   # najdalszy w KNN => rozpiętość skali
        feat_far = float(farr.mean() / mu)
        # --- cechy PRZESTRZENNE punktów (numpy, bez sklearn) ---
        feat_clark = feat_gridcv = feat_aspect = feat_tight = 0.0
        if coords is not None:
            xy = np.asarray(coords, dtype=np.float64)
            xx = xy[:, 0]; yy = xy[:, 1]
            w = xx.max() - xx.min() + 1e-9; h = yy.max() - yy.min() + 1e-9
            feat_clark = float(mu / (0.5 * np.sqrt(w * h / n) + 1e-9))   # Clark-Evans: <1 skupione, >1 rozproszone
            feat_aspect = float(max(w, h) / min(w, h))                   # wydłużenie chmury
            g = max(2, int(np.sqrt(n) / 2))                             # heterogeniczność gęstości (siatka g×g)
            Hh, _, _ = np.histogram2d(xx, yy, bins=g)
            cnt = Hh.flatten()
            feat_gridcv = float(cnt.std() / (cnt.mean() + 1e-9))
            feat_tight = float((nn1 < 0.5 * mu).mean())                 # frakcja ciasnych par (proxy skupień)
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
    import time
    t0 = time.time()
    step = grid_epochs if chunk <= 0 else chunk    # chunk<=0 => jeden launch (bez podglądu)
    if ladder and step > 5:                        # drabina wymaga drobnego ziarna decyzji
        step = 5
    done = 0
    prev_best = None; r_early = None; stagn = 0     # stan metryk (śledzenie zbieżności)
    mp = 0; cur_K = 8; cur_ops = 0; cur_ox = 0; pending_merge = 0   # stan drabiny: wskaźnik ruchu + operatory
    merge_cd = 0                                    # cooldown merge: po merge 2 eskalacje bez merge (3. znów pozwala)
    es_best = 1 << 62; noimp = 0; stopped_early = False   # early-stop: 3 kolejne chunki bez poprawy best
    dwell = 0; chunk_idx = 0; rh = []; rung_peak_r = 0.0; stagn_l = 0  # rh=historia best w KROKU, peak, stagnacja
    if metrics and not ladder:
        print(f"    [{tag}] ep  best  rel/ep%  r_norm  spread%  cv%  uniq%  stagn  s", flush=True)
    if ladder:
        print(f"    [{tag}] ep  m  K  op  ox  best  r_norm  uniq%  mrg  s", flush=True)
    while done < grid_epochs:
        cs = min(step, grid_epochs - done)
        if ladder:                                 # parametry chunka z bieżącego stanu drabiny
            kk1 = kk2 = kk3 = cur_K
            kls = 9; kox = cur_ox; kuq = 1; kum = 0; kops = cur_ops
            kle = 0.0; kopt = 0.0                  # bramki fazowe OFF — sterują ruchy
            kfm = pending_merge; pending_merge = 0               # merge jednorazowy (ruch "merge")
            if mp >= len(LADDER_MOVES) and (chunk_idx % merge_period == 0):   # terminal: merge regularny
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
        if ladder:                                 # --- KONTROLER eskalacji (deceleracja recent/peak) + log ---
            bsf = int(d_gbest.copy_to_host().min())
            uniqp = (d_hcount.copy_to_host() > 0).sum(axis=1).mean() / (M // C) * 100.0
            improved = (len(rh) >= 1 and bsf < rh[-1])
            rh.append(bsf)
            if len(rh) > decel_win:                 # świeży spadek w oknie decel_win chunków (best malejący => >=0)
                r = float(rh[-1 - decel_win] - rh[-1])
            else:
                r = None
            if r is not None and r > rung_peak_r:   # szczytowe tempo TEGO szczebla
                rung_peak_r = r
            r_norm = (r / rung_peak_r) if (r is not None and rung_peak_r > 0) else 1.0
            stagn_l = 0 if improved else stagn_l + 1
            merged = kfm
            decel = (r is not None and dwell >= d_min and rung_peak_r > 0 and r_norm < decel_rho)
            escalate = decel or (stagn_l >= d_max)               # deceleracja lub anty-deadlock
            if escalate and discovery:                           # ROLLOUT: próbuj każdego kandydata, loguj burst/czas
                rl_now = d_rl.copy_to_host()                      # metryki FAZY w punkcie decyzji
                ph_cv = float(rl_now.std() / (rl_now.mean() + 1e-9) * 100.0)
                gbs_h = d_gbstall.copy_to_host()
                ph_gbstall = float(gbs_h.mean())                  # średni stall
                ph_gbstall_max = float(gbs_h.max())               # czy KTÓRAŚ wyspa ma aktywną karę tabu (>best_tabu)
                ph_uniq = float(uniqp); cum_t = time.time() - t0
                _snap()
                cands = _cands(cur_ops, cur_K, cur_ox)
                if merge_cd > 0:                          # cooldown: 2 eskalacje po merge bez próby merge
                    cands = [cc for cc in cands if cc[0] != "merge"]
                    merge_cd -= 1
                best_rew = -1e18; best_cand = None
                for cand in cands:
                    _restore()                                   # każdy kandydat startuje z tego samego stanu+RNG
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
                    rew = burst / dtc if dtc > 0 else 0.0         # reward = burst / CZAS (cel jakość/czas)
                    dbcur.execute("INSERT INTO ladder_trials VALUES "
                                  "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (tag, n, feat_nn, feat_cv, feat_skew, feat_kurt, feat_far,
                                   feat_clark, feat_gridcv, feat_aspect, feat_tight,
                                   seed, mp, cur_ops, cur_K, cur_ox,
                                   ph_uniq, ph_cv, ph_gbstall, ph_gbstall_max, cum_t,
                                   cand[2], int(burst), dtc, rew, int(bsf)))
                    if rew > best_rew:
                        best_rew = rew; best_cand = cand
                dbcon.commit(); _restore()
                if best_cand is not None:                        # commituj zwycięzcę na stałe
                    cur_K, cur_ops, cur_ox, mg = _apply(best_cand, cur_K, cur_ops, cur_ox)
                    if mg:
                        pending_merge = 1
                        merge_cd = 2                              # po merge: następne 2 eskalacje bez merge
                mp += 1
                dwell = 0; stagn_l = 0; rh = [bsf]; rung_peak_r = 0.0
            elif escalate and mp < len(LADDER_MOVES):            # FIXED: kolejny ruch z LADDER_MOVES
                mv = LADDER_MOVES[mp]; mp += 1                    # APLIKUJ jeden atomowy ruch
                if mv[0] == "K":
                    cur_K = min(mv[1], KMAX_L)
                elif mv[0] == "op":
                    cur_ops |= mv[1]
                elif mv[0] == "ox":
                    cur_ox = mv[1]
                else:                                            # "merge" — jednorazowy w następnym chunku
                    pending_merge = 1
                dwell = 0; stagn_l = 0; rh = [bsf]; rung_peak_r = 0.0   # reset okna kroku
            else:
                dwell += 1
            chunk_idx += 1
            if bsf < es_best:                       # early-stop: licznik chunków bez poprawy best
                es_best = bsf; noimp = 0
            else:
                noimp += 1
            print(f"    [{tag}] {done:3d} m{mp:2d} K{cur_K:2d} op{cur_ops:2d} x{cur_ox}  {bsf}  "
                  f"{r_norm:4.2f}  {uniqp:4.0f}  {merged}  {time.time()-t0:.0f}", flush=True)
            # stop dopiero gdy 3 chunki bez poprawy I tabu inkumbenta WYCZERPANE (dało pełną szansę)
            tabu_done = (use_tabu == 0) or (int(d_gbstall.copy_to_host().max()) > best_tabu + 16)
            if noimp >= 3 and tabu_done:
                stopped_early = True
                break
        elif metrics:                              # bogata tabela metryk co chunk (także ostatni)
            bsf = int(d_gbest.copy_to_host().min())
            rl_now = d_rl.copy_to_host()               # długości CAŁEJ populacji (M osobników)
            mean_p = float(rl_now.mean()); std_p = float(rl_now.std())
            spread = (mean_p / bsf - 1.0) * 100.0      # rozrzut populacji nad best (maleje przy zbieżności)
            cv = std_p / mean_p * 100.0                # wsp. zmienności (różnorodność w przestrzeni celu)
            distinct = (d_hcount.copy_to_host() > 0).sum(axis=1)
            uniqp = distinct.mean() / (M // C) * 100.0
            if prev_best is None:
                rel_ep = 0.0; r_norm = 1.0
            else:
                rel_ep = (prev_best - bsf) / prev_best / cs * 100.0   # względna poprawa na epokę [%]
                if r_early is None and rel_ep > 0:
                    r_early = rel_ep
                r_norm = (rel_ep / r_early) if r_early else 0.0
                stagn = 0 if bsf < prev_best else stagn + 1           # chunki bez poprawy
            prev_best = bsf
            print(f"    [{tag}] {done:3d}  {bsf}  {rel_ep:6.3f}  {r_norm:5.2f}  "
                  f"{spread:6.2f}  {cv:5.2f}  {uniqp:4.0f}  {stagn:3d}  {time.time()-t0:.0f}",
                  flush=True)
        elif verbose and done < grid_epochs:       # podgląd best-so-far + unikalność po kawałku
            bsf = int(d_gbest.copy_to_host().min())
            uq = ""
            if use_uniq == 1:
                hc = d_hcount.copy_to_host()
                distinct = (hc > 0).sum(axis=1)        # unikalnych długości per kolonia
                uq = f" uniq={distinct.mean() / (M // C) * 100:.0f}%"
            print(f"    [{tag}] {done}/{grid_epochs} epok  best={bsf}{uq}  [{time.time()-t0:.0f}s]",
                  flush=True)
    if discovery:                                  # podsumowanie trajektorii (credit assignment dla DP)
        fbest = int(d_gbest.copy_to_host().min())
        dbcur.execute("INSERT INTO ladder_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (tag, n, feat_nn, feat_cv, feat_skew, feat_kurt, feat_far,
                       feat_clark, feat_gridcv, feat_aspect, feat_tight, seed,
                       "early_stop" if stopped_early else "cap", mp, fbest, time.time() - t0))
        dbcon.commit(); dbcon.close()
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
