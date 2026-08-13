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
def local_search(D, neigh, Knl, R, P, ri, pi, n, sweeps):
    for i in range(n):
        P[pi, R[ri, i]] = i
    for _r in range(2):
        two_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
        or_opt_neighbor(D, neigh, Knl, R, P, ri, pi, n, sweeps)
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
def eff(v, a, use_tabu, tabu_age):
    """Efektywna długość do SELEKCJI: osobnik starszy niż próg dostaje karę ~0.4%
    (analog kary tabu ×1.004 z oryginału) => łatwiej go wyprzeć = dywersyfikacja."""
    if use_tabu == 1 and a >= tabu_age:
        return v + v // 250
    return v


@cuda.jit
def evolve_ga(D, neigh, Knl, R, CH, migrant, P, scratch, existed, states, rlen, ml,
              age, gbest, gbest_route, n, T, pm, C, grid_epochs, sweeps, migrate_every,
              use_merge, merge_len, use_tabu, tabu_age, resume, epoch_offset, total_epochs):
    g = cuda.cg.this_grid()
    gid = cuda.grid(1)
    if resume == 0:                               # init tylko w 1. kawałku (resume=0)
        if gid < T:
            base = gid * pm
            for e in range(pm):                   # jednorazowy LS startu (init nie jest lok-opt)
                local_search(D, neigh, Knl, R, P, base + e, gid, n, sweeps)
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
        if gid < T:
            base = gid * pm
            for e in range(pm):                   # postarzanie osobników
                age[base + e] += 1
            # OX + LS dziecka + steady-state (dziecko wypiera efektywnie najsłabszego)
            for e in range(pm):
                e2 = e + 1 if e + 1 < pm else 0
                ox_crossover(R, base + e, base + e2, CH, base + e, existed, gid, n, states, gid)
                local_search(D, neigh, Knl, CH, P, base + e, gid, n, sweeps)
                clen = route_len(D, CH, base + e, n)
                worst = 0; wl = eff(rlen[base], age[base], use_tabu, tabu_age)
                for w in range(1, pm):
                    le = eff(rlen[base + w], age[base + w], use_tabu, tabu_age)
                    if le > wl:
                        wl = le; worst = w
                if clen < wl:
                    for i in range(n):
                        R[base + worst, i] = CH[base + e, i]
                    rlen[base + worst] = clen; age[base + worst] = 0
                    if clen < gbest[gid]:
                        gbest[gid] = clen
                        for i in range(n):
                            gbest_route[gid, i] = CH[base + e, i]
            # perturbacja efektywnie najsłabszego + LS
            worst = 0; wl = eff(rlen[base], age[base], use_tabu, tabu_age)
            for w in range(1, pm):
                le = eff(rlen[base + w], age[base + w], use_tabu, tabu_age)
                if le > wl:
                    wl = le; worst = w
            double_bridge(R, scratch, base + worst, gid, n, states, gid)
            local_search(D, neigh, Knl, R, P, base + worst, gid, n, sweeps)
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
            worst = 0; wl = eff(rlen[base], age[base], use_tabu, tabu_age)
            for w in range(1, pm):
                le = eff(rlen[base + w], age[base + w], use_tabu, tabu_age)
                if le > wl:
                    wl = le; worst = w
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


def solve_ga(D, T=2048, pm=4, C=4, grid_epochs=200, sweeps=50, Knl=10,
             migrate_every=10, tpb=128, seed=1, use_merge=0, merge_len=20,
             use_tabu=0, tabu_age=30, chunk=0, verbose=False, tag="", init_mode="kicks"):
    # T wysokie = wypełnia GPU (przy n<~1000 to niemal darmowe, mocno poprawia jakość);
    # dla dużych n LS jest droższy per wyspa, więc GPU nasyca się wcześniej.
    from tsp_io import nn_tour
    n = D.shape[0]
    M = T * pm
    nn = nn_tour(D, 0).astype(np.int32)
    neigh = knn(D, Knl)
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
    blocks = (T + tpb - 1) // tpb
    import time
    t0 = time.time()
    step = grid_epochs if chunk <= 0 else chunk    # chunk<=0 => jeden launch (bez podglądu)
    done = 0
    while done < grid_epochs:
        cs = min(step, grid_epochs - done)
        evolve_ga[blocks, tpb](d_D, d_neigh, Knl, d_R, d_CH, d_mig, d_P, d_scr, d_ex,
                               d_st, d_rl, d_ml, d_age, d_gbest, d_gbr, n, T, pm, C,
                               cs, sweeps, migrate_every, use_merge, merge_len,
                               use_tabu, tabu_age, 0 if done == 0 else 1, done, grid_epochs)
        cuda.synchronize()
        done += cs
        if verbose and done < grid_epochs:         # podgląd best-so-far po kawałku
            bsf = int(d_gbest.copy_to_host().min())
            print(f"    [{tag}] {done}/{grid_epochs} epok  best={bsf}  [{time.time()-t0:.0f}s]",
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
