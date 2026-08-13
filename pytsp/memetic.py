"""Kooperatywny memetyczny ILS na GPU (Numba, cooperative groups).

Każdy wątek = wyspa (jedna trasa) prowadząca Iterated Local Search:
  local search (2-opt best-of-K) -> akceptacja ILS (trzymaj best) -> perturbacja
  (double-bridge). Między wyspami MIGRACJA przez grid.sync (adopcja best lepszego
  sąsiada). CAŁY bieg w jednym cooperative launchu — CPU tylko odpala i czyta best.

To realizuje wymóg: co epokę synchronizacja TYLKO na GPU (grid.sync), zero CPU.
Operatory 3-opt/OX/or-opt dojdą w kolejnych incrementach; tu rdzeń pętli + sync.
"""
import os
import numpy as np
from numba import cuda, int32
from prng_dev import rnd01  # PRNG współdzielony


@cuda.jit(device=True, inline=True)
def route_len(D, R, gid, n):
    s = 0
    for i in range(n - 1):
        s += D[R[gid, i], R[gid, i + 1]]
    s += D[R[gid, n - 1], R[gid, 0]]
    return s


@cuda.jit(device=True, inline=True)
def two_opt_move(D, R, gid, n, states, K):
    """Jeden ruch 2-opt: best-of-K losowych, zastosuj jeśli poprawia."""
    best_gain = 0
    bi = -1
    bj = -1
    for _ in range(K):
        i = int(rnd01(states, gid) * (n - 1))
        j = int(rnd01(states, gid) * (n - 1))
        if i > j:
            i, j = j, i
        if j - i < 1 or (i == 0 and j == n - 1):
            continue
        a = R[gid, i]; b = R[gid, i + 1]; c = R[gid, j]; d = R[gid, (j + 1) % n]
        gain = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
        if gain < best_gain:
            best_gain = gain; bi = i; bj = j
    if bi >= 0:
        lo = bi + 1; hi = bj
        while lo < hi:
            t = R[gid, lo]; R[gid, lo] = R[gid, hi]; R[gid, hi] = t
            lo += 1; hi -= 1
        return True
    return False


@cuda.jit(device=True, inline=True)
def two_opt_full(D, R, gid, n, max_sweeps):
    """Pełny 2-opt do lokalnego optimum: skan wszystkich par, sweep aż brak poprawy."""
    for _sweep in range(max_sweeps):
        improved = False
        for i in range(n - 1):
            a = R[gid, i]
            b = R[gid, i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = R[gid, j]; d = R[gid, (j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d]:
                    lo = i + 1; hi = j
                    while lo < hi:
                        t = R[gid, lo]; R[gid, lo] = R[gid, hi]; R[gid, hi] = t
                        lo += 1; hi -= 1
                    b = R[gid, i + 1]     # zaktualizuj sąsiada po odwróceniu
                    improved = True
        if not improved:
            break


@cuda.jit(device=True, inline=True)
def double_bridge(R, S, gid, n, states):
    """Perturbacja 4-opt double-bridge: [0,a)+[b,c)+[a,b)+[c,n). S = scratch."""
    a = 1 + int(rnd01(states, gid) * (n - 3))
    b = a + 1 + int(rnd01(states, gid) * (n - a - 2))
    c = b + 1 + int(rnd01(states, gid) * (n - b - 1))
    if a < 1: a = 1
    if b <= a: b = a + 1
    if c <= b: c = b + 1
    if c >= n: c = n - 1
    idx = 0
    for i in range(0, a): S[gid, idx] = R[gid, i]; idx += 1
    for i in range(b, c): S[gid, idx] = R[gid, i]; idx += 1
    for i in range(a, b): S[gid, idx] = R[gid, i]; idx += 1
    for i in range(c, n): S[gid, idx] = R[gid, i]; idx += 1
    for i in range(n): R[gid, i] = S[gid, i]


@cuda.jit
def evolve_coop(D, cur, best, scratch, states, bestlen, n,
                grid_epochs, ls_iters, K, migrate_every):
    g = cuda.cg.this_grid()
    gid = cuda.grid(1)
    T = cur.shape[0]
    if gid < T:
        bestlen[gid] = route_len(D, best, gid, n)
    g.sync()
    for ge in range(grid_epochs):
        if gid < T:
            two_opt_full(D, cur, gid, n, ls_iters)   # ls_iters = max_sweeps
            clen = route_len(D, cur, gid, n)
            if clen < bestlen[gid]:
                for i in range(n):
                    best[gid, i] = cur[gid, i]
                bestlen[gid] = clen
            else:
                for i in range(n):
                    cur[gid, i] = best[gid, i]
        g.sync()                              # wszystkie best[] ustalone
        if gid < T and (ge + 1) % migrate_every == 0:
            other = int(rnd01(states, gid) * T)
            if other < T and bestlen[other] < bestlen[gid]:
                for i in range(n):            # adoptuj best lepszego sąsiada
                    cur[gid, i] = best[other, i]
        if gid < T:
            double_bridge(cur, scratch, gid, n, states)
        g.sync()                              # bariera epoki


def solve(D, T=512, grid_epochs=300, ls_iters=50, K=16, migrate_every=20,
          tpb=128, seed=1):
    from tsp_io import nn_tour
    n = D.shape[0]
    nn = nn_tour(D, 0).astype(np.int32)
    cur = np.tile(nn, (T, 1)); best = cur.copy()
    rng = np.random.default_rng(seed)
    states = rng.integers(-2**31, 2**31 - 1, size=(T, 5), dtype=np.int32)
    states[states == 0] = 1
    d_D = cuda.to_device(D); d_cur = cuda.to_device(cur); d_best = cuda.to_device(best)
    d_scr = cuda.device_array_like(cur); d_st = cuda.to_device(states)
    d_bl = cuda.device_array(T, np.int32)
    blocks = (T + tpb - 1) // tpb
    import time
    t0 = time.time()
    evolve_coop[blocks, tpb](d_D, d_cur, d_best, d_scr, d_st, d_bl, n,
                             grid_epochs, ls_iters, K, migrate_every)
    cuda.synchronize()
    dt = time.time() - t0
    bl = d_bl.copy_to_host()
    bi = int(bl.argmin()); bestt = d_best.copy_to_host()[bi]
    perm_ok = sorted(bestt.tolist()) == list(range(n))
    return int(bl.min()), perm_ok, dt


OPT = {"berlin52": 7542, "kroA100": 21282, "pcb3038": 137694}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from tsp_io import load_tsplib, load_txd, dist_matrix
    path = sys.argv[1] if len(sys.argv) > 1 else "instances/berlin52.tsp"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    ge = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    name = os.path.basename(path).split(".")[0]
    coords, _ = (load_txd(path) if path.endswith(".txd") else load_tsplib(path))
    D = dist_matrix(coords)
    best, perm_ok, dt = solve(D, T=T, grid_epochs=ge)
    opt = OPT.get(name)
    gap = f"{(best/opt-1)*100:.3f}%" if opt else "?"
    print(f"{name}: n={D.shape[0]} best={best} opt={opt} gap={gap} "
          f"perm_ok={perm_ok} [{dt:.1f}s]")
