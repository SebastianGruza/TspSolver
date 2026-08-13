"""Rdzeń GPU (Numba) — PRNG + tour_len + lokalny 2-opt. Increment walidacyjny:
dowodzi PRNG (XORShift 1:1 z Javą), lookup odległości i chirurgii trasy.

PRNG: XORShift z 5-int stanem per wątek, dokładnie jak TspGAKernel.random()
(int32 wrapping = semantyka int Javy). random01 wg oryginału (/MIN lub /MAX).
"""
import os
import numpy as np
from numba import cuda, int32

# ---------------- PRNG (device) — 1:1 z Java TspGAKernel.random() -------------
@cuda.jit(device=True, inline=True)
def rnd_int(states, gid):
    s0 = states[gid, 0]
    t = int32(s0 ^ (s0 >> 7))
    states[gid, 0] = states[gid, 1]
    states[gid, 1] = states[gid, 2]
    states[gid, 2] = states[gid, 3]
    states[gid, 3] = states[gid, 4]
    s4 = states[gid, 4]
    states[gid, 4] = int32((s4 ^ (s4 << 6)) ^ (t ^ (t << 13)))
    return int32((states[gid, 1] + states[gid, 1] + 1) * states[gid, 4])

@cuda.jit(device=True, inline=True)
def rnd01(states, gid):
    v = rnd_int(states, gid)
    if v < 0:
        return -v / 2147483648.0
    elif v > 0:
        return v / 2147483647.0
    return 0.0

# ---------------- tour_len (kernel) — walidacja vs CPU -----------------------
@cuda.jit
def tour_len_kernel(D, tours, out, n):
    gid = cuda.grid(1)
    if gid >= tours.shape[0]:
        return
    s = 0
    for i in range(n - 1):
        s += D[tours[gid, i], tours[gid, i + 1]]
    s += D[tours[gid, n - 1], tours[gid, 0]]
    out[gid] = s

# ---------------- 2-opt lokalny (kernel) -------------------------------------
# każdy wątek = jedna trasa; iters × (best-of-K losowych ruchów 2-opt), stosuje
# najlepszy poprawiający (odwrócenie segmentu). Standardowy 2-opt (spirit).
@cuda.jit
def two_opt_kernel(D, tours, states, n, iters, K):
    gid = cuda.grid(1)
    if gid >= tours.shape[0]:
        return
    for _ in range(iters):
        best_gain = 0
        bi = -1
        bj = -1
        for _k in range(K):
            i = int(rnd01(states, gid) * (n - 1))
            j = int(rnd01(states, gid) * (n - 1))
            if i > j:
                i, j = j, i
            if j - i < 1 or (i == 0 and j == n - 1):
                continue
            a = tours[gid, i]; b = tours[gid, i + 1]
            c = tours[gid, j]; d = tours[gid, (j + 1) % n]
            gain = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
            if gain < best_gain:
                best_gain = gain; bi = i; bj = j
        if bi >= 0:
            lo = bi + 1; hi = bj
            while lo < hi:
                tmp = tours[gid, lo]; tours[gid, lo] = tours[gid, hi]; tours[gid, hi] = tmp
                lo += 1; hi -= 1


def make_states(T, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.integers(-2**31, 2**31 - 1, size=(T, 5), dtype=np.int32)
    s[s == 0] = 1  # XORShift nie znosi zerowego stanu
    return s


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from tsp_io import load_tsplib, load_txd, dist_matrix, nn_tour, tour_len

    path = sys.argv[1] if len(sys.argv) > 1 else "instances/berlin52.tsp"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    if path.endswith(".txd"):
        coords, _ = load_txd(path)
    else:
        coords, _ = load_tsplib(path)
    D = dist_matrix(coords); n = D.shape[0]
    nn = nn_tour(D, 0)
    print(f"n={n}  NN_len={tour_len(D, nn)}  T={T} ITERS={ITERS} K={K}")

    # populacja: T kopii NN (rozjadą się przez różne seedy PRNG)
    tours = np.tile(nn, (T, 1)).astype(np.int32)
    states = make_states(T)
    d_D = cuda.to_device(D); d_t = cuda.to_device(tours)
    d_s = cuda.to_device(states); d_out = cuda.device_array(T, np.int32)
    tpb = 128; blocks = (T + tpb - 1) // tpb

    # walidacja tour_len GPU vs CPU
    tour_len_kernel[blocks, tpb](d_D, d_t, d_out, n); cuda.synchronize()
    gpu_len0 = int(d_out.copy_to_host()[0]); cpu_len0 = tour_len(D, nn)
    print(f"tour_len GPU={gpu_len0} CPU={cpu_len0} -> {'OK' if gpu_len0 == cpu_len0 else 'FAIL'}")

    import time
    t0 = time.time()
    two_opt_kernel[blocks, tpb](d_D, d_t, d_s, n, ITERS, K); cuda.synchronize()
    dt = time.time() - t0
    tour_len_kernel[blocks, tpb](d_D, d_t, d_out, n); cuda.synchronize()
    lens = d_out.copy_to_host()
    best = int(lens.min())
    # sprawdź, że najlepsza trasa nadal permutacja
    bt = d_t.copy_to_host()[int(lens.argmin())]
    perm_ok = sorted(bt.tolist()) == list(range(n))
    print(f"2-opt: best={best} mean={lens.mean():.0f} (NN={cpu_len0})  "
          f"perm_ok={perm_ok}  [{dt:.2f}s, {T*ITERS/dt/1e6:.0f}M ruchów/s]")
