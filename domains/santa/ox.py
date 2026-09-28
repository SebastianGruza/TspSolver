"""OX (order crossover, Sebastian's idea 24.09): child = segment of parent A at positions [i, i+L) + the remaining cities in
parent B's order (filling from position i+L cyclically). City 0 always at position 0. Parents = best tours of different
lines (Hilbert x greedy), which GPX cannot merge (no common segments). Children have jumps where B passed through the
segment's region -> cheap CPU repair (uncross without penalty + orlong) to pick the best one, then the full GPU
combination on the winner. Usage: ox.py A.csv B.csv OUT_PREFIX [--kids 8 --frac 0.5 --seed 1]"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D


def ox(A, B, i, L):
    """A, B: permutations without closure (A[0]=B[0]=0). Segment A[i:i+L] stays in place, the rest comes from B in order."""
    n = len(A); child = np.empty(n, np.int64); inseg = np.zeros(n, bool); inseg[A[i:i + L]] = True
    child[i:i + L] = A[i:i + L]
    rest = B[~inseg[B]]                        # cities outside the segment in B's order
    # fill positions i+L..n-1, then 0..i-1 (cyclically from the segment end, as in classic OX)
    k = 0
    for p in list(range(i + L, n)) + list(range(0, i)):
        child[p] = rest[k]; k += 1
    z = int(np.where(child == 0)[0][0]); child = np.concatenate([child[z:], child[:z]])
    return child.astype(np.int32)


def main():
    from domains.santa.uncross import uncross_all, score_arr
    from domains.santa.orlong import orlong
    from domains.santa.baseline import knn
    ap = argparse.ArgumentParser(); ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("out")
    ap.add_argument("--kids", type=int, default=8); ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1); n = len(xy)
    A = D.read_tour(a.A)[:-1].astype(np.int64); B = D.read_tour(a.B)[:-1].astype(np.int64)
    sA = D.score(np.append(A, 0), xy, prime); sB = D.score(np.append(B, 0), xy, prime)
    print(f"parents A={sA:.1f} B={sB:.1f}", flush=True)
    neigh = knn(xy, 10); rng = np.random.default_rng(a.seed); best = (1e18, None, None)
    for k in range(a.kids):
        P1, P2 = (A, B) if k % 2 == 0 else (B, A)
        L = int(n * a.frac); i = int(rng.integers(1, n - L))
        t = time.time(); C = ox(P1, P2, i, L); raw = D.score(np.append(C, 0), xy, prime)
        C, s1, mv1 = uncross_all(C, xy, prime, neigh, rounds=60, pen=0.0)
        C, s2, mv2 = orlong(C.astype(np.int64), xy, prime, neigh, rounds=10, PEN=0.0)
        C, s3, mv3 = uncross_all(C, xy, prime, neigh, rounds=30, pen=0.1)
        s = D.score(np.append(C, 0), xy, prime)
        print(f"kid {k}: seg={'A' if k % 2 == 0 else 'B'}[{i}:{i + L}] raw={raw:.1f} uncross0={mv1} orlong={mv2} uncross={mv3} -> {s:.1f} ({time.time() - t:.0f}s)", flush=True)
        D.validate(np.append(C, 0), n)
        if s < best[0]: best = (s, C, k)
    D.write_submission(np.append(best[1], 0), f"{a.out}_{int(best[0])}.csv")
    print(f"OX_BEST kid {best[2]} {best[0]:.1f} -> {a.out}_{int(best[0])}.csv", flush=True)


if __name__ == "__main__":
    main()
