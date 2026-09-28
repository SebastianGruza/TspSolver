"""Correctness tests: score with penalty vs the Kaggle definition; eval_moves agrees with the full recomputation; apply_batch
does not break the permutation and does not increase the cost; small n, synthetic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda

from domains.santa import data as D
from domains.santa.baseline import hilbert_order, knn
from domains.santa.gpu_ls import GpuLS, apply_batch, eval_moves, score_np


def ref_score(path, xy, prime):
    s = 0.0
    for k in range(1, len(path)):
        d = float(np.hypot(*(xy[path[k]] - xy[path[k - 1]])))
        if k % 10 == 0 and not prime[path[k - 1]]:
            d *= 1.1
        s += d
    return s


def test_score_matches_definition():
    rng = np.random.default_rng(1); n = 500
    xy = rng.uniform(0, 100, (n, 2)); prime = D.primes_upto(n - 1)
    R = np.concatenate([[0], rng.permutation(np.arange(1, n))]); path = np.concatenate([R, [0]])
    assert abs(D.score(path, xy, prime) - ref_score(path, xy, prime)) < 1e-6
    assert abs(score_np(R, xy, prime.astype(np.uint8)) - ref_score(path, xy, prime)) < 1e-6


def test_moves_exact_and_monotone():
    rng = np.random.default_rng(2); n = 3000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    s0 = score_np(R, xy, prime)
    # one round by hand: check that the gain of every applied move = difference of the full score
    pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
    ls.d_R.copy_to_device(R.astype(np.int32)); ls.d_pos.copy_to_device(pos)
    blocks = (n + 127) // 128
    eval_moves[blocks, 128](ls.d_xy, ls.d_prime, ls.d_R, ls.d_pos, ls.d_neigh, ls.K, n, ls.W, ls.Lmax, 0.1, ls.Lblk,
                            ls.d_gain, ls.d_j, ls.d_L, ls.d_rev, ls.d_type); cuda.synchronize()
    gain = ls.d_gain.copy_to_host(); mj = ls.d_j.copy_to_host(); mL = ls.d_L.copy_to_host()
    mrev = ls.d_rev.copy_to_host(); mtype = ls.d_type.copy_to_host()
    idx = np.argsort(-gain)[:50]
    checked = 0
    for i in idx:
        if gain[i] <= 0: break
        R1 = R.astype(np.int32).copy()
        one = np.array([i], dtype=np.int64)
        ap = apply_batch(R1, one, gain, mj, mL, mrev, mtype, n, ls.W)
        assert ap == 1
        assert len(np.unique(R1)) == n and R1[0] == 0
        real = s0 - score_np(R1, xy, prime)
        assert abs(real - gain[i]) < 1e-6, (i, mtype[i], gain[i], real)
        checked += 1
    assert checked >= 10
    R2, s1, hist = ls.run(R, max_rounds=200)
    assert s1 < s0 * 0.9 and len(np.unique(R2)) == n and R2[0] == 0
    assert abs(score_np(R2, xy, prime) - s1) < 1e-6


if __name__ == "__main__":
    test_score_matches_definition(); print("score OK")
    test_moves_exact_and_monotone(); print("moves OK")


def test_relaxed_penalty_returns_true_best():
    rng = np.random.default_rng(3); n = 3000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    R1, s1, _ = ls.run(R, max_rounds=100)
    R2, s2, _ = ls.run(R1, max_rounds=50, pen=0.02)
    assert abs(score_np(R2, xy, prime) - s2) < 1e-6 and s2 <= s1 + 1e-6


def test_prime_slot_fill_exact():
    rng = np.random.default_rng(5); n = 4000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    R, s0, _ = ls.run(R, max_rounds=100)          # local optimum
    pos = np.empty(n, np.int32); pos[R] = np.arange(n, dtype=np.int32)
    ls.d_R.copy_to_device(R.astype(np.int32)); ls.d_pos.copy_to_device(pos)
    eval_moves[(n + 127) // 128, 128](ls.d_xy, ls.d_prime, ls.d_R, ls.d_pos, ls.d_neigh, ls.K, n, ls.W, ls.Lmax, 0.1, ls.Lblk,
                                      ls.d_gain, ls.d_j, ls.d_L, ls.d_rev, ls.d_type); cuda.synchronize()
    gain = ls.d_gain.copy_to_host(); mj = ls.d_j.copy_to_host(); mL = ls.d_L.copy_to_host()
    mrev = ls.d_rev.copy_to_host(); mtype = ls.d_type.copy_to_host()
    idx4 = [i for i in np.argsort(-gain) if gain[i] > 0 and mtype[i] == 4][:20]
    for i in idx4:
        R1 = R.astype(np.int32).copy()
        assert apply_batch(R1, np.array([i], dtype=np.int64), gain, mj, mL, mrev, mtype, n, ls.W) == 1
        assert len(np.unique(R1)) == n and R1[0] == 0 and prime[R1[i]] == 1
        assert abs((s0 - score_np(R1, xy, prime)) - gain[i]) < 1e-6
    print("type4 checked", len(idx4))


def test_lk_chain_exact_and_improves():
    from domains.santa.gpu_lk import GpuLK, apply_lk_batch
    rng = np.random.default_rng(7); n = 4000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    R, s0, _ = ls.run(R, max_rounds=100)                # local optimum of Or-opt/2-opt
    for oropt in (False, True):
      lk = GpuLK(ls, W=200, depth=8, br=(8, 5, 3), oropt=oropt)
      gain, depth, cuts, mvs, dirs = lk.eval(R)
      idx = [i for i in np.argsort(-gain) if gain[i] > 0][:40]
      assert len(idx) > 0
      deep = 0
      for a in idx:
        R1 = R.astype(np.int32).copy()
        assert apply_lk_batch(R1, np.array([a], dtype=np.int64), gain, depth, cuts, mvs, dirs, n) == 1
        assert len(np.unique(R1)) == n and R1[0] == 0
        assert abs((s0 - score_np(R1, xy, prime)) - gain[a]) < 1e-6, (a, depth[a], gain[a], s0 - score_np(R1, xy, prime))
        deep += int(depth[a] >= 2)
      R2, s2, hist = lk.run(R, max_rounds=50)
      assert s2 < s0 - 1e-6 and abs(score_np(R2, xy, prime) - s2) < 1e-6
    print("lk chains checked", len(idx), "depth>=2:", deep, "dir-1:", int((dirs[idx] == -1).sum()), "gain", round(s0 - s2, 1))


def test_dp_window_is_optimal_and_exact():
    """Window DP (k=8, m=6) == brute force over 720 permutations with penalty; gain == difference of the full score; permutation OK."""
    import itertools
    from domains.santa.gpu_dp import GpuDP
    rng = np.random.default_rng(11); n = 2000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    dp = GpuDP(ls, k=8)
    s0 = score_np(R, xy, prime)
    R1 = R.copy(); ap, g = dp.pass_offset(R1, 1)
    s1 = score_np(R1, xy, prime)
    assert ap > 0, ap
    assert len(np.unique(R1)) == n and R1[0] == 0
    assert abs((s0 - s1) - g) < 1e-6, (s0 - s1, g)
    # brute force on 30 random windows of offset 1: DP optimum == optimum over the inner permutations
    def win_cost(seq, s):
        c = 0.0
        for t in range(len(seq) - 1):
            p = s + t; d = float(np.hypot(*(xy[seq[t + 1]] - xy[seq[t]])))
            if p % 10 == 9 and not prime[seq[t]]: d *= 1.1
            c += d
        return c
    k = 8; checked = 0
    for w in rng.choice((n - 1) // k, 30, replace=False):
        s = 1 + w * k
        if s + k > n: continue
        inner = list(R[s + 1:s + k - 1]); best = min(win_cost([R[s]] + list(pm) + [R[s + k - 1]], s) for pm in itertools.permutations(inner))
        got = win_cost(list(R1[s:s + k]), s)
        assert abs(got - best) < 1e-6, (s, got, best)
        checked += 1
    assert checked >= 20
    R2, s2, hist = dp.run(R1, sweeps=1)
    assert s2 <= s1 + 1e-6 and abs(score_np(R2, xy, prime) - s2) < 1e-6
    print("dp windows OK: brute-force checked", checked, "gain pass1", round(s0 - s1, 1), "sweep", round(s1 - s2, 1))


def test_xswap_exact_far_and_adjacent():
    """xswap: GPU gain == difference of the full score (including adjacent and reversed segments), permutation OK, improves."""
    from domains.santa.gpu_xswap import GpuXSwap, apply_xswap_batch
    rng = np.random.default_rng(13); n = 4000
    xy = rng.uniform(0, 1000, (n, 2)); prime = D.primes_upto(n - 1).astype(np.uint8)
    neigh = knn(xy, 8); R = hilbert_order(xy)
    ls = GpuLS(xy, prime, neigh, W=200, Lmax=3)
    R, s0, _ = ls.run(R, max_rounds=100)
    xs = GpuXSwap(ls, Lmax=3)
    gain, mj, mL, mrev = xs.eval(R)
    idx = [i for i in np.argsort(-gain) if gain[i] > 0][:60]
    assert len(idx) > 0
    far = adj = revs = 0
    for i in idx:
        R1 = R.astype(np.int32).copy()
        assert apply_xswap_batch(R1, np.array([i], dtype=np.int64), gain, mj, mL, mrev, n) == 1
        assert len(np.unique(R1)) == n and R1[0] == 0
        real = s0 - score_np(R1, xy, prime)
        assert abs(real - gain[i]) < 1e-6, (i, mj[i], mL[i], mrev[i], gain[i], real)
        far += int(abs(int(mj[i]) - int(i)) > 200); adj += int(abs(int(mj[i]) - int(i)) == int(mL[i])); revs += int(mrev[i] > 0)
    R2, s2, hist = xs.run(R, max_rounds=30)
    assert s2 < s0 and abs(score_np(R2, xy, prime) - s2) < 1e-6
    print("xswap OK:", len(idx), "checked; far (>200 pos.):", far, "adjacent:", adj, "with reversal:", revs, "gain", round(s0 - s2, 1))
