"""Tour crossover with exact penalty (a partition-crossover variant on segments): for parents A and B we look for maximal
segments A[i..j] whose set of cities is EXACTLY the set of cities of some segment B[k..l] (same length, same
endpoints, possibly reversed). Substituting A[i..j] := B[k..l] (or its reverse) changes no position outside the segment,
so the cost with penalty is computed exactly on positions i..j. Child = A with the segments substituted where B is
better (greedy choice by gain, disjoint segments). Guarantee: child <= A.
Detection: for every i with increasing j, we track the min/max position in B of the cities A[i..j]; condition maxB-minB == j-i and matching endpoints.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _seg_cost(R, xy, prime, lo, hi, pen):
    """cost of steps lo..hi-1 (position p: R[p] -> R[p+1])."""
    c = 0.0
    for p in range(lo, hi):
        u = R[p]; v = R[p + 1]
        dx = xy[u, 0] - xy[v, 0]; dy = xy[u, 1] - xy[v, 1]
        d = np.sqrt(dx * dx + dy * dy)
        if (p % 10 == 9) and prime[u] == 0:
            d *= (1.0 + pen)
        c += d
    return c


@njit(cache=True)
def find_segments(RA, RB, posB, n, maxlen, minlen, per_i=4):
    """returns arrays (i, j, k, rev) of candidates: A[i..j] == set of B[k..k+(j-i)], rev=1 when B is reversed relative to A.
    Segments IDENTICAL in A and B (same edges in the same order) are skipped - their gain is 0 and there are O(n*maxlen) of them.
    At most per_i candidates per i (consecutive closures)."""
    out_i = []; out_j = []; out_k = []; out_r = []
    for i in range(1, n - 1):
        p0 = posB[RA[i]]; mn = p0; mx = p0
        ident_f = True; ident_r = True; taken = 0
        for j in range(i + 1, min(n - 1, i + maxlen)):
            pb = posB[RA[j]]
            if pb < mn: mn = pb
            if pb > mx: mx = pb
            L = j - i
            if ident_f and pb != p0 + L: ident_f = False
            if ident_r and pb != p0 - L: ident_r = False
            if mx - mn > maxlen:
                break
            if mx - mn != L or L < minlen - 1 or ident_f or ident_r:
                continue
            a0 = RA[i]; a1 = RA[j]
            if RB[mn] == a0 and RB[mx] == a1:
                out_i.append(i); out_j.append(j); out_k.append(mn); out_r.append(0); taken += 1
            elif RB[mn] == a1 and RB[mx] == a0:
                out_i.append(i); out_j.append(j); out_k.append(mn); out_r.append(1); taken += 1
            if taken >= per_i:
                break
    return np.array(out_i, np.int64), np.array(out_j, np.int64), np.array(out_k, np.int64), np.array(out_r, np.int64)


@njit(cache=True)
def _child_seg_cost(RA, RB, xy, prime, i, j, k, rev, pen):
    """cost of steps i-1..j (including the boundary edges) after inserting the B segment at positions i..j."""
    L = j - i
    c = 0.0
    prev = RA[i - 1]
    for t in range(L + 1):
        city = RB[k + L - t] if rev else RB[k + t]
        p = i - 1 + t
        u = prev
        dx = xy[u, 0] - xy[city, 0]; dy = xy[u, 1] - xy[city, 1]
        d = np.sqrt(dx * dx + dy * dy)
        if (p % 10 == 9) and prime[u] == 0:
            d *= (1.0 + pen)
        c += d; prev = city
    u = prev; v = RA[j + 1]
    dx = xy[u, 0] - xy[v, 0]; dy = xy[u, 1] - xy[v, 1]
    d = np.sqrt(dx * dx + dy * dy)
    if (j % 10 == 9) and prime[u] == 0:
        d *= (1.0 + pen)
    return c + d


def crossover(RA, RB, xy, prime, pen=0.1, maxlen=3000, minlen=3, log=None):
    """child from A and B (A = base). Returns (child, gain, number of substitutions, number of candidates)."""
    n = len(RA)
    posB = np.empty(n, np.int64); posB[RB] = np.arange(n)
    si, sj, sk, sr = find_segments(RA.astype(np.int64), RB.astype(np.int64), posB, n, maxlen, minlen)
    gains = np.empty(len(si))
    for t in range(len(si)):
        old = _seg_cost(RA, xy, prime, si[t] - 1, sj[t] + 1, pen)
        new = _child_seg_cost(RA, RB, xy, prime, si[t], sj[t], sk[t], sr[t], pen)
        gains[t] = old - new
    order = np.argsort(-gains)
    used = np.zeros(n, np.uint8); child = RA.copy(); applied = 0; total = 0.0
    for t in order:
        if gains[t] <= 1e-9:
            break
        i, j, k, rev = si[t], sj[t], sk[t], sr[t]
        if used[i:j + 1].any():
            continue
        used[i:j + 1] = 1
        seg = RB[k:k + (j - i) + 1]
        child[i:j + 1] = seg[::-1] if rev else seg
        applied += 1; total += gains[t]
        if log:
            log(f"gpx: A[{i}..{j}] <- B[{k}..] rev={rev} len={j - i + 1} gain={gains[t]:.1f}")
    return child, total, applied, len(si)
