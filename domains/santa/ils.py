"""Iterated local search with LOCAL acceptance (for the GPU):
1. pick disjoint position windows [s, s+w) every ~2w with a random offset; in each window a double-bridge kick
   (4 cuts inside the window, segments B,C swapped) -- the cost changes only inside the window;
2. GPU LS (penalty-aware Or-opt/2-opt) to convergence (a few rounds);
3. per window: cost of the steps in the window after vs before; worse windows -> restore the saved window contents (the LS
   may have moved cities outside the window via Or-opt to a neighbour -- hence we compare and revert on the EXTENDED window
   [s-W, s+w+W) and revert only when the contents of the extended window are a permutation of the same contents; otherwise
   we leave it -- globally we check anyway that the total cost did not increase, and on an increase revert everything).
Repeat. The best tour is saved after every improving iteration.
"""
from __future__ import annotations

import time

import numpy as np

from domains.santa.gpu_ls import PEN, score_np


def window_costs(R, xy, prime, starts, w):
    """cost of the steps leaving positions [s, s+w) for each window."""
    n = len(R); out = np.empty(len(starts))
    for k, s in enumerate(starts):
        p = np.arange(s, min(s + w, n)); q = (p + 1) % n
        d = np.sqrt(((xy[R[q]] - xy[R[p]]) ** 2).sum(1))
        pen = (p % 10 == 9) & (prime[R[p]] == 0)
        out[k] = d.sum() + PEN * d[pen].sum()
    return out


def edge_badness(R, xy, prime, nn1):
    """"badness" of the edge leaving position p: excess of its length over the distance to the nearest neighbour of city R[p]
    (+ penalty, if the step is penalized). Targeted kicks (idea: kick where the tour is bad, not randomly across the continent)."""
    n = len(R); p = np.arange(n - 1); q = p + 1
    d = np.sqrt(((xy[R[q]] - xy[R[p]]) ** 2).sum(1))
    b = np.maximum(0.0, d - nn1[R[p]])
    pen = (p % 10 == 9) & (prime[R[p]] == 0)
    b[pen] += PEN * d[pen]
    out = np.zeros(n); out[:n - 1] = b
    return out


def double_bridge_window(R, s, w, rng, weights=None):
    """4 random cuts in [s+1, s+w): A B C D -> A C B D (changes only positions inside the window); weights = cut
    probability proportional to edge badness (targeted) or None (uniform)."""
    if w < 12:
        return
    if weights is not None:
        wgt = weights[s + 1:s + w] + 1e-3 * (weights[s + 1:s + w].mean() + 1e-9)
        cuts = np.sort(rng.choice(np.arange(s + 1, s + w), size=3, replace=False, p=wgt / wgt.sum()))
    else:
        cuts = np.sort(rng.choice(np.arange(s + 1, s + w), size=3, replace=False))
    a, b, c = cuts
    seg = np.concatenate([R[s:a], R[b:c], R[a:b], R[c:s + w]])
    R[s:s + w] = seg


def ils(R, xy, prime, ls, *, w=400, margin=300, iters=200, kick_frac=1.0, seed=0, log=None, time_limit=None,
        save_cb=None, ls_rounds=8, pen=None, stall_iters=40, stall_gain=150.0, kick="uniform"):
    """stall: when the improvement of best over the last stall_iters iterations < stall_gain -> STOP (plateau; GPU budget goes to experiments).
    kick: 'uniform' = windows from the grid, random cuts; 'bad' = windows drawn with probability proportional to window badness
    (kick_frac share of them), cuts proportional to edge badness."""
    n = len(R); rng = np.random.default_rng(seed); R = R.astype(np.int32).copy()
    nn1 = None
    if kick == "bad":
        neigh = getattr(ls, "neigh", None)
        if neigh is None and hasattr(ls, "ls"): neigh = getattr(ls.ls, "neigh", None)
        nn1 = np.sqrt(((xy[np.asarray(neigh)[:, 0]] - xy) ** 2).sum(1)) if neigh is not None else np.zeros(len(xy))
    best = score_np(R, xy, prime); t0 = time.time(); acc_hist = []
    stride = w + 2 * margin
    for it in range(iters):
        off = int(rng.integers(0, stride)); starts = np.arange(off, n - w, stride)
        starts = starts[starts > 0]
        bad = None
        if kick == "bad":
            bad = edge_badness(R, xy, prime, nn1)
            cs = np.concatenate([[0.0], np.cumsum(bad)]); wb = cs[starts + w] - cs[starts]
            m = max(1, int(round(len(starts) * min(kick_frac, 1.0))))
            pr = wb + 1e-3 * (wb.mean() + 1e-9)
            starts = np.sort(rng.choice(starts, size=min(m, len(starts)), replace=False, p=pr / pr.sum()))
        elif kick_frac < 1.0:
            starts = starts[rng.random(len(starts)) < kick_frac]
        R_before = R.copy()
        # extended windows for comparison/revert
        lo = np.maximum(starts - margin, 1); hi = np.minimum(starts + w + margin, n)
        cost_before = np.array([window_costs(R_before, xy, prime, [l], h - l)[0] for l, h in zip(lo, hi)])
        for s in starts:
            double_bridge_window(R, s, w, rng, bad)
        if pen is not None and pen < PEN:
            # the doctor's recipe: search "as if there were no penalty" (the relaxed tour directly, not the best at the full price),
            # THEN return to the full price (LS/LK repair with penalty 0.1) and only then evaluate the windows honestly
            try:
                R1, _, _ = ls.run(R, max_rounds=ls_rounds, min_gain=1e-6, log=None, pen=pen, keep_best=False)
            except TypeError:
                R1, _, _ = ls.run(R, max_rounds=ls_rounds, min_gain=1e-6, log=None, pen=pen)
            R2, s_after, hist = ls.run(R1, max_rounds=ls_rounds, min_gain=1e-6, log=None)
        else:
            R2, s_after, hist = ls.run(R, max_rounds=ls_rounds, min_gain=1e-6, log=None, **({"pen": pen} if pen is not None else {}))
        cost_after = np.array([window_costs(R2, xy, prime, [l], h - l)[0] for l, h in zip(lo, hi)])
        worse = cost_after > cost_before + 1e-9
        reverted = 0
        for k in np.where(worse)[0]:
            l, h = lo[k], hi[k]
            if np.array_equal(np.sort(R2[l:h]), np.sort(R_before[l:h])):   # same contents -> safe revert
                R2[l:h] = R_before[l:h]; reverted += 1
        s_new = score_np(R2, xy, prime)
        if s_new < best - 1e-9:
            best = s_new; R = R2
            if save_cb:
                save_cb(R, best)
        else:
            R = R_before                                   # global safety net
        acc_hist.append((it, len(starts), int(worse.sum()), reverted, best, time.time() - t0))
        if stall_iters and len(acc_hist) > stall_iters and acc_hist[-1 - stall_iters][4] - best < stall_gain:
            if log:
                log(f"ils STOP: plateau ({acc_hist[-1 - stall_iters][4] - best:.1f} < {stall_gain} in {stall_iters} iterations)")
            break
        if log:
            log(f"ils {it}: windows={len(starts)} worse={int(worse.sum())} reverted={reverted} best={best:.1f} "
                f"t={time.time() - t0:.0f}s")
        if time_limit and time.time() - t0 > time_limit:
            break
    return R, best, acc_hist
