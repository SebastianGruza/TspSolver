"""Traveling Santa 2018 -- Prime Paths: data, primes, score, validation, submission writing.

Rules (Kaggle): the tour starts and ends at CityId 0 (the pole) and visits every city once. Cost = sum of Euclidean
distances (float), and EVERY TENTH step (step no. 10, 20, 30, ... counting from 1) costs 10% more unless the step
STARTS at a city with a prime number. Step no. k goes from path[k-1] to path[k]; penalty when k % 10 == 0 and
not prime(path[k-1]). So the penalty is decided by the city at POSITION 9, 19, 29, ... (0-indexed) in the tour.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

DATA = Path.home() / "santa" / "data"


def primes_upto(n: int) -> np.ndarray:
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = False
    return s


def load_cities(path: Path | str = DATA / "cities.csv") -> np.ndarray:
    """-> xy float64 [n, 2], index = CityId (0 = the pole)."""
    p = Path(path)
    if p.with_suffix(".npy").exists():
        return np.load(p.with_suffix(".npy"))
    rows = []
    with open(p, newline="") as f:
        r = csv.reader(f); next(r)
        for cid, x, y in r:
            rows.append((int(cid), float(x), float(y)))
    rows.sort()
    assert [c for c, _, _ in rows] == list(range(len(rows)))
    xy = np.array([(x, y) for _, x, y in rows], dtype=np.float64)
    np.save(p.with_suffix(".npy"), xy)
    return xy


def synth_cities(n: int = 197_769, seed: int = 0) -> np.ndarray:
    """Synthetic data for development (distribution like the original: rectangle ~5100x3400, clusters)."""
    rng = np.random.default_rng(seed)
    k = 400
    centers = rng.uniform([0, 0], [5100, 3400], size=(k, 2))
    lab = rng.integers(0, k, n)
    xy = centers[lab] + rng.normal(0, 60, size=(n, 2))
    uni = rng.random(n) < 0.3                       # 30% of points scattered uniformly
    xy[uni] = rng.uniform([0, 0], [5100, 3400], size=(int(uni.sum()), 2))
    xy = np.clip(xy, 0, [5100, 3400]); xy[0] = [316.8, 2202.3]
    return xy


def score(path: np.ndarray, xy: np.ndarray, is_prime: np.ndarray) -> float:
    """path: int array [n+1], path[0]==path[-1]==0. Returns the cost with penalties (float64)."""
    p = np.asarray(path, dtype=np.int64)
    d = np.sqrt(((xy[p[1:]] - xy[p[:-1]]) ** 2).sum(1))
    k = np.arange(1, len(p))
    pen = (k % 10 == 0) & (is_prime[p[:-1]] == 0)          # works for bool and uint8
    return float(d.sum() + 0.1 * d[pen].sum())


def plain_len(path: np.ndarray, xy: np.ndarray) -> float:
    p = np.asarray(path, dtype=np.int64)
    return float(np.sqrt(((xy[p[1:]] - xy[p[:-1]]) ** 2).sum(1)).sum())


def validate(path: np.ndarray, n: int) -> None:
    p = np.asarray(path, dtype=np.int64)
    assert len(p) == n + 1, f"length {len(p)} != {n + 1}"
    assert p[0] == 0 and p[-1] == 0, "the tour must start and end at 0"
    seen = np.zeros(n, dtype=bool); seen[p[:-1]] = True
    assert seen.all() and len(np.unique(p[:-1])) == n, "not a permutation"


def write_submission(path: np.ndarray, out: Path | str) -> None:
    with open(out, "w") as f:
        f.write("Path\n")
        f.write("\n".join(map(str, np.asarray(path).tolist())) + "\n")


def read_tour(path_file: Path | str) -> np.ndarray:
    """submission.csv (Path) or an LKH .tour file (TOUR_SECTION, 1-indexed, -1 terminates)."""
    txt = Path(path_file).read_text().splitlines()
    if txt and txt[0].strip() == "Path":
        p = np.array([int(x) for x in txt[1:] if x.strip()], dtype=np.int64)
        return p
    ids = []; on = False
    for l in txt:
        s = l.strip()
        if s == "TOUR_SECTION": on = True; continue
        if on:
            if s in ("-1", "EOF"): break
            ids.append(int(s) - 1)
    p = np.array(ids, dtype=np.int64)
    z = int(np.where(p == 0)[0][0]); p = np.concatenate([p[z:], p[:z], [0]])
    return p


def write_tsplib(xy: np.ndarray, out: Path | str, name="santa", scale=1000) -> None:
    """LKH: EUC_2D computes distances as nint(d); we scale by x scale so rounding does not spoil the geometry."""
    with open(out, "w") as f:
        f.write(f"NAME : {name}\nTYPE : TSP\nDIMENSION : {len(xy)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(xy):
            f.write(f"{i + 1} {x * scale:.0f} {y * scale:.0f}\n")
        f.write("EOF\n")
