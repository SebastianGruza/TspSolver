"""Alpha candidates (Helsgaun): alpha(i,j) = d(i,j) - max edge on the path i->j in the MST (1-tree approximation without
subgradient). Candidate list = the k pairs with the smallest alpha among KNN(kk) -- ordered by increasing alpha.
MST on the KNN graph (scipy), max edge on the path via the Kruskal reconstruction tree + LCA (binary lifting).
Quality measure: coverage of the LKH tour edges (fraction of tour edges present in the candidate lists) -- as in Helsgaun's papers."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree


def knn_graph(xy, kk):
    t = cKDTree(xy)
    d, idx = t.query(xy, k=kk + 1)
    return idx[:, 1:].astype(np.int64), d[:, 1:]


def mst_edges(xy, kk=16):
    n = len(xy); idx, d = knn_graph(xy, kk)
    rows = np.repeat(np.arange(n), kk); cols = idx.ravel(); w = d.ravel() + 1e-9
    g = coo_matrix((w, (rows, cols)), shape=(n, n))
    g = g.maximum(g.T)                      # symmetrization (max = the same distance)
    T = minimum_spanning_tree(g).tocoo()
    return np.stack([T.row, T.col], 1), T.data, idx, d


def kruskal_tree(n, edges, w):
    """Kruskal reconstruction tree: leaves = cities, inner nodes = MST edges in increasing weight order.
    Max edge on the path (i,j) = weight of node LCA(i,j)."""
    order = np.argsort(w); parent = np.full(2 * n - 1, -1, np.int64); val = np.zeros(2 * n - 1)
    uf = np.arange(n); rep = np.arange(n)          # uf: union-find over cities; rep: representative -> tree node
    def find(x):
        r = x
        while uf[r] != r: r = uf[r]
        while uf[x] != r: uf[x], x = r, uf[x]
        return r
    nxt = n
    for e in order:
        a, b = int(edges[e, 0]), int(edges[e, 1]); ra, rb = find(a), find(b)
        if ra == rb: continue
        parent[rep[ra]] = nxt; parent[rep[rb]] = nxt; val[nxt] = w[e]
        uf[rb] = ra; rep[ra] = nxt; nxt += 1
    return parent[:nxt], val[:nxt]


def lca_maxedge(parent, val, qi, qj):
    """max edge on the path for pairs (qi,qj): LCA in the Kruskal tree via binary lifting."""
    m = len(parent); root_mask = parent < 0
    depth = np.zeros(m, np.int64)
    # depths: topologically from the root (parent index always > child index in the Kruskal construction)
    for v in range(m - 2, -1, -1):
        if parent[v] >= 0: depth[v] = depth[parent[v]] + 1
    LOG = int(np.ceil(np.log2(max(2, depth.max() + 1)))) + 1
    up = np.empty((LOG, m), np.int64); up[0] = np.where(root_mask, np.arange(m), parent)
    for k in range(1, LOG): up[k] = up[k - 1][up[k - 1]]
    a = qi.astype(np.int64).copy(); b = qj.astype(np.int64).copy()
    da, db = depth[a], depth[b]
    swap = da < db; a[swap], b[swap] = b[swap], a[swap]; da, db = depth[a], depth[b]
    diff = da - db
    for k in range(LOG):
        sel = (diff >> k) & 1 == 1
        a[sel] = up[k][a[sel]]
    eq = a == b
    for k in range(LOG - 1, -1, -1):
        ua, ub = up[k][a], up[k][b]; mv = (~eq) & (ua != ub)
        a[mv] = ua[mv]; b[mv] = ub[mv]
    lca = np.where(eq, a, up[0][a])
    return val[lca]


def alpha_candidates(xy, k=8, kk=16):
    """-> neigh [n,k] int32 sorted by increasing alpha (alpha=0 for MST edges)."""
    n = len(xy)
    edges, w, idx, d = mst_edges(xy, kk)
    parent, val = kruskal_tree(n, edges, w)
    qi = np.repeat(np.arange(n), kk); qj = idx.ravel()
    mx = lca_maxedge(parent, val, qi, qj)
    alpha = (d.ravel() + 1e-9) - mx
    alpha = alpha.reshape(n, kk)
    order = np.argsort(alpha, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(idx, order, 1).astype(np.int32), np.take_along_axis(alpha, order, 1)


def edge_coverage(neigh, path):
    """fraction of tour edges (path with closure) contained in the candidate lists (in either direction)."""
    p = np.asarray(path); a = p[:-1]; b = p[1:]
    S = {(int(i), int(j)) for i, row in enumerate(neigh) for j in row}
    hit = sum(1 for i, j in zip(a, b) if (int(i), int(j)) in S or (int(j), int(i)) in S)
    return hit / len(a)
