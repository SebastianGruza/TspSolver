"""Warstwa danych TSP — loader TSPLIB (EUC_2D) + full.txd (miasta PL),
macierz odległości (int32, zaokrąglenie TSPLIB nint), inicjalizacja NN.

Odległości int32 z zaokrągleniem round(sqrt(dx²+dy²)) — konwencja TSPLIB EUC_2D,
konieczna by trafiać w publikowane optima (berlin52=7542 itd.).
"""
import numpy as np


def load_tsplib(path):
    """Parsuje TSPLIB EUC_2D (NODE_COORD_SECTION). Zwraca (coords[n,2] float, name)."""
    name = "tsp"; coords = []; in_nodes = False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("NAME"):
                name = s.split(":", 1)[-1].strip() or name
            elif up.startswith("NODE_COORD_SECTION"):
                in_nodes = True
            elif up.startswith("EOF"):
                break
            elif in_nodes:
                parts = s.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return np.asarray(coords, dtype=np.float64), name


def load_txd(path):
    """Parsuje full.txd (miasta PL: 'Nazwa;X;Y', 1. linia licznik, 2. nagłówek)."""
    coords = []; names = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        lines = fh.read().splitlines()
    for line in lines:
        if ";" not in line:
            continue
        parts = line.split(";")
        if len(parts) < 3:
            continue
        try:
            x = float(parts[-2]); y = float(parts[-1])
        except ValueError:
            continue  # nagłówek
        names.append(parts[0]); coords.append((x, y))
    return np.asarray(coords, dtype=np.float64), names


def dist_matrix(coords, euc2d_round=True):
    """Pełna macierz odległości. euc2d_round → int32 round(sqrt) (konwencja TSPLIB)."""
    d = coords[:, None, :] - coords[None, :, :]
    dm = np.sqrt((d * d).sum(-1))
    if euc2d_round:
        return np.rint(dm).astype(np.int32)
    return dm.astype(np.float32)


def nn_tour(D, start=0):
    """Trasa nearest-neighbor (do inicjalizacji / sanity)."""
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    cur = start; visited[cur] = True; tour[0] = cur
    for i in range(1, n):
        row = D[cur].copy()
        row[visited] = np.iinfo(D.dtype).max if np.issubdtype(D.dtype, np.integer) else np.inf
        cur = int(np.argmin(row)); visited[cur] = True; tour[i] = cur
    return tour


def tour_len(D, tour):
    """Długość zamkniętej trasy (suma krawędzi + powrót)."""
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


if __name__ == "__main__":
    import sys, os
    for path in sys.argv[1:]:
        if path.endswith(".txd"):
            coords, names = load_txd(path); name = os.path.basename(path)
        else:
            coords, name = load_tsplib(path)
        D = dist_matrix(coords)
        nn = nn_tour(D, 0)
        # sanity: trasa jest permutacją
        assert sorted(nn.tolist()) == list(range(len(coords))), "NN nie jest permutacją!"
        print(f"{name}: n={len(coords)}  NN_len={tour_len(D, nn)}  "
              f"D[int32 {D.nbytes/1e6:.1f}MB]  coords[{coords[0]}]")
