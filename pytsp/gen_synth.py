"""Generator syntetycznych instancji TSP z REALNYCH miast (GeoNames per-country dump).
   Realna struktura (klastrowanie/gęstość), subsample do n log-jednostajnie w [N_MIN, N_MAX].
   Projekcja lat/lon -> płaski km (equirectangular, centr. na centroidzie) -> TSPLIB EUC_2D.
   Uzycie: gen_synth.py [n_instancji=180] [seed=12345]
   'z wiekszego mniejsze' = random subsample; 'z mniejszego wieksze' = augment (rzadko, z umiarem)."""
import os, sys, zipfile, math, csv
import numpy as np

GEO = os.path.expanduser("~/geonames")
OUT = os.path.expanduser("~/TspSolver/instances/synth")
os.makedirs(OUT, exist_ok=True)
R_EARTH = 6371.0
N_MIN, N_MAX = 400, 15000
N_INSTANCES = int(sys.argv[1]) if len(sys.argv) > 1 else 180
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
SCALE = 10.0                                   # km -> jednostki (0.1 km rozdzielczosc po rint EUC_2D)
CC = "PL DE FR IT ES GB RO UA SE NO FI GR PT CZ NL HU US CA MX BR AR AU IN RU JP TR ZA EG".split()

def load_country(cc):
    """(lat,lon) [Nx2] populated places (feature_class P), zdeduplikowane ~100m."""
    zp = os.path.join(GEO, cc + ".zip")
    if not os.path.exists(zp):
        return None
    lat, lon = [], []
    with zipfile.ZipFile(zp) as z, z.open(cc + ".txt") as fh:
        for raw in fh:
            p = raw.split(b"\t")
            if len(p) < 15 or p[6] != b"P":
                continue
            try:
                la = float(p[4]); lo = float(p[5])
            except ValueError:
                continue
            lat.append(la); lon.append(lo)
    if not lat:
        return None
    a = np.column_stack([lat, lon])
    key = np.round(a * 1000).astype(np.int64)          # dedup do 0.001 st (~100m)
    _, idx = np.unique(key[:, 0] * 10_000_000 + key[:, 1], return_index=True)
    return a[np.sort(idx)]

def project(latlon):
    """equirectangular -> km * SCALE, centrowane na centroidzie."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    lat0, lon0 = lat.mean(), lon.mean()
    x = R_EARTH * np.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = R_EARTH * np.radians(lat - lat0)
    return np.column_stack([x, y]) * SCALE

def augment(base, need, rng):
    """dolej 'need' punktow: resample istniejacych + jitter 1% rozpietosci (zachowuje gestosc)."""
    pts = base[rng.integers(0, len(base), size=need)].astype(float)
    span = base.max(0) - base.min(0)
    return pts + rng.normal(0, 0.01, pts.shape) * span

def write_tsp(path, name, xy):
    with open(path, "w") as f:
        f.write("NAME: %s\nTYPE: TSP\nDIMENSION: %d\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
                % (name, len(xy)))
        for i, (x, y) in enumerate(xy, 1):
            f.write("%d %.3f %.3f\n" % (i, x, y))
        f.write("EOF\n")

# --- wczytaj kraje ---
pool = {}
for cc in CC:
    d = load_country(cc)
    if d is not None and len(d) >= N_MIN:
        pool[cc] = d
        print("  %s: %d miast" % (cc, len(d)))
ccs = sorted(pool, key=lambda c: -len(pool[c]))
print("krajow uzytych: %d, laczna pula miast: %d" % (len(ccs), sum(len(pool[c]) for c in ccs)))

# --- generuj: rowna siatka log(n), kraje rotacyjnie (najrzadziej uzyty -> diversity) ---
rng = np.random.default_rng(SEED)
logs = np.linspace(math.log10(N_MIN), math.log10(N_MAX), N_INSTANCES)
rng.shuffle(logs)
used = {c: 0 for c in ccs}
rows = []
for lg in logs:
    n = max(N_MIN, min(N_MAX, int(round(10 ** lg))))
    cand = [c for c in ccs if len(pool[c]) >= n]
    if cand:
        c = min(cand, key=lambda x: used[x])
        latlon = pool[c][rng.choice(len(pool[c]), size=n, replace=False)]
        src = "subsample"
    else:                                              # nikt nie ma tylu -> najwiekszy + augment (z umiarem)
        c = ccs[0]; need = n - len(pool[c])
        latlon = np.vstack([pool[c], augment(pool[c], need, rng)])
        src = "augment+%d" % need
    used[c] += 1
    name = "syn_%s_%d_%02d" % (c, n, used[c])
    write_tsp(os.path.join(OUT, name + ".tsp"), name, project(latlon))
    rows.append((name, c, n, src))

with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "country", "n", "source"]); w.writerows(rows)
aug = sum(1 for r in rows if r[3].startswith("augment"))
print("wygenerowano %d instancji (%d subsample, %d augment) -> %s" % (len(rows), len(rows) - aug, aug, OUT))
print("rozklad krajow:", {c: used[c] for c in ccs if used[c]})
