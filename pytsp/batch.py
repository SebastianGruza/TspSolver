"""Nocny przelot instancji 2k+ — od największej (gr9882) do najmniejszej (d2103).

Budżet: STAŁY ge per instancja (kilkukrotnie mniejszy niż oryginał). Wynik
dopisywany do CSV PO KAŻDEJ instancji (można podglądać w trakcie, nie czekając
na ostatnią). Odporny na błędy: porażka jednej instancji nie zatrzymuje reszty.

Konfiguracja przez ENV:
  T=8192 GE=80 MERGE=1 TABU=0 OUT=results_2k.csv INST_DIR=instances

Uruchom (gpu-node):
  export CUDA_HOME=$HOME/cudashim
  cd ~/TspSolver
  T=8192 GE=80 MERGE=1 TABU=0 .venv/bin/python pytsp/batch.py
"""
import os, sys, time, csv
sys.path.insert(0, os.path.dirname(__file__))
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga

# (nazwa, optimum wg README oryginału) — MALEJĄCO po n
# UWAGA gr9882: plik z uwaterloo (EUC_2D). Jeśli to nie Twój plik -> podmień.
INSTANCES = [
    ("gr9882", 300899), ("rl5934", 556045), ("rl5915", 565530),
    ("fnl4461", 182566), ("fl3795", 28772), ("pcb3038", 137694),
    ("pr2392", 378032), ("u2319", 234256), ("u2152", 64253), ("d2103", 80450),
]

T     = int(os.environ.get("T", 8192))
GE    = int(os.environ.get("GE", 80))
MERGE = int(os.environ.get("MERGE", 1))
TABU  = int(os.environ.get("TABU", 0))
SEED  = int(os.environ.get("SEED", 1))
CHUNK = int(os.environ.get("CHUNK", 20))          # podgląd best-so-far co CHUNK epok
UNIQ  = int(os.environ.get("UNIQ", 0))            # kontrola unikalności per kolonia (kara duplikatów)
K1 = int(os.environ.get("K1", 0))                 # K per próg budżetu; 0 => auto z n
K2 = int(os.environ.get("K2", 0))                 #   k1=max(5,n//600) k2=max(10,n//300)
K3 = int(os.environ.get("K3", 0))                 #   k3=max(15,n//200)
INST_DIR = os.environ.get("INST_DIR", "instances")
OUT   = os.environ.get("OUT", "results_2k.csv")


def main():
    only = set(sys.argv[1:])  # opcjonalnie: policz tylko wskazane nazwy
    todo = [(n, o) for (n, o) in INSTANCES if not only or n in only]
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["instance", "n", "best", "opt", "gap_pct", "time_s",
                        "cum_min", "T", "ge", "merge", "tabu", "when"])
            fh.flush()
    print(f"[batch] {len(todo)} instancji | T={T} ge={GE} merge={MERGE} tabu={TABU} -> {OUT}",
          flush=True)
    t_start = time.time()
    for name, opt in todo:
        path = f"{INST_DIR}/{name}.tsp"
        try:
            coords, _, ewt = load_tsplib(path)
            D = dist_matrix(coords, ewt); n = D.shape[0]
            best, perm_ok, dt = solve_ga(D, T=T, grid_epochs=GE, use_merge=MERGE,
                                         use_tabu=TABU, seed=SEED, chunk=CHUNK,
                                         verbose=True, tag=name, use_uniq=UNIQ,
                                         k1=K1, k2=K2, k3=K3)
            gap = (best / opt - 1.0) * 100.0
            cum = (time.time() - t_start) / 60.0
            row = [name, n, best, opt, f"{gap:.3f}", f"{dt:.1f}", f"{cum:.1f}",
                   T, GE, MERGE, TABU, time.strftime("%Y-%m-%d %H:%M:%S")]
            flag = "" if perm_ok else "  !!! NIE-PERMUTACJA"
            print(f"[batch] {name}: n={n} best={best} opt={opt} gap={gap:.3f}% "
                  f"[{dt:.0f}s, cum {cum:.0f}min]{flag}", flush=True)
        except Exception as e:
            cum = (time.time() - t_start) / 60.0
            row = [name, "ERR", "", opt, "", "", f"{cum:.1f}", T, GE, MERGE, TABU,
                   f"ERROR: {str(e)[:80]}"]
            print(f"[batch] {name}: BŁĄD -> {e}", flush=True)
        with open(OUT, "a", newline="") as fh:          # DOPISZ po każdej
            csv.writer(fh).writerow(row); fh.flush()
    print(f"[batch] DONE — {len(todo)} instancji w {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
