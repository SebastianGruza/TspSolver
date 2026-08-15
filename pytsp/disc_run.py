"""Discovery: rollout kandydatów per eskalacja -> log do SQLite. Uruchom na wielu instancjach.
   Użycie: disc_run.py <instancja> [ge]   (np. disc_run.py rl5934 300)"""
import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
inst = sys.argv[1] if len(sys.argv) > 1 else "rl5934"
ge = int(sys.argv[2]) if len(sys.argv) > 2 else 300
c, _, e = load_tsplib("instances/%s.tsp" % inst); D = dist_matrix(c, e)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=ge, seed=1, ladder=True, discovery=True,
                     trial_chunks=2, db_path="results/ladder_trials.db",
                     decel_win=1, d_min=1, decel_rho=0.5, use_tabu=1, tag=inst)
print("%s DISCOVERY: best=%d perm=%s [%.0fs]  -> results/ladder_trials.db" % (inst, b, ok, dt), flush=True)
