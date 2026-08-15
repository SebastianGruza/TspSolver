"""Discovery: rollout kandydatów per eskalacja -> log do SQLite. Early-stop (3 chunki bez poprawy).
   Użycie: disc_run.py <instancja> [ge_cap|0=250] [seed] [db_path]"""
import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
inst = sys.argv[1] if len(sys.argv) > 1 else "rl5934"
ge = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) > 0 else 250   # cap; early-stop terminuje
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
db = sys.argv[4] if len(sys.argv) > 4 else "results/ladder_trials.db"
c, _, e = load_tsplib("instances/%s.tsp" % inst); D = dist_matrix(c, e)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=ge, seed=seed, ladder=True, discovery=True,
                     trial_chunks=3, db_path=db, decel_win=1, d_min=1, decel_rho=0.5,
                     use_tabu=1, tag=inst, coords=c)
print("%s DISCOVERY(seed=%d): best=%d perm=%s [%.0fs] -> %s" % (inst, seed, b, ok, dt, db), flush=True)
