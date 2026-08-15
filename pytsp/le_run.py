import sys, os; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
LATE = float(os.environ.get("LE_LATE", "0.6"))   # próg późnego włączenia (0.6 lub 0.75)
arms = [("base(off)", 2.0), ("uniform", 0.0), ("late@%.2f" % LATE, LATE)]
for name, ls in arms:
    b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1,
                         use_uniq=1, seed=1, ls_mode=1, le_start=ls, chunk=25,
                         verbose=True, tag="le_%s" % name.split("(")[0].split("@")[0])
    print("rl5934 long-edge=%s (le_start=%.2f): best=%d gap=%.3f%% perm=%s [%.0fs]"
          % (name, ls, b, (b/opt-1)*100, ok, dt), flush=True)
