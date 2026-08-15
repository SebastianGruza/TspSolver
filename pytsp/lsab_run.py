import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
for m in (0, 1):
    name = "klasyczny(2/3-opt)" if m == 0 else "relokacje(Or1/2/3+longedge)"
    b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1,
                         use_uniq=1, seed=1, ls_mode=m, chunk=25, verbose=True,
                         tag="m%d" % m)
    print("rl5934 ls_mode=%d %s: best=%d gap=%.3f%% perm=%s [%.0fs]  (baseline adaptK 2.645%%)"
          % (m, name, b, (b/opt-1)*100, ok, dt), flush=True)
