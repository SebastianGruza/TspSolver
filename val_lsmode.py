import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/kroA100.tsp"); D = dist_matrix(c, e)
for m in (0, 1):
    name = "klasyczny" if m == 0 else "relokacje"
    b, ok, dt = solve_ga(D, T=256, grid_epochs=150, use_merge=1, use_tabu=1,
                         use_uniq=1, seed=1, ls_mode=m, tag="m%d" % m)
    gap = (b / 21282 - 1) * 100
    print("kroA100 ls_mode=%d (%s): best=%d gap=%.3f%% perm=%s [%.0fs]"
          % (m, name, b, gap, ok, dt), flush=True)
