import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/kroA100.tsp"); D = dist_matrix(c, e)
for ox in (3,):
    b, ok, dt = solve_ga(D, T=256, grid_epochs=150, use_merge=1, use_tabu=1, use_uniq=1,
                         seed=1, ls_mode=2, opt_start=0.6, ox_mode=ox, tag="ge")
    print("kroA100 ox_mode=3 greedy-edge: best=%d gap=%.3f%% perm=%s [%.0fs]"
          % (b, (b/21282-1)*100, ok, dt), flush=True)
