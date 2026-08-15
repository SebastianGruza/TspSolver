import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/kroA100.tsp"); D = dist_matrix(c, e)
b, ok, dt = solve_ga(D, T=256, grid_epochs=120, use_merge=1, use_tabu=1, use_uniq=1,
                     seed=1, ls_mode=3, ox_mode=0, k1=8, k2=8, k3=8, tag="fast")
print("kroA100 ls_mode=3 ultra-fast: best=%d gap=%.3f%% perm=%s [%.0fs]"
      % (b, (b/21282-1)*100, ok, dt), flush=True)
