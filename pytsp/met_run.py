import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
print("METRYKI rl5934 ultra-fast (ls_mode=3, K=8, OX klasyk), chunk=5, ge=100  opt=%d" % opt, flush=True)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1, use_uniq=1,
                     seed=1, ls_mode=3, ox_mode=0, k1=8, k2=8, k3=8,
                     chunk=5, metrics=True, tag="fast")
print("rl5934 ultra-fast final: best=%d gap=%.3f%% perm=%s [%.0fs]"
      % (b, (b/opt-1)*100, ok, dt), flush=True)
