import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
n = D.shape[0]
k1 = round(max(8, n // 600) * 1.2); k2 = round(max(12, n // 300) * 1.2); k3 = round(max(15, n // 200) * 1.2)
print("greedy-edge rl5934: preset hybryda@0.6 K+20%%=%d/%d/%d  (baseline OX-klasyk=2.246%%, OX-mix50=2.169%%)"
      % (k1, k2, k3), flush=True)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1, use_uniq=1,
                     seed=1, ls_mode=2, opt_start=0.6, le_start=0.0, ox_mode=3,
                     k1=k1, k2=k2, k3=k3, chunk=25, verbose=True, tag="greedy-edge")
print("rl5934 greedy-edge: best=%d gap=%.3f%% perm=%s [%.0fs]  (vs OX-klasyk 2.246%%)"
      % (b, (b/opt-1)*100, ok, dt), flush=True)
