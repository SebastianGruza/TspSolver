import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
n = D.shape[0]
# K auto n-scaled, podniesione o 20%
k1 = round(max(5, n // 600) * 1.2)
k2 = round(max(10, n // 300) * 1.2)
k3 = round(max(15, n // 200) * 1.2)
print("hybryda: n=%d  K(+20%%)=%d/%d/%d  klasyk od startu, relokacje RAZEM od 60%%  (klasyk=2.855%%)"
      % (n, k1, k2, k3), flush=True)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1, use_uniq=1,
                     seed=1, ls_mode=2, opt_start=0.6, le_start=0.0,
                     k1=k1, k2=k2, k3=k3, chunk=25, verbose=True, tag="hyb")
print("rl5934 HYBRYDA (2/3-opt od startu + relokacje@0.6 RAZEM, K+20%%): best=%d gap=%.3f%% perm=%s [%.0fs]"
      % (b, (b/opt-1)*100, ok, dt), flush=True)
