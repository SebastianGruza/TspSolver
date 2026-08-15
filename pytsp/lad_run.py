import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/rl5934.tsp"); D = dist_matrix(c, e); opt = 556045
print("DRABINA DROBNOZIARNISTA rl5934 (atomowe ruchy), ge=500 T=8192  ANALIZA PO CZASIE", flush=True)
print("ref hybryda best(t): 578473@767s 575772@1710s 570023@2756s 568532@4174s (=2.246%%)", flush=True)
b, ok, dt = solve_ga(D, T=8192, grid_epochs=500, seed=1, ladder=True, use_tabu=1, tag="lad",
                     decel_win=1, d_min=1, decel_rho=0.5)
print("rl5934 DRABINA: best=%d gap=%.3f%% perm=%s [%.0fs]" % (b, (b/opt-1)*100, ok, dt), flush=True)
