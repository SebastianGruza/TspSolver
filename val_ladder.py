import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/kroA100.tsp"); D = dist_matrix(c, e)
b, ok, dt = solve_ga(D, T=256, grid_epochs=150, seed=1, ladder=True, tag="lad")
print("kroA100 DRABINA: best=%d gap=%.3f%% perm=%s [%.0fs]" % (b, (b/21282-1)*100, ok, dt), flush=True)
