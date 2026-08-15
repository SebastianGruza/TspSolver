import sys; sys.path.insert(0, "pytsp")
from tsp_io import load_tsplib, dist_matrix
from memetic_ga import solve_ga
c, _, e = load_tsplib("instances/fnl4461.tsp"); D = dist_matrix(c, e); opt = 182566
n = D.shape[0]
k1 = round(max(8, n // 600) * 1.2)
k2 = round(max(12, n // 300) * 1.2)
k3 = round(max(15, n // 200) * 1.2)
print("fnl4461: n=%d  K(+20%%,progi 8/12)=%d/%d/%d  opt=%d" % (n, k1, k2, k3, opt), flush=True)
for m, name in ((0, "klasyk"), (2, "hybryda@0.6")):
    b, ok, dt = solve_ga(D, T=8192, grid_epochs=100, use_merge=1, use_tabu=1, use_uniq=1,
                         seed=1, ls_mode=m, opt_start=0.6, le_start=0.0,
                         k1=k1, k2=k2, k3=k3, chunk=25, verbose=True, tag=name)
    print("fnl4461 %s: best=%d gap=%.3f%% perm=%s [%.0fs]"
          % (name, b, (b/opt-1)*100, ok, dt), flush=True)
