import sqlite3, re
c = sqlite3.connect("results/ladder_trials_v3.db")
done = c.execute("SELECT instance,n,total_time FROM ladder_runs").fetchall()
tot = sum(t for _,_,t in done)
print("ukonczono %d biegow, laczny czas %.0f min (sr %.1f min/bieg)" % (len(done), tot/60, tot/60/len(done)))
# lista wszystkich 36 wg kolejnosci w skrypcie, oznacz zrobione
INSTS = "fl417 gr431 pr439 pcb442 d493 att532 ali535 u574 rat575 p654 d657 gr666 u724 rat783 pr1002 u1060 vm1084 pcb1173 d1291 rl1304 rl1323 nrw1379 fl1400 u1432 fl1577 d1655 vm1748 u1817 rl1889 d2103 u2152 u2319 pr2392 pcb3038 fl3795 fnl4461".split()
doneset = set(i for i,_,_ in done)
rem = [i for i in INSTS if i not in doneset]
def nof(x): return int(re.search(r"(\d+)", x).group(1))
print("\npozostalo w seedzie 1: %d instancji (te WIEKSZE):" % len(rem))
print("  " + " ".join(rem))
# czas ~ skaluje z n; grube oszacowanie pozostalych po sredniej na 1000 wierzcholkow
tpn = tot / sum(nof(i) for i,_,_ in done)   # s na wierzcholek
est_rem_seed1 = sum(nof(i) for i in rem) * tpn
est_full_seed = sum(nof(i) for i in INSTS) * tpn
print("\nszac. reszta seed 1: ~%.0f min | pelen seed (36): ~%.0f min | 3 seedy: ~%.1f h" % (
    est_rem_seed1/60, est_full_seed/60, est_full_seed*3/3600))
