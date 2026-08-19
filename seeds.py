import sqlite3
from collections import defaultdict
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT DISTINCT instance,seed FROM ladder_trials").fetchall()
byseed = defaultdict(list)
for i, s in rows: byseed[s].append(i)
print("biegow wg seed (ukonczone/rozpoczete):")
for s in sorted(byseed):
    print("  seed %s: %d instancji" % (s, len(byseed[s])))
print("\nunikalnych instancji dotknietych: %d" % len(set(i for i,_ in rows)))
# ile w ladder_runs (ukonczone biegi z early_stop)
done = c.execute("SELECT COUNT(*) FROM ladder_runs").fetchone()[0]
print("ukonczonych biegow (ladder_runs): %d" % done)
