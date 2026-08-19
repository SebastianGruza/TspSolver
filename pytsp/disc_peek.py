"""Podgląd discovery DB — NAJŚWIEŻSZE NA GÓRZE (ostatnia generacja/step na samej górze).
   Domyślnie v2 (najnowsza kampania), fallback do bazowej.
   Argumenty w dowolnej kolejności: <instancja> <step> <*.db>"""
import sys, os, sqlite3
from collections import defaultdict
db = "results/ladder_trials_v2.db" if os.path.exists("results/ladder_trials_v2.db") else "results/ladder_trials.db"
inst = None; step = None
for a in sys.argv[1:]:
    if a.endswith(".db"): db = a
    elif a.isdigit(): step = int(a)
    else: inst = a
print("DB:", db)
c = sqlite3.connect(db)
# postęp — instancje wg recency (ostatnio pisana na górze)
print("=== postęp (najświeższa instancja na górze) ===")
for r in c.execute("SELECT instance,COUNT(*),MAX(step),MIN(best),MAX(rowid) AS mr "
                   "FROM ladder_trials GROUP BY instance ORDER BY mr DESC"):
    print("  %-8s  %4d krotek  do step %2d  best-so-far=%d" % r[:4])
# szczegóły — grupy (instancja,step) najświeższe najwyżej, wewnątrz wg reward
q = "SELECT rowid,instance,step,state_ops,state_K,candidate,burst,round(dt,2),round(reward,1) FROM ladder_trials"
w = []
if inst: w.append("instance='%s'" % inst)
if step is not None: w.append("step=%d" % step)
if w: q += " WHERE " + " AND ".join(w)
groups = {}
for rid, i2, st, ops, K, cand, burst, dt, rew in c.execute(q):
    g = groups.setdefault((i2, st), {"mr": rid, "ops": ops, "K": K, "rows": []})
    if rid > g["mr"]: g["mr"] = rid
    g["rows"].append((cand, burst, dt, rew))
for key in sorted(groups, key=lambda k: -groups[k]["mr"]):    # najświeższa generacja na górze
    g = groups[key]; i2, st = key
    print("\n[%s] step %d  ops=%d K=%d   (kandydaci wg reward=burst/s):" % (i2, st, g["ops"], g["K"]))
    for cand, burst, dt, rew in sorted(g["rows"], key=lambda r: -r[3]):
        print("   %-10s burst=%7d  dt=%5.2f  reward=%8.1f" % (cand, burst, dt, rew))
