import sqlite3, statistics as st
from collections import Counter, defaultdict
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT instance,seed,step,candidate,burst,dt,reward,state_ops FROM ladder_trials").fetchall()
print("dane: %d wierszy, %d instancji, %d biegow\n" % (
    len(rows), len(set(r[0] for r in rows)), len(set((r[0], r[1]) for r in rows))))
by = defaultdict(list)
for r in rows: by[r[3]].append(r)
step_groups = defaultdict(list)
for r in rows: step_groups[(r[0], r[1], r[2])].append(r)
wins = Counter()
for g in step_groups.values(): wins[max(g, key=lambda r: r[6])[3]] += 1
print("operator   |trials| sr.reward | sr.burst | sr.dt | max-reward-w-kroku")
for cand in sorted(by, key=lambda k: -st.mean([x[6] for x in by[k]])):
    v = by[cand]
    print("  %-9s| %4d | %8.1f | %7.0f | %5.2f | %d" % (
        cand, len(v), st.mean([x[6] for x in v]), st.mean([x[4] for x in v]),
        st.mean([x[5] for x in v]), wins[cand]))
print("\nnajlepszy pierwszy ruch (step0 max-reward):")
op0 = Counter()
for (i, s, stp), g in step_groups.items():
    if stp == 0: op0[max(g, key=lambda r: r[6])[3]] += 1
for cand, n in op0.most_common(): print("  %-9s %d" % (cand, n))
print("\ninterakcja 3opt (reward swiezo vs po Or2/Or3):")
fresh = [r[6] for r in rows if r[3] == "3opt" and r[7] == 0]
after = [r[6] for r in rows if r[3] == "3opt" and (r[7] & 12)]
if fresh: print("  3opt ops=0:      sr.reward=%.1f (n=%d)" % (st.mean(fresh), len(fresh)))
if after: print("  3opt po Or2/Or3: sr.reward=%.1f (n=%d)" % (st.mean(after), len(after)))
print("\nswap/segvar (operatory z Javy) — czy sie broni:")
for cand in ("swap", "segvar"):
    v = by.get(cand, [])
    pos = [x for x in v if x[4] > 0]
    if v: print("  %-7s: %d trials, %d z burst>0 (%.0f%%), max-reward-win=%d" % (
        cand, len(v), len(pos), 100*len(pos)/len(v), wins[cand]))
