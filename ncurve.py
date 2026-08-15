import sqlite3, re, math
from collections import defaultdict
import numpy as np
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT instance,candidate,reward FROM ladder_trials").fetchall()
def nof(x): return int(re.search(r"(\d+)", x).group(1))
# per (inst,op) sredni reward; per inst n
r_io = defaultdict(list); insts = {}
for inst, cand, rew in rows:
    r_io[(inst, cand)].append(rew); insts[inst] = nof(inst)
ops = sorted(set(c for (_, c) in r_io))
inst_list = sorted(insts, key=lambda k: insts[k])
# macierz sredniego rewardu op x inst, normalizacja wewnatrz instancji (udzial)
share = {}   # (inst,op) -> udzial [0..1]
for inst in inst_list:
    tot = sum(np.mean(r_io[(inst, o)]) for o in ops if (inst, o) in r_io)
    for o in ops:
        if (inst, o) in r_io and tot > 0:
            share[(inst, o)] = np.mean(r_io[(inst, o)]) / tot
# regresja udzial ~ a + b*log10(n) per operator
print("operator   | nachylenie b (log10 n) | znak | udzial@500 | @1500 | @3500")
fits = {}
for o in ops:
    xs = [math.log10(insts[i]) for i in inst_list if (i, o) in share]
    ys = [share[(i, o)] for i in inst_list if (i, o) in share]
    if len(xs) < 5: continue
    b, a = np.polyfit(xs, ys, 1)
    fits[o] = (a, b)
    pv = lambda n: max(0, a + b*math.log10(n))
    arrow = "↑ rośnie" if b > 0.005 else ("↓ maleje" if b < -0.005 else "≈ płaski")
    print("  %-9s| %+21.4f | %-8s| %.3f | %.3f | %.3f" % (o, b, arrow, pv(500), pv(1500), pv(3500)))
# predykowany RANKING preferencji przy danym n (po udziale)
print("\npredykowana preferencja operatorow (top wg udzialu):")
for n in (500, 1500, 3500):
    pref = sorted(fits, key=lambda o: -(fits[o][0] + fits[o][1]*math.log10(n)))
    print("  n=%4d: %s" % (n, " > ".join(pref[:7])))
# walidacja: korelacja rangi (Spearman-lite) udzialu vs n dla kluczowych operatorow
print("\nkierunek (korelacja udzial vs log n):")
for o in ("Or2", "Or3", "K+4", "segvar", "LE", "3opt", "merge"):
    xs = np.array([math.log10(insts[i]) for i in inst_list if (i, o) in share])
    ys = np.array([share[(i, o)] for i in inst_list if (i, o) in share])
    if len(xs) >= 5:
        r = np.corrcoef(xs, ys)[0, 1]
        print("  %-8s r=%+.2f" % (o, r))
