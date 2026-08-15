import sqlite3, statistics as st, re
from collections import defaultdict
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT instance,seed,step,candidate,burst,dt,reward,state_ops FROM ladder_trials").fetchall()
def nof(name):
    m = re.search(r"(\d+)", name); return int(m.group(1)) if m else 0
step_groups = defaultdict(list)
for r in rows: step_groups[(r[0], r[1], r[2])].append(r)
# per-instancja: K+4 reward + win-rate, posortowane wg n
per = defaultdict(lambda: {"k4": [], "wins": 0, "steps": 0})
for (i, s, stp), g in step_groups.items():
    per[i]["steps"] += 1
    if max(g, key=lambda r: r[6])[3] == "K+4": per[i]["wins"] += 1
for r in rows:
    if r[3] == "K+4": per[r[0]]["k4"].append(r[6])
print("instancja |    n | K+4 sr.reward | K+4 wins | win-rate")
data = []
for i in sorted(per, key=lambda k: nof(k)):
    n = nof(i); k4 = per[i]["k4"]
    rew = st.mean(k4) if k4 else 0
    wr = 100*per[i]["wins"]/per[i]["steps"]
    data.append((n, rew, per[i]["wins"], wr))
    print("  %-9s|%5d | %12.1f | %6d   | %.0f%%" % (i, n, rew, per[i]["wins"], wr))
# trend: korelacja n vs K+4 reward i n vs win-rate
def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = (sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5
    return num/den if den else 0
ns = [d[0] for d in data]
print("\nkorelacja n vs K+4 reward:   r=%+.2f" % corr(ns, [d[1] for d in data]))
print("korelacja n vs K+4 win-rate: r=%+.2f" % corr(ns, [d[3] for d in data]))
# porownanie: mala polowa vs duza polowa instancji
data.sort()
half = len(data)//2
small = data[:half]; big = data[half:]
print("\n              | K+4 sr.reward | K+4 sr.win-rate")
print("male  (n<=%4d)| %12.1f  | %.0f%%" % (small[-1][0], st.mean([d[1] for d in small]), st.mean([d[3] for d in small])))
print("duze  (n>=%4d)| %12.1f  | %.0f%%" % (big[0][0], st.mean([d[1] for d in big]), st.mean([d[3] for d in big])))
# kontrast: Or2 (lokalny) — czy tez rosnie z n?
o2 = defaultdict(list)
for r in rows:
    if r[3] == "Or2": o2[r[0]].append(r[6])
o2d = [(nof(i), st.mean(v)) for i, v in o2.items() if v]
if len(o2d) > 3:
    print("\nkontrast Or2 (lokalny): korelacja n vs reward r=%+.2f" % corr([x[0] for x in o2d], [x[1] for x in o2d]))
