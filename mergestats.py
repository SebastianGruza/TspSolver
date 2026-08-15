import sqlite3, statistics as st
from collections import Counter, defaultdict
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT instance,seed,step,candidate,burst,dt,reward,state_ops,state_K FROM ladder_trials").fetchall()
step_groups = defaultdict(list)
for r in rows: step_groups[(r[0], r[1], r[2])].append(r)

# 1. rozklad rewardu merge — czy bimodalny (rzadkie duze spiki)
mg = sorted([r[6] for r in rows if r[3] == "merge"])
mb = sorted([r[4] for r in rows if r[3] == "merge"])
print("MERGE reward: n=%d  min=%.0f  mediana=%.0f  sr=%.0f  p90=%.0f  max=%.0f" % (
    len(mg), mg[0], st.median(mg), st.mean(mg), mg[int(0.9*len(mg))], mg[-1]))
zero = sum(1 for x in mb if x <= 0); small = sum(1 for x in mb if 0 < x < 100)
big = sum(1 for x in mb if x >= 500)
print("  burst: <=0: %d (%.0f%%) | 1-99: %d | >=500: %d (%.0f%%)  <- ksztalt" % (
    zero, 100*zero/len(mb), small, big, 100*big/len(mb)))
print("  dla porownania Or2 reward: mediana=%.0f max=%.0f (staly wysoki)" % (
    st.median([r[6] for r in rows if r[3]=="Or2"]), max(r[6] for r in rows if r[3]=="Or2")))

# 2. KIEDY merge wygrywa — step (faza) i ops (ile operatorow juz dodane)
print("\nkiedy merge WYGRYWA (max-reward w kroku):")
wsteps = []; wops = []
for (i,s,stp),g in step_groups.items():
    w = max(g, key=lambda r: r[6])
    if w[3] == "merge": wsteps.append(stp); wops.append(bin(w[7]).count("1"))
if wsteps:
    print("  %d wygranych | step: min=%d mediana=%d max=%d | operatorow-w-ops: mediana=%d" % (
        len(wsteps), min(wsteps), int(st.median(wsteps)), max(wsteps), int(st.median(wops))))
    print("  -> merge wygrywa PÓŹNO?" if st.median(wsteps) > 4 else "  -> merge wygrywa rownomiernie")

# 3. merge spiki — konkretne przypadki (top burst)
print("\nnajwieksze spiki merge (instancja/step/burst/reward):")
mspikes = sorted([(r[0],r[2],r[4],r[6],r[8]) for r in rows if r[3]=="merge"], key=lambda x:-x[2])[:6]
for i,stp,b,rw,K in mspikes: print("  %-9s step%2d  burst=%6d reward=%6.0f K=%d" % (i,stp,b,rw,K))

# 4. ile instancji w OGÓLE uzylo merge jako zwyciezcy
minst = set(i for (i,s,stp),g in step_groups.items() if max(g,key=lambda r:r[6])[3]=="merge")
print("\nmerge byl zwyciezca w %d/%d instancji: %s" % (
    len(minst), len(set(r[0] for r in rows)), ", ".join(sorted(minst))))
