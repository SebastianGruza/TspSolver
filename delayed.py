import sqlite3, statistics as st
from collections import defaultdict
c = sqlite3.connect("results/ladder_trials_v3.db")
rows = c.execute("SELECT instance,seed,step,candidate,burst,reward,best FROM ladder_trials").fetchall()
# grupy per (inst,seed,step)
G = defaultdict(list)
for r in rows: G[(r[0], r[1], r[2])].append(r)
# per bieg: sekwencja (step -> committed_op, winner_burst, gbest)
runs = defaultdict(dict)
for (i, s, stp), g in G.items():
    w = max(g, key=lambda r: r[5])           # zwyciezca kroku (max reward)
    gbest = min(r[6] for r in g)             # gbest-so-far w tym kroku
    runs[(i, s)][stp] = (w[3], w[4], gbest)

# dla kazdego kroku: immediate = burst zwyciezcy; downstream = spadek gbest w krokach [k+1..k+3]
merge_im, merge_dn, other_im, other_dn = [], [], [], []
for key, seq in runs.items():
    steps = sorted(seq)
    for idx, k in enumerate(steps):
        op, imb, gk = seq[k]
        # gbest po max 3 kroki dalej
        future = [seq[steps[j]][2] for j in range(idx+1, min(idx+4, len(steps)))]
        if not future: continue
        dn = gk - min(future)                # ile gbest spadl w oknie downstream (>=0)
        if op == "merge": merge_im.append(imb); merge_dn.append(dn)
        else: other_im.append(imb); other_dn.append(dn)

def show(name, im, dn):
    if not im: print("  %s: brak"%name); return
    mi, md = st.mean(im), st.mean(dn)
    ratio = md/mi if mi > 0 else float("inf")
    print("  %-8s n=%3d | immediate burst=%6.0f | downstream(3kr) gbest-drop=%7.0f | downstream/immediate=%.2f" % (
        name, len(im), mi, md, ratio))
print("EFEKT OPÓŹNIONY (gbest-drop w oknie 3 kroków PO commitcie):")
show("MERGE", merge_im, merge_dn)
show("inne",  other_im, other_dn)
# ile merge ma ~0 immediate ale >0 downstream (ukryty zysk)
hidden = sum(1 for a, b in zip(merge_im, merge_dn) if a < 50 and b > 200)
print("\nmerge z immediate<50 ale downstream>200 (UKRYTY zysk): %d/%d" % (hidden, len(merge_im)))
# to samo dla innych — kontrola
hidden_o = sum(1 for a, b in zip(other_im, other_dn) if a < 50 and b > 200)
print("inne  z immediate<50 ale downstream>200:               %d/%d" % (hidden_o, len(other_im)))
