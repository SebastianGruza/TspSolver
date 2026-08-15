"""Pomiar amortized-planning gap: czy wybor bandyty (argmax reward) jest przewidywalny z cech phi(s)?
   Leave-One-Instance-Out CV. Metryki: top-1 acc (z/bez stay) vs baseline, regret rewardu, recall/operator.
   Odpowiada na obawe: ile przewagi bandyty to wiedza OGOLNA (amortyzowalna) vs lookahead specyficzny dla stanu."""
import sqlite3, os, sys, math
from collections import defaultdict, Counter
import numpy as np
db = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/TspSolver/results/ladder_trials_v3.db")
FEATS = ["n","feat_nn","feat_cv","feat_skew","feat_kurt","feat_far","feat_clark","feat_gridcv",
         "feat_aspect","feat_tight","state_ops","state_K","state_ox","phase_uniq","phase_cv",
         "phase_gbstall","phase_gbstall_max"]
c = sqlite3.connect(db)
sel = "instance,seed,step,candidate,reward," + ",".join(FEATS)
rows = c.execute("SELECT %s FROM ladder_trials" % sel).fetchall()
# grupuj w decyzje (instance,seed,step): cechy + {cand:reward} + zwyciezca=argmax reward
dec = {}
for r in rows:
    inst, seed, step, cand, rew = r[0], r[1], r[2], r[3], r[4]
    feat = np.array(r[5:], float)
    feat[0] = math.log10(feat[0])                    # n -> log10
    key = (inst, seed, step)
    d = dec.setdefault(key, {"inst": inst, "feat": feat, "rw": {}})
    d["rw"][cand] = rew
for d in dec.values():
    d["win"] = max(d["rw"], key=d["rw"].get)
D = list(dec.values())
insts = sorted(set(d["inst"] for d in D))
print("decyzji: %d | instancji: %d | klas(operatorow): %d" % (len(D), len(insts), len(set(d["win"] for d in D))))
print("rozklad zwyciezcow:", dict(Counter(d["win"] for d in D).most_common()))

# model: sklearn RF jesli jest, inaczej kNN w numpy (standaryzowane cechy)
try:
    from sklearn.ensemble import RandomForestClassifier
    HAVE_RF = True
except Exception:
    HAVE_RF = False
print("model:", "RandomForest" if HAVE_RF else "kNN(numpy)")

def predict_loio():
    preds = [None]*len(D)
    X = np.array([d["feat"] for d in D]); y = np.array([d["win"] for d in D])
    inst_of = np.array([d["inst"] for d in D])
    for ho in insts:
        tr = inst_of != ho; te = ~tr
        if HAVE_RF:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
            m.fit(X[tr], y[tr])
            pr = m.predict(X[te])
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
            Xtr, Xte = (X[tr]-mu)/sd, (X[te]-mu)/sd
            ytr = y[tr]; pr = []
            for xq in Xte:
                dist = ((Xtr-xq)**2).sum(1); idx = np.argsort(dist)[:5]
                pr.append(Counter(ytr[idx]).most_common(1)[0][0])
            pr = np.array(pr)
        for j, i in enumerate(np.where(te)[0]): preds[i] = pr[j]
    return preds

pred = predict_loio()
maj = Counter(d["win"] for d in D).most_common(1)[0][0]   # baseline: zawsze najczestszy
# --- metryki ---
def acc(sub): 
    return np.mean([pred[i] == D[i]["win"] for i in sub]) if sub else float("nan")
all_i = list(range(len(D)))
nonstay_i = [i for i in all_i if D[i]["win"] != "stay"]
maj_acc = np.mean([D[i]["win"] == maj for i in all_i])
maj_acc_ns = np.mean([D[i]["win"] == maj for i in nonstay_i])   # majority i tak przewiduje maj (nie-stay -> 0 jesli maj=stay)
print("\n=== TOP-1 ACCURACY (Leave-One-Instance-Out) ===")
print("  wszystkie decyzje:   model=%.1f%%   baseline(zawsze '%s')=%.1f%%" % (100*acc(all_i), maj, 100*maj_acc))
print("  bez 'stay' (realny wybor operatora): model=%.1f%%   baseline=%.1f%%" % (100*acc(nonstay_i), 100*maj_acc_ns))
# --- regret rewardu: ile tracimy idac za predykcja zamiast argmax ---
def regret(sub):
    fr = []; cov = 0
    for i in sub:
        d = D[i]; best = d["rw"][d["win"]]
        p = pred[i]
        if p in d["rw"]:
            cov += 1
            if best > 0: fr.append((best - d["rw"][p]) / best)
    return (np.mean(fr) if fr else float("nan")), cov/len(sub)
r_all, cov_all = regret(all_i); r_ns, cov_ns = regret(nonstay_i)
# regret baseline (zawsze maj) i losowego dostepnego
def regret_pol(sub, polf):
    fr=[]
    for i in sub:
        d=D[i]; best=d["rw"][d["win"]]; p=polf(d)
        if p in d["rw"] and best>0: fr.append((best-d["rw"][p])/best)
    return np.mean(fr) if fr else float("nan")
r_maj = regret_pol(nonstay_i, lambda d: maj)
print("\n=== REGRET REWARDU (frakcja utraconego reward/decyzja; 0=idealne) ===")
print("  model  wszystkie: %.1f%% (coverage %.0f%%) | bez stay: %.1f%% (cov %.0f%%)" % (100*r_all, 100*cov_all, 100*r_ns, 100*cov_ns))
print("  baseline(maj) bez stay: %.1f%%" % (100*r_maj))
# --- recall per-operator: czy merge mniej przewidywalny ---
print("\n=== RECALL PER-OPERATOR (jak czesto model odzyskuje danego zwyciezce) ===")
byop = defaultdict(lambda: [0,0])
for i in all_i:
    w = D[i]["win"]; byop[w][1]+=1; byop[w][0]+= (pred[i]==w)
for op,(hit,tot) in sorted(byop.items(), key=lambda kv:-kv[1][1]):
    if tot>=6: print("  %-9s recall=%3.0f%%  (n=%d)" % (op, 100*hit/tot, tot))
