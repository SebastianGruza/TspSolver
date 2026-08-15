"""Kanoniczna 'waznosc' instancji (inverse-density, kNN na log n) + wazone krzywe preferencji operatorow.
   Odbiasowuje nadreprezentacje malych instancji -> wazona probka ~ jednostajna w log(n).
   Uzycie: .venv/bin/python pytsp/opweights.py [db]"""
import sqlite3, re, math, sys, os
from collections import defaultdict
import numpy as np

INSTS = ("fl417 gr431 pr439 pcb442 d493 att532 ali535 u574 rat575 p654 d657 gr666 u724 rat783 "
         "pr1002 u1060 vm1084 pcb1173 d1291 rl1304 rl1323 nrw1379 fl1400 u1432 fl1577 d1655 vm1748 "
         "u1817 rl1889 d2103 u2152 u2319 pr2392 pcb3038 fl3795 fnl4461").split()
nof = lambda x: int(re.search(r"(\d+)", x).group(1))

def importance_weights(insts=INSTS, k=3, clip=(0.3, 4.0)):
    """kNN density na log10(n): waga ~ promien k-tego sasiada, clip, norm Sigma=m."""
    x = np.log10(np.array([nof(i) for i in insts], float)); m = len(x)
    rk = np.array([np.sort(np.abs(x - xi))[k] for xi in x])   # dyst. do k-tego sasiada
    w = rk / rk.mean()
    w = np.clip(w, *clip)
    w = w * m / w.sum()
    return {insts[i]: float(w[i]) for i in range(m)}

def ess(wmap):
    w = np.array(list(wmap.values())); return (w.sum()**2) / (w**2).sum()

def shares(db):
    c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    rows = c.execute("SELECT instance,candidate,reward FROM ladder_trials").fetchall(); c.close()
    r_io = defaultdict(list); done = set()
    for inst, cand, rew in rows: r_io[(inst, cand)].append(rew); done.add(inst)
    ops = sorted(set(cd for (_, cd) in r_io))
    sh = {}
    for inst in done:
        tot = sum(np.mean(r_io[(inst, o)]) for o in ops if (inst, o) in r_io)
        for o in ops:
            if (inst, o) in r_io and tot > 0: sh[(inst, o)] = np.mean(r_io[(inst, o)]) / tot
    return sh, ops, sorted(done, key=nof)

def wls(xs, ys, ws):
    b, a = np.polyfit(xs, ys, 1, w=np.sqrt(ws)); return a, b

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/TspSolver/results/ladder_trials_v3.db")
    W = importance_weights()
    print("=== WAZNOSC instancji (kNN k=3, log n), ESS=%.1f/%d ===" % (ess(W), len(W)))
    top = sorted(W, key=lambda i: -W[i])[:5]; bot = sorted(W, key=lambda i: W[i])[:5]
    print("  promowane: " + ", ".join("%s=%.2f" % (i, W[i]) for i in top))
    print("  dlawione:  " + ", ".join("%s=%.2f" % (i, W[i]) for i in bot))
    sh, ops, done = shares(db)
    print("\n=== WAZONE krzywe preferencji (udzial ~ log n, waga=waznosc) ===")
    print("operator   | b_wls  | kierunek | udz@500 | @1500 | @3500")
    fits = {}
    for o in ops:
        xs = np.array([math.log10(nof(i)) for i in done if (i, o) in sh])
        ys = np.array([sh[(i, o)] for i in done if (i, o) in sh])
        ws = np.array([W[i] for i in done if (i, o) in sh])
        if len(xs) < 5: continue
        a, b = wls(xs, ys, ws); fits[o] = (a, b)
        pv = lambda n: max(0.0, a + b * math.log10(n))
        d = "↑" if b > 0.005 else ("↓" if b < -0.005 else "≈")
        print("  %-9s| %+.4f |    %s     |  %.3f  | %.3f | %.3f" % (o, b, d, pv(500), pv(1500), pv(3500)))
    print("\n=== WAZONA preferencja operatorow (ranking wg predykowanego udzialu) ===")
    for n in (500, 1000, 2000, 3500):
        pref = sorted(fits, key=lambda o: -(fits[o][0] + fits[o][1] * math.log10(n)))
        pref = [o for o in pref if o != "stay"][:7]
        print("  n=%4d: %s" % (n, "  >  ".join(pref)))
    print("\n(ukonczonych instancji w fit: %d, zakres n=%d..%d)" % (len(done), nof(done[0]), nof(done[-1])))
