"""Agregacja discovery: statystycznie najlepszy kandydat per stan (ops,K-bucket).
   -> podpowiada stałą kolejność ruchów do zabetonowania w LADDER_MOVES."""
import sqlite3, collections
from statistics import median
con = sqlite3.connect("results/ladder_trials.db")
rows = con.execute("SELECT state_ops, state_K, candidate, reward, burst FROM ladder_trials").fetchall()
print("wierszy:", len(rows), " instancje:",
      [r[0] for r in con.execute("SELECT DISTINCT instance FROM ladder_trials")])
agg = collections.defaultdict(list)
for ops, K, cand, rew, burst in rows:
    kb = "lo" if K < 16 else ("mid" if K < 26 else "hi")     # kubełek K
    agg[(ops, kb, cand)].append(rew)
print("\nstate_ops  K   candidate    n   med_reward")
for (ops, kb, cand), rs in sorted(agg.items(), key=lambda x: (x[0][0], -median(x[1]))):
    print("%8d  %3s  %-10s  %3d  %8.1f" % (ops, kb, cand, len(rs), median(rs)))
# greedy path po operatorach (ignoruje K/merge/ox -> pokazuje kolejność operatorów)
print("\n--- greedy operator order (max med_reward per stan ops) ---")
ops = 0; order = []
opbits = {"3opt": 16, "Or2": 4, "Or3": 8, "LE": 32}
for _ in range(4):
    cands = [(median(rs), c) for (o, kb, c), rs in agg.items() if o == ops and c in opbits]
    if not cands:
        break
    _, best = max(cands); order.append(best); ops |= opbits[best]
    print("ops=%2d -> %s" % (ops, best))
print("kolejność operatorów:", " -> ".join(order))
