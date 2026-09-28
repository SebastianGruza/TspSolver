"""Every N minutes: find the best tour (sub_*.csv, gpx*.csv, best_*.csv from the last 24 h) and render www/best.html (+ best.json)."""
import glob, json, os, re, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D
RES = os.path.expanduser("~/santa/results"); WWW = os.path.expanduser("~/santa/www")
PY = os.path.expanduser("~/TspSolver/.venv/bin/python")
xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1)
last = None
while True:
    try:
        cands = []
        big = lambda f: os.path.getsize(f) > 1_000_000 and "synth" not in f and "t20k" not in f and "lkh" not in f.lower()   # full instance, not synthetic
        for f in glob.glob(f"{RES}/sub_*.csv"):
            m = re.search(r"_(\d+)\.csv$", f)
            if m and big(f): cands.append((float(m.group(1)), f))
        best_named = (1e18, None)
        for sc_, f in sorted(cands):                                   # file-name score must match the real score
            try:
                if abs(D.score(D.read_tour(f), xy, prime) - sc_) < 1000: best_named = (sc_, f); break
            except Exception:
                pass
        others = [f for f in glob.glob(f"{RES}/gpx*.csv") + glob.glob(f"{RES}/best_*.csv") if big(f) and os.path.getmtime(f) > time.time() - 86400]
        best = best_named
        for f in others:
            try:
                s = D.score(D.read_tour(f), xy, prime)
                if s < best[0]: best = (s, f)
            except Exception:
                pass
        if best[1] and best[1] != last:
            s = D.score(D.read_tour(best[1]), xy, prime)
            subprocess.run([PY, "/home/seba/mf-santa/domains/santa/tour_view.py", best[1], f"{WWW}/best.tmp.html"], check=True)
            os.replace(f"{WWW}/best.tmp.html", f"{WWW}/best.html")
            json.dump({"file": os.path.basename(best[1]), "score": s, "ts": time.strftime("%Y-%m-%d %H:%M")}, open(f"{WWW}/best.json", "w"))
            print(time.strftime("%H:%M"), "best", best[1], round(s, 1), flush=True); last = best[1]
    except Exception as e:
        print("ERR", e, flush=True)
    time.sleep(600)
