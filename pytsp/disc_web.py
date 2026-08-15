"""Serwer WWW podglądu discovery — na żywo, najświeższe na górze.
   Odświeżanie w tle przez fetch()/Ajax (bez reloadu strony): '/' = szkielet+JS, '/data' = JSON.
   Postęp wzbogacony o opt + Aparapi (best/czas) z README.md + czas discovery z disc.log.
   Uruchom: .venv/bin/python pytsp/disc_web.py [port] [db] [disc.log]"""
import http.server, sqlite3, os, sys, re, time, json

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
DB = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/TspSolver/results/ladder_trials_v3.db")
LOG = sys.argv[3] if len(sys.argv) > 3 else "/tmp/disc2.log"
README = os.path.expanduser("~/TspSolver/README.md")

def load_ref():                                   # README: {inst: (opt, aparapi_best, aparapi_czas)} best run
    ref = {}
    try:
        for ln in open(README):
            f = ln.strip().split("|")
            if len(f) >= 7 and re.match(r"^[a-z]+[0-9]+$", f[1].strip()):
                try:
                    nm = f[1].strip(); ab = int(f[2]); opt = int(f[3]); tm = int(f[6])
                except ValueError:
                    continue
                if nm not in ref or ab < ref[nm][1]:
                    ref[nm] = (opt, ab, tm)
    except Exception:
        pass
    return ref

def disc_times():                                 # disc.log: {inst: sekundy_discovery}
    t = {}
    try:
        for ln in open(LOG):
            m = re.match(r"\s*(\w+) DISCOVERY.*\[(\d+)s\]", ln)
            if m: t[m.group(1)] = int(m.group(2))
    except Exception:
        pass
    return t

def build_data():                                 # -> dict serializowalny do JSON (cała logika gap/Δ tutaj)
    ref = load_ref(); dt = disc_times()
    try:
        c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=2)
        prog = c.execute("SELECT instance,COUNT(*),MAX(step),MIN(best),MAX(rowid) mr "
                         "FROM ladder_trials GROUP BY instance ORDER BY mr DESC").fetchall()
        rows = c.execute("SELECT rowid,instance,step,state_ops,state_K,candidate,burst,round(dt,2),round(reward,1) "
                         "FROM ladder_trials ORDER BY rowid DESC LIMIT 600").fetchall()
        c.close()
    except Exception as e:
        return {"err": str(e), "ts": time.strftime("%H:%M:%S")}
    P = []
    for i, (inst, cnt, mx, mn, mr) in enumerate(prog):
        opt, ab, atm = ref.get(inst, (None, None, None))
        gap = "%.3f" % ((mn / opt - 1) * 100) if opt else "—"
        apg = "%.3f" % ((ab / opt - 1) * 100) if (opt and ab) else "—"
        dc = "%d" % dt[inst] if inst in dt else "…"
        cls = "run" if i == 0 and inst not in dt else ""
        dcls = ""; delta = "—"
        if ab is not None:
            better_q = mn < ab; better_t = (inst in dt and atm and dt[inst] < atm)
            delta = ("%+.3fpp" % ((mn / opt - ab / opt) * 100)) if opt else ("%d" % (mn - ab))
            if better_q: delta += (" · −%d%%t" % round((1 - dt[inst] / atm) * 100)) if better_t else " ·+t"
            dcls = "win" if better_q else "los"
        P.append({"inst": inst, "best": mn, "gap": gap, "czas": dc,
                  "opt": opt or "—", "ap": ab or "—", "apg": apg, "aptm": atm or "—",
                  "delta": delta, "cls": cls, "dcls": dcls})
    groups = {}
    for rid, i2, st, ops, K, cand, burst, d, rew in rows:
        g = groups.setdefault((i2, st), {"mr": rid, "ops": ops, "K": K, "rows": []})
        if rid > g["mr"]: g["mr"] = rid
        g["rows"].append((cand, burst, d, rew))
    G = []
    for key in sorted(groups, key=lambda k: -groups[k]["mr"])[:16]:
        g = groups[key]; i2, st = key
        best = max(r[3] for r in g["rows"]) if g["rows"] else None
        rws = [{"cand": cand, "burst": burst, "dt": d, "rew": rew, "win": (rew == best)}
               for cand, burst, d, rew in sorted(g["rows"], key=lambda r: -r[3])]
        G.append({"inst": i2, "step": st, "ops": g["ops"], "K": g["K"], "rows": rws})
    return {"ts": time.strftime("%H:%M:%S"), "db": os.path.basename(DB), "prog": P, "gens": G}

CSS = """
body{background:#0e0b16;color:#e9e6f2;font:14px ui-monospace,Menlo,monospace;margin:0;padding:1.2rem}
h1{font-size:1.1rem;margin:0 0 .2rem;color:#a78bfa}.sub{color:#a99fc0;font-size:.8rem;margin:0 0 1rem}
table{border-collapse:collapse;margin:.3rem 0 1rem;font-size:.82rem}
th,td{padding:.25rem .6rem;text-align:right;white-space:nowrap}
th{color:#7a708f;text-transform:uppercase;font-size:.63rem;letter-spacing:.05em;border-bottom:1px solid #2a2440}
td.l,th.l{text-align:left}tr.run td{color:#fbbf24}
tr.win td.d{color:#4ade80;font-weight:bold}tr.los td.d{color:#f87171}
.gen{border:1px solid #2a2440;border-radius:8px;padding:.5rem .8rem;margin:.5rem .6rem .5rem 0;background:#171423;display:inline-block;vertical-align:top;min-width:330px}
.gh{color:#a78bfa;font-size:.8rem;margin-bottom:.2rem}.op{color:#38bdf8}
tr.wn td{color:#4ade80;font-weight:bold}
#dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.45rem;vertical-align:middle;background:#7a708f}
#dot.ok{background:#4ade80;box-shadow:0 0 7px #4ade80}#dot.err{background:#f87171;box-shadow:0 0 7px #f87171}
tr.flash td{animation:fl 1.3s ease-out}@keyframes fl{from{background:#14532d}to{background:transparent}}
"""

JS = r"""
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let prevBest={};
function progTable(P){
  let h="<table><tr><th class=l>instancja</th><th>best (disc)</th><th>gap%</th><th>czas</th>"
       +"<th>opt</th><th>Aparapi</th><th>Ap.gap%</th><th>Ap.czas</th><th class=l>Δ</th></tr>";
  for(const r of P){
    const fl=(prevBest[r.inst]!==undefined && r.best<prevBest[r.inst])?" flash":"";
    prevBest[r.inst]=r.best;
    h+=`<tr class="${r.cls} ${r.dcls}${fl}"><td class=l>${esc(r.inst)}</td><td>${r.best}</td>`
      +`<td>${r.gap}</td><td>${r.czas}</td><td>${r.opt}</td><td>${r.ap}</td>`
      +`<td>${r.apg}</td><td>${r.aptm}</td><td class=d>${esc(r.delta)}</td></tr>`;
  }
  return h+"</table>";
}
function gensHtml(G){
  let h="";
  for(const g of G){
    h+=`<div class=gen><div class=gh>${esc(g.inst)} · step ${g.step} · <span class=op>ops=${g.ops} K=${g.K}</span></div>`
      +"<table><tr><th class=l>ruch</th><th>burst</th><th>dt</th><th>reward</th></tr>";
    for(const r of g.rows)
      h+=`<tr${r.win?" class=wn":""}><td class=l>${esc(r.cand)}</td><td>${r.burst}</td><td>${r.dt.toFixed(2)}</td><td>${r.rew.toFixed(1)}</td></tr>`;
    h+="</table></div>";
  }
  return h;
}
async function tick(){
  try{
    const res=await fetch('/data',{cache:'no-store'});
    const d=await res.json();
    if(d.err){$('#dot').className='err';$('#ts').textContent=d.ts+' · DB: '+d.err+' (za chwilę)';return;}
    $('#prog').innerHTML=progTable(d.prog);
    $('#gens').innerHTML=gensHtml(d.gens);
    $('#ts').textContent='DB '+d.db+' · '+d.ts+' · na żywo';
    $('#dot').className='ok';
  }catch(e){$('#dot').className='err';$('#ts').textContent='brak połączenia — wznawiam…';}
}
let timer=null;
function start(){if(!timer){tick();timer=setInterval(tick,3000);}}
function stop(){clearInterval(timer);timer=null;}
document.addEventListener('visibilitychange',()=>document.hidden?stop():start());
start();
"""

PAGE = ("<!doctype html><meta charset=utf-8><title>Discovery live</title><style>%s</style>"
        "<h1>Discovery — na żywo</h1>"
        "<p class=sub><span id=dot></span><span id=ts>łączenie…</span></p>"
        "<div id=prog></div>"
        "<h1 style='font-size:.9rem'>Ostatnie generacje (wg reward=burst/s)</h1>"
        "<div id=gens></div>"
        "<script>%s</script>") % (CSS, JS)

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(build_data(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass

http.server.ThreadingHTTPServer.allow_reuse_address = True
with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H) as srv:
    print("serwer na :%d DB=%s LOG=%s" % (PORT, DB, LOG), flush=True)
    srv.serve_forever()
