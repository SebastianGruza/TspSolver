"""Live view of Santa runs: parses ~/santa/results/*.log (round/lk/relax/full/ils lines with a score) and LKH logs,
serves a convergence chart + table. Run: python domains/santa/web.py [port=8902]"""
import http.server, json, os, re, sys, time, glob, sqlite3, subprocess
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8902
RES = Path.home() / "santa" / "results"
LKH = Path.home() / "santa" / "lkh_real"
REF = {"Kaggle top": 1512000, "doctor's LKH benchmark": 1516256, "doctor (9 days)": 1529000}
RX = re.compile(r"^(?:relax |full )?(round|lk|ils|frag|xswap2|xswap|dp)\b[^:]*:.*?(?:score|best)=([\d.]+).*?t=(\d+)s")
MAX_AGE = 24 * 3600          # show only runs from the last day (+ live ones)


def parse_run(path):
    pts = []; t_off = 0; last_t = 0; start = ""
    for line in open(path, errors="ignore"):
        m = RX.match(line)
        if not m:
            if line.startswith("start="):
                s = re.search(r"score=([\d.]+)", line)
                if s: pts.append([0, float(s.group(1)), "start"])
                st = re.match(r"start=(\S+)", line)
                if st: start = Path(st.group(1)).name.replace("sub_", "").replace(".csv", "")
            continue
        ph, sc, t = m.group(1), float(m.group(2)), int(m.group(3))
        if t < last_t:                      # new phase -> time from zero, we shift
            t_off += last_t
        last_t = t
        pts.append([t_off + t, sc, ph])
    done = [l for l in open(path, errors="ignore") if l.startswith("DONE")]
    alive = os.path.getmtime(path) > time.time() - 120 or Path(path).stem in _running()
    return {"name": Path(path).stem, "points": pts, "done": done[-1].strip()[:160] if done else "", "alive": alive,
            "best": min((p[1] for p in pts), default=None), "start": start, "mtime": os.path.getmtime(path),
            "age_min": int((time.time() - os.path.getmtime(path)) / 60)}


_RUN_CACHE = {"t": 0.0, "tags": set()}


def _running():
    """tags of the run.py runs currently active (ILS logs once per iteration - the log mtime is not enough)."""
    if time.time() - _RUN_CACHE["t"] > 15:
        try:
            out = subprocess.run(["pgrep", "-af", "santa/run.py"], capture_output=True, text=True).stdout
            _RUN_CACHE["tags"] = set(re.findall(r"--tag (\S+)", out))
        except Exception:
            pass
        _RUN_CACHE["t"] = time.time()
    return _RUN_CACHE["tags"]


def parse_lkh():
    out = []
    for f in sorted(glob.glob(str(LKH / "lkh_s*.log"))):
        pts = []
        for line in open(f, errors="ignore"):
            m = re.match(r"^\* (\d+): Cost = (\d+), Time = ([\d.]+) sec", line)
            if m:
                pts.append([float(m.group(3)), int(m.group(2)) / 1000.0])
        out.append({"name": Path(f).stem + " (plain, no penalty)", "points": pts})
    return out


JOURNAL = Path.home() / "mf-santa" / "docs" / "santa" / "JOURNAL.md"


def md_to_html(md: str) -> str:
    """Minimal markdown: #/##/###, **bold**, `code`, lists '- ', '---', paragraphs."""
    import html as H
    out = []; para = []; inlist = False
    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>"); para = []
    def inline(t):
        t = H.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', t)
        return t
    for line in md.splitlines():
        if line.startswith("- "):
            flush()
            if not inlist: out.append("<ul>"); inlist = True
            out.append("<li>" + inline(line[2:]) + "</li>"); continue
        if inlist: out.append("</ul>"); inlist = False
        if line.startswith("### "): flush(); out.append("<h3>" + inline(line[4:]) + "</h3>")
        elif line.startswith("## "): flush(); out.append("<h2>" + inline(line[3:]) + "</h2>")
        elif line.startswith("# "): flush(); out.append("<h1>" + inline(line[2:]) + "</h1>")
        elif line.strip() == "---": flush(); out.append("<hr>")
        elif not line.strip(): flush()
        else: para.append(line)
    flush()
    if inlist: out.append("</ul>")
    return "\n".join(out)


JOURNAL_CSS = """<style>body{font:16px/1.6 system-ui;max-width:78ch;margin:32px auto;padding:0 16px;color:#1a2029;background:#f5f6f8}
h1{font-size:1.7rem;line-height:1.2}h2{margin-top:2em;border-top:2px solid #1a2029;padding-top:.6em}p{margin:.6em 0}
code{background:#e3eff3;padding:1px 4px;border-radius:3px}hr{border:0;border-top:1px solid #d9dee4;margin:1.5em 0}
a{color:#1f6f8b}.nav{font-size:13px;color:#5c6570;margin-bottom:16px}</style>"""


def state():
    now = time.time()
    paths = [p for p in glob.glob(str(RES / "*.log")) if os.path.getsize(p) > 0 and "synth" not in p
             and os.path.getmtime(p) > now - MAX_AGE]
    runs = [parse_run(p) for p in paths]
    runs = [r for r in runs if r["points"]]
    runs.sort(key=lambda r: (not r["alive"], -r["mtime"]))
    runs = runs[:14]
    return {"ts": time.strftime("%H:%M:%S"), "runs": runs, "lkh": parse_lkh(), "ref": REF}


HTML = """<!doctype html><meta charset=utf-8><title>Santa live</title>
<style>body{font:13px system-ui;margin:16px;background:#f5f6f8;color:#1a2029}canvas{background:#fff;border:1px solid #d9dee4}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:3px 10px;border-bottom:1px solid #d9dee4;text-align:left}
.alive{color:#1f6f8b;font-weight:600}.dead{color:#98a2ad}</style>
<h2>Traveling Santa 2018 · live view <span id=ts style="font-weight:400;color:#5c6570"></span> · <a href="/journal" style="font-size:14px;font-weight:400">journal</a></h2>
<canvas id=c width=1100 height=460></canvas>
<table id=t></table>
<script>
const CL=['#1f6f8b','#b8741a','#4a8f3f','#8b1f4a','#555','#1f4a8b','#8b6a1f'];
async function tick(){const d=await (await fetch('/api/state')).json();document.getElementById('ts').textContent='· '+d.ts;
const c=document.getElementById('c'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
let ys=[],xs=[];d.runs.forEach(r=>r.points.forEach(p=>{ys.push(p[1]);xs.push(p[0])}));d.lkh.forEach(r=>r.points.forEach(p=>{ys.push(p[1]);xs.push(p[0])}));
Object.values(d.ref).forEach(v=>ys.push(v));if(!ys.length)return;
let ymin=Math.min(...ys)*0.998,ymax=Math.min(Math.max(...ys),1.66e6)*1.002,xmax=Math.max(...xs,60);
const L=70,B=30,W=c.width-L-20,H=c.height-B-10;const X=x=>L+W*x/xmax,Y=y=>10+H*(1-(y-ymin)/(ymax-ymin));
g.strokeStyle='#d9dee4';g.fillStyle='#5c6570';g.font='11px monospace';
for(let i=0;i<=6;i++){const y=ymin+(ymax-ymin)*i/6;g.beginPath();g.moveTo(L,Y(y));g.lineTo(L+W,Y(y));g.stroke();g.fillText((y/1e6).toFixed(4)+'M',2,Y(y)+4)}
for(let i=0;i<=8;i++){const x=xmax*i/8;g.fillText((x/60).toFixed(0)+'min',X(x)-12,c.height-8)}
Object.entries(d.ref).forEach(([k,v],i)=>{g.strokeStyle='#c0392b';g.setLineDash([4,3]);g.beginPath();g.moveTo(L,Y(v));g.lineTo(L+W,Y(v));g.stroke();g.setLineDash([]);g.fillStyle='#c0392b';g.fillText(k+' '+v,L+4,Y(v)-3)});
let rows='<tr><th>run</th><th>start</th><th>best</th><th>last</th><th>time (phase sum)</th><th>last entry</th><th>status</th></tr>';
[...d.runs,...d.lkh.map(r=>({...r,lkh:1}))].forEach((r,i)=>{const col=CL[i%CL.length];g.strokeStyle=col;g.lineWidth=r.lkh?2:1.5;g.beginPath();
r.points.forEach((p,j)=>{const y=Math.min(p[1],ymax);j?g.lineTo(X(p[0]),Y(y)):g.moveTo(X(p[0]),Y(y))});g.stroke();
const last=r.points[r.points.length-1];g.fillStyle=col;g.fillText(r.name,X(last[0])+3,Y(Math.min(last[1],ymax)));
const best=r.lkh?Math.min(...r.points.map(p=>p[1])):r.best;
rows+=`<tr style="color:${col}"><td>${r.name}</td><td>${r.start||''}</td><td>${best?best.toFixed(0):''}</td><td>${last[1].toFixed(0)}</td><td>${(last[0]/60).toFixed(0)} min</td><td>${r.age_min!=null?r.age_min+' min ago':''}</td><td class=${r.alive||r.lkh?'alive':'dead'}>${r.done?r.done:(r.alive||r.lkh?'running':'stopped')}</td></tr>`});
document.getElementById('t').innerHTML=rows}
tick();setInterval(tick,10000);
</script>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api/state"):
            b = json.dumps(state()).encode(); ct = "application/json"
        elif self.path.startswith("/journal"):
            md = JOURNAL.read_text(encoding="utf-8") if JOURNAL.exists() else "# (no journal)"
            b = ('<!doctype html><meta charset=utf-8><title>Santa · journal</title>' + JOURNAL_CSS +
                 '<div class=nav><a href="/">← dashboard</a> · experiment journal, entries every ~6 h · '
                 + time.strftime("%Y-%m-%d %H:%M") + '</div>' + md_to_html(md)).encode("utf-8"); ct = "text/html; charset=utf-8"
        else:
            b = HTML.encode(); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
