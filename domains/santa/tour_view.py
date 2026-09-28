"""Static page with the whole tour (pan/zoom on a canvas): python tour_view.py TOUR.csv OUT.html
Data: coordinates in tour order (Float32) + step flags (bit0 = penalty step, bit1 = prime city at a penalty
position) + ratio of edge length to the 1-NN distance (Float16->Float32) for highlighting long edges."""
import base64, sys, numpy as np
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D
from domains.santa.baseline import knn

tour, out = sys.argv[1], sys.argv[2]
data = sys.argv[3] if len(sys.argv) > 3 else None                 # another instance (TSPLIB->csv): no penalty
xy = D.load_cities(data) if data else D.load_cities(); prime = D.primes_upto(len(xy) - 1)
if data: prime[:] = 1
P = D.read_tour(tour); n = len(xy)
sc = D.score(P, xy, prime); plain = D.plain_len(P, xy)
nn = knn(xy, 1); nn1 = np.sqrt(((xy[nn[:, 0]] - xy) ** 2).sum(1))
C = xy[P].astype(np.float32)                                   # n+1 points in tour order
d = np.sqrt(((C[1:] - C[:-1]) ** 2).sum(1)); ratio = (d / np.maximum(nn1[P[:-1]], 1e-9)).astype(np.float32)
k = np.arange(1, n + 1); pen = (k % 10 == 0)
flags = (pen & (prime[P[:-1]] == 0)).astype(np.uint8) | ((pen & (prime[P[:-1]] == 1)).astype(np.uint8) << 1)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
html = f"""<!doctype html><meta charset=utf-8><title>Santa tour view</title>
<style>body{{margin:0;font:13px system-ui;background:#111;color:#ddd}}#bar{{position:fixed;top:0;left:0;right:0;padding:6px 10px;background:#222c;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
canvas{{display:block}}label{{white-space:nowrap}}input[type=range]{{width:120px}}</style>
<div id=bar><b>{tour.split('/')[-1]}</b> score {sc:,.0f} · plain {plain:,.0f} · penalty {sc-plain:,.0f} · n={n}
<label><input type=checkbox id=hue checked> colour = position in tour</label>
<label><input type=checkbox id=penc checked> penalty steps (red)</label>
<label><input type=checkbox id=longc checked> long edges ≥ <span id=thv>6</span>× 1-NN <input type=range id=th min=2 max=20 value=6></label>
<span id=info></span> · wheel = zoom, drag = pan, double-click = reset</div>
<canvas id=c></canvas>
<script>
const dec=s=>{{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer}};
const C=new Float32Array(dec("{b64(C)}")),R=new Float32Array(dec("{b64(ratio)}")),F=new Uint8Array(dec("{b64(flags)}"));
const N=C.length/2;let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;for(let i=0;i<N;i++){{const x=C[2*i],y=C[2*i+1];if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y}}
const c=document.getElementById('c'),g=c.getContext('2d');let W,H,scale,ox,oy;
function reset(){{W=c.width=innerWidth;H=c.height=innerHeight-40;scale=Math.min(W/(maxx-minx),H/(maxy-miny))*0.97;ox=(W-(maxx-minx)*scale)/2-minx*scale;oy=(H-(maxy-miny)*scale)/2-miny*scale;draw()}}
const X=x=>x*scale+ox,Y=y=>H-(y*scale+oy);
function draw(){{g.fillStyle='#111';g.fillRect(0,0,W,H);const hue=document.getElementById('hue').checked,pen=document.getElementById('penc').checked,lon=document.getElementById('longc').checked,th=+document.getElementById('th').value;
const vx0=(-ox)/scale,vx1=(W-ox)/scale,vy0=(-oy)/scale,vy1=(H-oy)/scale;let vis=0,nseg=hue?64:1;g.lineWidth=scale>3?1.2:0.6;
for(let s=0;s<nseg;s++){{g.strokeStyle=hue?`hsl(${{s*360/nseg}},80%,60%)`:'#8ab';g.beginPath();const a=Math.floor(s*N/nseg),b=Math.min(N,Math.floor((s+1)*N/nseg)+1);let pd=false;
for(let i=a;i<b;i++){{const x=C[2*i],y=C[2*i+1];if(x<vx0-1||x>vx1+1||y<vy0-1||y>vy1+1){{pd=false;continue}}if(!pd){{g.moveTo(X(x),Y(y));pd=true}}else g.lineTo(X(x),Y(y));vis++}}g.stroke()}}
if(lon){{g.strokeStyle='#ff0';g.lineWidth=2;g.beginPath();for(let i=0;i<N-1;i++){{if(R[i]<th)continue;const x=C[2*i],y=C[2*i+1];if(x<vx0||x>vx1||y<vy0||y>vy1)continue;g.moveTo(X(x),Y(y));g.lineTo(X(C[2*i+2]),Y(C[2*i+3]))}}g.stroke()}}
if(pen&&scale>0.5){{g.fillStyle='#f33';for(let i=0;i<N-1;i++){{if(!(F[i]&1))continue;const x=C[2*i],y=C[2*i+1];if(x<vx0||x>vx1||y<vy0||y>vy1)continue;g.fillRect(X(x)-1.5,Y(y)-1.5,3,3)}}
g.fillStyle='#3f3';for(let i=0;i<N-1;i++){{if(!(F[i]&2))continue;const x=C[2*i],y=C[2*i+1];if(x<vx0||x>vx1||y<vy0||y>vy1)continue;g.fillRect(X(x)-1.5,Y(y)-1.5,3,3)}}}}
let nl=0;for(let i=0;i<N-1;i++)if(R[i]>=th)nl++;document.getElementById('info').textContent=`visible pts: ${{vis}} · long edges ≥${{th}}×: ${{nl}} · zoom ${{scale.toFixed(2)}}`}}
let drag=null;c.onmousedown=e=>drag=[e.clientX,e.clientY,ox,oy];c.onmousemove=e=>{{if(!drag)return;ox=drag[2]+e.clientX-drag[0];oy=drag[3]-(e.clientY-drag[1]);draw()}};c.onmouseup=c.onmouseleave=()=>drag=null;
c.onwheel=e=>{{e.preventDefault();const f=e.deltaY<0?1.25:0.8;const mx=e.clientX,my=H-(e.clientY-40);ox=mx-(mx-ox)*f;oy=my-(my-oy)*f;scale*=f;draw()}};
c.ondblclick=reset;document.getElementById('th').oninput=e=>{{document.getElementById('thv').textContent=e.target.value;draw()}};
['hue','penc','longc'].forEach(id=>document.getElementById(id).onchange=draw);addEventListener('resize',reset);reset();
</script>"""
open(out, "w").write(html); print("OK", out, len(html) // 1024, "KB", "score", round(sc, 1))
