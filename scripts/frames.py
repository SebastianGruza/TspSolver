"""Animation frames "best tour over time" (PNG 1920x1080, PIL): --history renders one frame per from-scratch record
improvement (sub_*.csv by modification time, excluding synth/t20k/lkh), --loop renders the current record hourly (best.json).
Wideo: ffmpeg -framerate 2 -pattern_type glob -i '~/santa/frames/*.png' -c:v libx264 -pix_fmt yuv420p santa.mp4"""
import glob, json, os, re, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/seba/mf-santa")
from domains.santa import data as D
RES = os.path.expanduser("~/santa/results"); FR = os.path.expanduser("~/santa/frames"); os.makedirs(FR, exist_ok=True)
W, H = 1920, 1080; PAD = 40
xy = D.load_cities(); prime = D.primes_upto(len(xy) - 1)
mn = xy.min(0); ext = xy.max(0) - mn; sc = min((W - 2 * PAD) / ext[0], (H - 2 * PAD - 60) / ext[1])
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34); FONT2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except Exception: FONT = FONT2 = ImageFont.load_default()
import colorsys


def render(tour_file, out, label, idx=None):
    P = D.read_tour(tour_file); s = D.score(P, xy, prime); plain = D.plain_len(P, xy)
    C = xy[P]; px = PAD + (C[:, 0] - mn[0]) * sc; py = H - 60 - PAD - (C[:, 1] - mn[1]) * sc
    img = Image.new("RGB", (W, H), (14, 14, 18)); dr = ImageDraw.Draw(img)
    n = len(P); nseg = 96
    for k in range(nseg):
        a = k * n // nseg; b = min(n, (k + 1) * n // nseg + 1)
        r, g, bl = colorsys.hsv_to_rgb(k / nseg, 0.75, 0.95)
        dr.line(list(zip(px[a:b], py[a:b])), fill=(int(r * 255), int(g * 255), int(bl * 255)), width=1)
    # kroki karne na czerwono
    kk = np.arange(1, n); pen = (kk % 10 == 0) & (prime[P[:-1]] == 0); pp = np.where(pen)[0]
    for i in pp[::max(1, len(pp) // 6000)]:
        dr.point((px[i], py[i]), fill=(255, 60, 60))
    dr.rectangle([0, H - 60, W, H], fill=(24, 24, 30))
    dr.text((PAD, H - 52), f"{s:,.0f}".replace(",", " ") + f"   (plain {plain:,.0f}, kara {s - plain:,.0f})".replace(",", " "), fill=(235, 235, 240), font=FONT)
    dr.text((W - 780, H - 46), label, fill=(160, 170, 185), font=FONT2)
    img.save(out, optimize=True); return s


def history():
    files = [f for f in glob.glob(f"{RES}/sub_*.csv") if os.path.getsize(f) > 1_000_000 and not re.search(r"synth|t20k|lkh", f, re.I)]
    files.sort(key=os.path.getmtime); best = 1e18; k = 0
    for f in files:
        m = re.search(r"_(\d+)\.csv$", f)
        if not m: continue
        sc_ = float(m.group(1))
        if sc_ >= best: continue
        try:
            s = D.score(D.read_tour(f), xy, prime)
        except Exception: continue
        if abs(s - sc_) > 1000 or s >= best: continue
        best = s; k += 1
        out = f"{FR}/h{k:03d}_{time.strftime('%Y%m%d_%H%M', time.localtime(os.path.getmtime(f)))}.png"
        if not os.path.exists(out):
            render(f, out, f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}  {os.path.basename(f)}")
            print("frame", out, round(s, 1), flush=True)


def loop():
    last = None
    while True:
        try:
            b = json.load(open(os.path.expanduser("~/santa/www/best.json")))
            f = f"{RES}/{b['file']}"; stamp = time.strftime("%Y%m%d_%H")
            out = f"{FR}/t{stamp}.png"
            if not os.path.exists(out):
                s = render(f, out, f"{time.strftime('%Y-%m-%d %H:%M')}  {b['file']}"); print("frame", out, round(s, 1), flush=True)
        except Exception as e:
            print("ERR", e, flush=True)
        time.sleep(600)


if __name__ == "__main__":
    if "--history" in sys.argv: history()
    if "--loop" in sys.argv: loop()
