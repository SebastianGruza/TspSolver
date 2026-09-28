"""Build the LinkedIn carousel PDF (1080x1080 pages) for the Santa 2018 experiment."""
import sys
from pathlib import Path
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "santa_2018_agent_7_dni.pdf"

# ---- progress chart: (hours since 21.09 00:48, best score from scratch) ----
PTS = [(0.05, 1590816), (0.6, 1576303), (1.4, 1566788), (7.4, 1550502), (10.4, 1549722), (13.4, 1547632),
       (16.4, 1546271), (18.8, 1545839), (31.4, 1541244), (39, 1538779), (43, 1538458), (55, 1536727),
       (67, 1531947), (79, 1528697), (81, 1528685), (103, 1527422), (139, 1527022), (145, 1526454), (152, 1526256)]
REFS = [(1529000, "dr Jakuczun, 9 dni: 1 529 000"), (1516256, "LKH z karą (wzorzec): 1 516 256"),
        (1512000, "czołówka Kaggle 2018: ≈1 512 000")]
LABELS = [(0.05, 1590816, "LS na GPU, 1. godzina", 8, -10), (18.8, 1545839, "bramki: LK 8×5×3, okno 3000", 8, -8),
          (31.4, 1541244, "frag (mini-ILS w bloku GPU)", 8, -8), (67, 1531947, "uncross bez kary", 8, -8),
          (79, 1528697, "ulga 0 + naprawa", 8, 16), (103, 1527422, "or-opt w łańcuchu LK", 8, 18),
          (152, 1526256, "okno LK 5000: 1 526 256", -8, 22)]


def chart(w=960, h=470):
    x0, x1, y0, y1 = 70, w - 20, 24, h - 44
    hmax, vlo, vhi = 156, 1510000, 1595000
    X = lambda t: x0 + (x1 - x0) * t / hmax
    Y = lambda v: y1 - (y1 - y0) * (v - vlo) / (vhi - vlo)
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="IBM Plex Sans, DejaVu Sans, sans-serif">']
    for v in range(1510000, 1600000, 10000):
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#2a2f3a" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="12" fill="#8a8f9a">{v/1e6:.2f}M</text>')
    for d in range(0, 7):
        t = d * 24
        s.append(f'<line x1="{X(t):.1f}" y1="{y0}" x2="{X(t):.1f}" y2="{y1}" stroke="#2a2f3a" stroke-width="1"/>')
        s.append(f'<text x="{X(t+12):.1f}" y="{y1+18}" text-anchor="middle" font-size="12" fill="#8a8f9a">dzień {d+1}</text>')
    for v, lab in REFS:
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="#f2b134" stroke-width="1.2" stroke-dasharray="6 5"/>')
        s.append(f'<text x="{x1-4}" y="{Y(v)-5:.1f}" text-anchor="end" font-size="12.5" fill="#f2b134">{lab}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(t):.1f},{Y(v):.1f}" for i, (t, v) in enumerate(PTS))
    s.append(f'<path d="{path}" fill="none" stroke="#3ec1b0" stroke-width="3" stroke-linejoin="round"/>')
    for t, v in PTS:
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.5" fill="#0f1116" stroke="#3ec1b0" stroke-width="2"/>')
    for t, v, lab, dx, dy in LABELS:
        anchor = "end" if dx < 0 else "start"
        s.append(f'<text x="{X(t)+dx:.1f}" y="{Y(v)+dy:.1f}" text-anchor="{anchor}" font-size="12.5" fill="#e8e6df">{lab}</text>')
    s.append(f'<text x="{x0}" y="{y0-8}" font-size="12" fill="#8a8f9a">start: Hilbert 1 842 660 (poza skalą)</text>')
    s.append("</svg>")
    return "\n".join(s)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Sans+Condensed:wght@700&family=IBM+Plex+Mono:wght@500&display=swap');
@page { size: 1080px 1080px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; background: #0f1116; color: #e8e6df; font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif; font-size: 17.5px; line-height: 1.38; }
.page { width: 1080px; height: 1080px; padding: 58px 64px 52px; page-break-after: always; position: relative; display: block; overflow: hidden; }
.page > * + * { margin-top: 14px; }
.page:last-child { page-break-after: auto; }
.kicker { font-size: 13.5px; letter-spacing: .12em; text-transform: uppercase; color: #f2b134; font-weight: 600; }
h1 { font-family: "IBM Plex Sans Condensed", "DejaVu Sans Condensed", sans-serif; font-weight: 700; font-size: 60px; line-height: 1.04; margin: 0; text-wrap: balance; }
h2 { font-family: "IBM Plex Sans Condensed", "DejaVu Sans Condensed", sans-serif; font-weight: 700; font-size: 38px; line-height: 1.08; margin: 0; text-wrap: balance; }
p { margin: 0; }
.lead { font-size: 20px; color: #cfd2d8; }
.num { font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; }
.big { font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-size: 74px; font-weight: 500; color: #3ec1b0; line-height: 1; }
.foot { position: absolute; left: 64px; right: 64px; bottom: 26px; display: flex; justify-content: space-between; font-size: 12.5px; color: #6f7480; }
table { border-collapse: collapse; width: 100%; font-size: 15.5px; }
th { text-align: left; font-weight: 600; color: #8a8f9a; padding: 5px 10px 6px 0; border-bottom: 1px solid #3a4050; font-size: 13.5px; text-transform: uppercase; letter-spacing: .06em; }
td { padding: 6px 10px 6px 0; border-bottom: 1px solid #23283266; vertical-align: top; }
td.n, th.n { text-align: right; font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.hi td { color: #3ec1b0; font-weight: 600; }
ul { margin: 0; padding-left: 20px; } li { margin: 0 0 6px; } li::marker { color: #f2b134; }
.two { display: flex; gap: 26px; align-items: flex-start; } .two > * { flex: 1 1 0; min-width: 0; }
.fig { background: #07080b; border: 1px solid #23283a; border-radius: 4px; overflow: hidden; }
.fig img { display: block; width: 100%; height: auto; }
.cap { font-size: 13.5px; color: #8a8f9a; margin-top: 6px; }
.lesson { border-left: 4px solid #f2b134; padding: 8px 14px; background: #171a22; font-size: 17px; }
.lesson b { color: #f2b134; }
.grid3 { display: flex; gap: 18px; } .grid3 > div { flex: 1; background: #171a22; padding: 14px 16px; border-radius: 4px; }
.grid3 .v { font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-size: 30px; color: #3ec1b0; }
.grid3 .k { font-size: 13.5px; color: #8a8f9a; text-transform: uppercase; letter-spacing: .06em; }
.small { font-size: 14.5px; color: #aeb2bb; }
.chart { background: #0b0d12; border: 1px solid #23283a; border-radius: 4px; padding: 10px 8px 4px; }
.spacer { height: 40px; }
.cover { background: linear-gradient(180deg, #0f1116 0%, #0f1116 55%, #07080b 100%); }
.cover .fig { border: none; }
"""


def page(n, kicker, body, title=None):
    t = f"<h2>{title}</h2>" if title else ""
    return f'<section class="page"><div class="kicker">{kicker}</div>{t}{body}<div class="foot"><span>Santa 2018 Prime Paths · agent Claude Fable 5.1 + RTX 3090</span><span>{n} / 15</span></div></section>'


P = []

# 1 — cover
P.append(f'''<section class="page cover">
<div class="kicker">Eksperyment · 21–27 września 2026 · 7 dni</div>
<h1>Co, jeżeli dać agentowi klasy Fable 5.1 kartę RTX 3090 i sporą autonomię w zadaniu TSP?</h1>
<p class="lead">Kaggle Traveling Santa 2018 – Prime Paths: 197 769 miast, co 10. krok +10 %, chyba że zaczyna się
w mieście o numerze pierwszym. Solver pisany od zera, bez LKH. Człowiek decyduje o kierunku, agent projektuje, koduje, mierzy.</p>
<div class="fig"><img src="../../images/santa_final_1526256.png" style="height:540px; object-fit:cover; object-position:center"></div>
<div class="two" style="align-items:flex-end">
<div><div class="big">1 526 256</div><div class="small" style="margin-top:6px">wynik końcowy od zera · kolor = pozycja na trasie</div></div>
<div class="small" style="text-align:right">2 744 poniżej wyniku dr. Jakuczuna (9 dni, 45 eksperymentów)<br>10 000 (0,66 %) powyżej trasy LKH z karą</div>
</div>
<div class="foot"><span>Santa 2018 Prime Paths · agent Claude Fable 5.1 + RTX 3090</span><span>1 / 15</span></div>
</section>''')

# 2 — rules & references
P.append(page(2, "Zadanie i reguły gry", f'''
<div class="two">
<div>
<h2>Cztery liczby odniesienia. Jedna reguła: żaden wynik bez przeliczenia po pełnej cenie.</h2>
<p style="margin-top:14px">Inspiracja: karuzela dr. Wita Jakuczuna (DS360) o własnym solverze w Julii: 9 dni, 45 eksperymentów, 2 112 000 → 1 529 000.
Jego wzorcem była zwykła trasa LKH z doliczoną karą.</p>
<p style="margin-top:12px"><b>Reguły przyjęte na starcie:</b></p>
<ul>
<li>Od zera: własne operatory na GPU, bez LKH i jego „serca” (bez kandydatów α Helda–Karpa z subgradientem π, bez sekwencyjnego 5-opt). LKH policzony tylko jako punkt odniesienia.</li>
<li>Każdy zapisany wynik przeliczony niezależnie od zera. Zero dryfu liczników.</li>
<li>Krótkie biegi, plateau = stop. Hipoteza przechodzi bramkę: 10 min × 3 seedy od tej samej trasy, mediana, próg ≈ 3× rozrzut, zapisany przed biegiem.</li>
<li>Dziennik co 12 h z liczbami, decyzjami i wpadkami agenta.</li>
</ul>
</div>
<div>
<table>
<tr><th>punkt odniesienia</th><th class="n">wynik</th><th class="n">my vs</th></tr>
<tr><td>czołówka Kaggle 2018</td><td class="n">≈1 512 000</td><td class="n">+0,94 %</td></tr>
<tr><td>LKH z karą (wzorzec doktora)</td><td class="n">1 516 256</td><td class="n">+0,66 %</td></tr>
<tr><td>LKH + nasze operatory (linia pomocnicza, dzień 1)</td><td class="n">1 516 106</td><td class="n">—</td></tr>
<tr><td>dr Jakuczun, od zera, 9 dni</td><td class="n">1 529 000</td><td class="n">−2 744</td></tr>
<tr class="hi"><td>ten eksperyment, od zera, 7 dni</td><td class="n">1 526 256</td><td class="n">—</td></tr>
</table>
<div class="grid3" style="margin-top:22px">
<div><div class="v">1</div><div class="k">RTX 3090, dzielona</div></div>
<div><div class="v">197 769</div><div class="k">miast</div></div>
<div><div class="v">~31</div><div class="k">eksperymentów E1–E31</div></div>
</div>
<p class="small" style="margin-top:18px">Sprzęt: VM w homelabie, 12 rdzeni, jedna karta dzielona z innymi usługami (od dnia 6 stale ok. 10 GB VRAM zajęte przez inne modele).
Trasa Santa to 800 kB, więc VRAM nie był ograniczeniem; ograniczeniem był czas GPU.</p>
</div>
</div>'''))

# 3 — curve
P.append(page(3, "Krzywa siedmiu dni", f'''
<h2>Od 1 842 660 do 1 526 256. Sześć skoków, każdy to inny operator, nie dłuższy bieg.</h2>
<div class="chart">{chart()}</div>
<table style="font-size:14.5px">
<tr><th>dzień</th><th>skok</th><th class="n">wynik</th><th>skąd zysk</th></tr>
<tr><td>1</td><td>LS na GPU (wątek na pozycję), potem łańcuch LK</td><td class="n">1 576 303</td><td>−266 k w godzinę od krzywej Hilberta</td></tr>
<tr><td>2</td><td>bramki: szerokość łańcucha 8×5×3, okno 3000, kandydaci α</td><td class="n">1 545 839</td><td>−4,7 k; głębokość 8 i K=16 odrzucone</td></tr>
<tr><td>2–3</td><td>frag: fragment 128 pozycji jako mini-ILS w bloku GPU</td><td class="n">1 541 244</td><td>−4,6 k w jedną noc</td></tr>
<tr><td>3–4</td><td>uncross bez kary, potem ulga 0 w fazie szukania + naprawa</td><td class="n">1 528 697</td><td>−12,5 k; poniżej doktora po 3 dniach</td></tr>
<tr><td>5</td><td>or-opt jako krok w łańcuchu LK</td><td class="n">1 527 422</td><td>jedyny operator z zyskiem na twardym plateau</td></tr>
<tr><td>7</td><td>okno łańcucha LK 5000 zamiast 3000</td><td class="n">1 526 256</td><td>4× większe tempo na plateau</td></tr>
</table>'''))

# 4 — day 1 architecture
P.append(page(4, "Dzień 1 · architektura", f'''
<h2>Wątek na pozycję trasy, nie na osobnika. Pierwsza godzina: −266 000.</h2>
<div class="two">
<div>
<p>Kernel populacyjny frameworku (wątek na osobnika, O(n) na ruch) jest bezużyteczny przy 197 k miast. Decyzja agenta w 5. minucie:
jedna trasa, GPU wycenia równolegle wszystkie ruchy (wątek = pozycja), partie ruchów o rozłącznych oknach nakładane na CPU.</p>
<ul style="margin-top:12px">
<li><b>Kara liczona dokładnie na zakresie.</b> Kara zależy od pozycji w trasie, więc każdy ruch jest wyceniany na dotkniętym odcinku. Test: zysk ruchu na GPU = różnica pełnego wyniku (1e-6).</li>
<li><b>Or-opt L≤3, 2-opt, bloki 10/20/30</b> (przesunięcie o wielokrotność 10 nie zmienia siatki kary; pomysł z karuzeli doktora), swap, prime na pozycję karną.</li>
<li><b>Łańcuch LK na GPU</b> od ustalonego t1: sekwencja 2-opt, kopia okna per wątek, DFS 5×3×2, oba kierunki. Pierwsza wersja znalazła 27 łańcuchów: błąd granicy odwrócenia. Po naprawie 5 723 łańcuchy, −14,5 k.</li>
<li><b>LKH policzony na boku</b>: 3 h → plain 1 503 070, z karą 1 516 795; nasz LS+LK na tej trasie: 1 516 106. Czyli poniżej wzorca doktora, ale w „nie jego dyscyplinie”. Odłożone.</li>
</ul>
<div class="lesson" style="margin-top:14px"><b>Diagnoza kary (E5).</b> Na trasie LKH tylko 5 opłacalnych ruchów „prime na pozycję karną”. Kara nie zdejmuje się tanio: luka to długość i kara razem, więc potrzebna jest klasa ruchów LK z karą od pierwszej rundy.</div>
</div>
<div>
<div class="fig"><img src="../../images/santa_day1_hilbert_ls.png"></div>
<p class="cap">Dzień 1, 1 566 117 po ILS: krzywa Hilberta domknięta przez miasto 0. Szew przez pół mapy (krawędź 3 525 przy medianie 6,6) kosztował 0,23 %. Operatory lokalne nie mogły go usunąć. Zauważony dopiero z obrazka w dniu 3.</p>
<table style="margin-top:10px; font-size:14px">
<tr><th>krok, dzień 1</th><th class="n">wynik</th><th class="n">czas</th></tr>
<tr><td>krzywa Hilberta</td><td class="n">1 842 660</td><td class="n">—</td></tr>
<tr><td>GPU LS (Or-opt, 2-opt, okno 1000)</td><td class="n">1 590 816</td><td class="n">50 s</td></tr>
<tr><td>+ łańcuch LK</td><td class="n">1 576 303</td><td class="n">6 rund</td></tr>
<tr><td>+ ILS z LK, 6 h</td><td class="n">1 550 502</td><td class="n">plateau</td></tr>
</table>
</div>
</div>'''))

# 5 — gates
P.append(page(5, "Dzień 2 · bramki zamiast wyczucia", f'''
<h2>Gdy krzywa stoi, kupujemy pomiar, nie pomysł. Cztery hipotezy, jeden zwycięzca.</h2>
<div class="two">
<div>
<p class="small">Bramki 1: start z trasy 1 550 5xx, 10 min × 3 seedy, mediana, próg = 3× rozrzut ≈ 280.</p>
<table style="margin-top:8px">
<tr><th>wariant</th><th class="n">mediana</th><th class="n">vs baza</th><th class="n">rozrzut</th><th>werdykt</th></tr>
<tr><td>baza: LK 5×3×2, gł. 5, K=10, okna 400</td><td class="n">1 550 230</td><td class="n">—</td><td class="n">92</td><td></td></tr>
<tr class="hi"><td>A: rozgałęzienie LK 8×5×3</td><td class="n">1 549 722</td><td class="n">−509</td><td class="n">72</td><td>przyjęty</td></tr>
<tr><td>B: głębokość 8</td><td class="n">1 550 202</td><td class="n">−29</td><td class="n">93</td><td>remis</td></tr>
<tr><td>C: 16 najbliższych kandydatów</td><td class="n">1 550 326</td><td class="n">+96</td><td class="n">64</td><td>gorszy</td></tr>
<tr><td>D: mniejsze okna kopnięć (150)</td><td class="n">1 550 145</td><td class="n">−85</td><td class="n">76</td><td>pod progiem</td></tr>
</table>
<p class="small" style="margin-top:16px">Bramki 2: start z plateau 1 546 271 (bieg 2 przerwany po 6 h: ostatnie 2 h dawały ~8 na iterację), próg ≈ 145.</p>
<table style="margin-top:8px">
<tr><th>wariant</th><th class="n">mediana</th><th class="n">vs baza</th><th class="n">rozrzut</th><th>werdykt</th></tr>
<tr><td>baza 2 (A, okno 1000)</td><td class="n">1 546 241</td><td class="n">—</td><td class="n">48</td><td>+30 w 10 min</td></tr>
<tr><td>E: okno łańcucha 3000</td><td class="n">1 545 970</td><td class="n">−271</td><td class="n">58</td><td>przyjęty</td></tr>
<tr><td>F: kandydaci α z 1-drzewa (K=8)</td><td class="n">1 546 134</td><td class="n">−107</td><td class="n">43</td><td>pod progiem</td></tr>
<tr class="hi"><td>G: E + F</td><td class="n">1 545 879</td><td class="n">−362</td><td class="n">52</td><td>nowa baza</td></tr>
</table>
</div>
<div>
<div class="lesson"><b>Szerokość bije głębokość.</b> Szerszy wybór na wczesnych poziomach łańcucha dał −509; głębsze łańcuchy bez szerszego wyboru nic.</div>
<div class="lesson" style="margin-top:12px"><b>Więcej sąsiadów szkodzi.</b> K=16 pogorszył wynik: czas oceny rośnie, a dalsi sąsiedzi nie wchodzą do dobrych łańcuchów. Następna hipoteza to <i>inny</i> zbiór kandydatów, nie większy.</div>
<div class="lesson" style="margin-top:12px"><b>Zasięg łańcucha był realnym ograniczeniem.</b> Okno 3000 samo daje −271. Kolejność kandydatów (α) ma znaczenie dopiero, gdy łańcuch ma gdzie sięgnąć: sama nie przechodzi progu, w złożeniu dokłada 91.</div>
<div class="lesson" style="margin-top:12px"><b>Pomiar przed biegiem.</b> Pokrycie krawędzi trasy LKH: 10 najbliższych 99,99 %, α8 100,00 %. Wniosek przed testem: α nie zmieni pokrycia, tylko kolejność. Efekt mały, zgodnie z prognozą.</div>
<p class="small" style="margin-top:14px">Baza w 10 minut daje 30. Każdy przyjęty wariant to realny skok, nie szum. Zwycięzca bramki dostaje bieg nocny; przegrany nie dostaje drugiej szansy „na wyczucie”.</p>
</div>
</div>'''))

# 6 — frag
P.append(page(6, "Dzień 2–3 · zmiana skali", f'''
<h2>Frag: 1 545 fragmentów po 128 pozycji, każdy z własnym ILS w bloku GPU. −4 600 w jedną noc.</h2>
<div class="two">
<div>
<p>Pomysł Sebastiana: „okna dokładne”. Najpierw Held–Karp na fragmentach trasy z zamrożonymi końcami (k = 12/14/16; kara dokładna,
bo pozycje we fragmencie są stałe). Zysk rósł z k (−125, −193, −321) i miał najmniejszy rozrzut ze wszystkich operatorów, ale 2¹⁴ stanów na wątek to sufit dokładności.</p>
<p style="margin-top:12px"><b>Sufit dokładności zamieniony na sufit heurystyki (E16):</b> jeden blok GPU na fragment 128 pozycji w pamięci współdzielonej; wewnątrz
kopnięcie double-bridge → 2-opt + Or-opt do optimum lokalnego z dokładną karą → powrót do best. 128 wątków wycenia ruchy, wątek 0 redukuje, cztery przesunięcia podziału na przebieg.
Pierwszy test na najlepszej trasie: −338 w 97 s. Bramka G dawała −362 w 10 minut.</p>
<table style="margin-top:14px; font-size:14.5px">
<tr><th>ramię bandyty (noc 3, rollout 5 min × 2)</th><th class="n">mediana</th><th class="n">zysk/s</th></tr>
<tr class="hi"><td>frag, 30 kopnięć, ulga 0,06</td><td class="n">1 542 755</td><td class="n">10,3</td></tr>
<tr><td>frag, 30 kopnięć</td><td class="n">1 542 781</td><td class="n">10,2</td></tr>
<tr><td>frag, 10 kopnięć</td><td class="n">1 543 605</td><td class="n">7,4</td></tr>
<tr><td>DP k=16 + frag 10</td><td class="n">1 543 931</td><td class="n">6,4</td></tr>
<tr><td>ulga 0,06 / DP cykl / DP 16 (bez frag)</td><td class="n">~1 545 480</td><td class="n">1,2</td></tr>
<tr><td>baza G</td><td class="n">1 545 790</td><td class="n">0,2</td></tr>
</table>
</div>
<div>
<div class="grid3" style="flex-direction:column; gap:12px">
<div><div class="v">−362 / 10 min</div><div class="k">najlepsza bramka LK, dzień 2</div></div>
<div><div class="v">−338 / 97 s</div><div class="k">frag przy 10 kopnięciach, pierwszy test</div></div>
<div><div class="v">1 541 244</div><div class="k">po 70 min gry bandyty, rekord od zera</div></div>
</div>
<div class="lesson" style="margin-top:14px"><b>Skąd skok.</b> Liczba kopnięć na sekundę: 1 545 fragmentów × 30 kopnięć naraz zamiast 200 kopnięć na iterację ILS na całej trasie.</div>
<div class="lesson" style="margin-top:12px"><b>DP w złożeniu z frag przeszkadza.</b> Zabiera budżet słabszemu operatorowi. Więcej kopnięć = szybciej (30 vs 10: +40 %).</div>
<div class="lesson" style="margin-top:12px"><b>Plateau przychodzi ostro.</b> Po ~1,5 h zyski spadły 30×. Frag z 10–30 kopnięciami wyczerpał klasę optimów; wersje „więcej tego samego” (100 kopnięć, F=256, Or-3opt) dały &lt;0,1 pkt/s.</div>
</div>
</div>'''))

# 7 — penalty relief bug
P.append(page(7, "Dzień 3–4 · ulga w karze", f'''
<h2>„Tryb naprzemienny” nigdy u nas nie działał. Dwa błędy, jeden fix, −9 000 w trzy godziny.</h2>
<div class="two">
<div>
<p>Doktor pisał: zadziałało naprzemienne szukanie <i>jakby kary nie było</i> z uczciwą wyceną best; nie zadziałał jeden długi bieg bez kary z naprawą na końcu.
Nasz ILS miał od dnia 1 fazę ulgi (kara 0,03) i „dawał mało”.</p>
<ul style="margin-top:12px">
<li><b>Błąd 1.</b> Bramka nocna: ramię „kara 0” identyczne co do bitu z „kara 0,03”. Skrypt uruchamiający nie przekazywał parametru kary do ILS. Przez trzy dni ILS zawsze szukał z pełną karą; wszystkie wnioski o relaksacji mierzyły co innego.</li>
<li><b>Błąd 2.</b> Po naprawie: kara 0 w ILS = zysk 0. 118 z 197 okien wychodziło gorzej po prawdziwej cenie, a LK bez kary miesza miasta między oknami, więc nie dało się ich cofnąć. Bezpiecznik odrzucał całe iteracje.</li>
<li><b>Brakujący krok: naprawa.</b> Szukaj jak bez kary → wróć do pełnej ceny (LS/LK z karą) → dopiero potem uczciwa ocena okien. To jest pełne zdanie doktora.</li>
</ul>
<table style="margin-top:14px; font-size:14.5px">
<tr><th>ramię (od 1 538 458, 10 min × 3)</th><th class="n">zysk</th><th class="n">rozrzut</th></tr>
<tr><td>pełna kara w ILS (stan sprzed fixu)</td><td class="n">−68</td><td class="n">74</td></tr>
<tr><td>ulga 0,03 + naprawa</td><td class="n">−293</td><td class="n">106</td></tr>
<tr class="hi"><td>ulga 0 + naprawa</td><td class="n">−523</td><td class="n">76</td></tr>
<tr><td>ulga 0,03, kopnięcia celowane w „złe” okna</td><td class="n">−292</td><td class="n">22</td></tr>
</table>
</div>
<div>
<div class="grid3" style="flex-direction:column; gap:12px">
<div><div class="v">−9 151 / 2 h 55</div><div class="k">linia od trasy G z ulgą 0 + naprawą</div></div>
<div><div class="v">1 528 697</div><div class="k">dzień 4 rano: poniżej doktora (1 529 000) po 3 dniach</div></div>
</div>
<div class="lesson" style="margin-top:14px"><b>Wzór doktora powtórzony co do kierunku:</b> 0 &gt; 0,03 &gt; 0,1. Im bliżej pełnej ceny, tym mniejszy zysk.</div>
<div class="lesson" style="margin-top:12px"><b>Dźwignia jest w procedurze, nie w trasie startowej.</b> Linia wystartowała z trasy o 7,4 k gorszej od rekordu i skończyła 1,2 k lepiej.</div>
<div class="lesson" style="margin-top:12px"><b>Kopnięcia celowane</b> (∝ „zło” krawędzi: nadwyżka nad odległością do sąsiada + kara): ten sam zysk, rozrzut 5× mniejszy. Stabilizują, nie przyspieszają.</div>
<div class="lesson" style="margin-top:12px"><b>Gdzie jest luka.</b> Nasza kara 10,7 k vs LKH 13,7 k, nasz plain +24,7 k vs LKH. Luka to długość, nie kara. Płacimy za primes mniej niż wzorzec.</div>
</div>
</div>'''))

# 8 — operator's eye
P.append(page(8, "Dzień 3 · oko operatora", f'''
<h2>Człowiek patrzący na obrazek znalazł w dwie minuty to, czego 60 godzin operatorów lokalnych nie widziało.</h2>
<div class="two">
<div>
<div class="fig"><img src="../../images/crossings_before_uncross.jpg"></div>
<p class="cap">Podgląd trasy (kolor = pozycja w trasie). Zielone i fioletowe nitki przecinają się na całej mapie: 1 789 skrzyżowań na trasie greedy po 2,5 h ILS. Żaden operator w oknie tego nie widzi, bo końce krawędzi są odległe w trasie o dziesiątki tysięcy pozycji.</p>
</div>
<div>
<ul>
<li><b>Szew Hilberta.</b> Jedna krawędź 3 525 przy medianie 6,6, przez pół mapy; 0,23 % wyniku, wtedy połowa luki. Głęboki LK bez okna: 0 ruchów. Lek: inny start (greedy-edge + 2-opt długich krawędzi + uncross: 1 679 690 zamiast 1 842 660).</li>
<li><b>uncross (E21).</b> 2-opt rozplątujący skrzyżowania między dowolnie odległymi w trasie krawędziami (kd-tree wzdłuż odcinka). Z pełną karą: 0 ruchów, bo odwrócenie długiego odcinka przetasowuje siatkę kary. <b>Bez kary + naprawa:</b> 872 ruchy w 4 s, skrzyżowań 1 789 → 11, wynik 1 539 838 → 1 532 713.</li>
<li><b>orlong (E22)</b>, „wszyć filament w region sąsiadów”: Or-opt segmentów 3…5000 bez limitu odległości, kara w O(1) przez prefiksy po resztach mod 10. −1 744 w 37 s na rekordzie.</li>
<li><b>excursion merge (E24)</b>, z podglądu Grecji: dwie wyprawy na tę samą wyspę scalone w jedną. Kandydaci z listy sąsiadów przegapiali ruch +459; dopiero pełny przegląd punktów wstawienia go znalazł.</li>
</ul>
<div class="lesson" style="margin-top:14px"><b>Reguła, która została do końca.</b> Każdy operator „daleki” liczy bez kary i naprawia po. Pełna kara blokuje ruchy siatką pozycji, nie geometrią.</div>
</div>
</div>'''))

# 9 — starts
P.append(page(9, "Dzień 4 · starty", f'''
<h2>Start decyduje o czasie pierwszego kroku. O wyniku decydują operatory dalekie.</h2>
<table>
<tr><th>konstrukcja startowa</th><th class="n">surowa</th><th class="n">po LS/LK (krok 1)</th><th class="n">po ILS z ulgą 0 + post</th><th>uwaga</th></tr>
<tr><td>krzywa Hilberta</td><td class="n">1 842 660</td><td class="n">1 576 303 (1 h)</td><td class="n">1 530 522</td><td>szew 3 525</td></tr>
<tr><td>Moore (zamknięty Hilbert)</td><td class="n">1 835 476</td><td class="n">—</td><td class="n">—</td><td>max krawędź 324, ale gęstość zignorowana</td></tr>
<tr><td>Moore z medianami (adaptacja do gęstości)</td><td class="n">1 979 000+</td><td class="n">—</td><td class="n">—</td><td>gorszy: odrzucony</td></tr>
<tr class="hi"><td>greedy-edge + 2-opt długich + uncross</td><td class="n">1 679 690</td><td class="n">1 553 012 (8 min)</td><td class="n">1 528 710</td><td>filamenty na łączeniach fragmentów</td></tr>
<tr><td>komórki (partycja Karpa, 1024 komórek)</td><td class="n">1 576 699</td><td class="n">1 552 096</td><td class="n">1 535 860 (2,5 h, ubity)</td><td></td></tr>
<tr><td>gmoore (fragmenty greedy wzdłuż Moore)</td><td class="n">1 806 383</td><td class="n">1 543 388 (1 h)</td><td class="n">—</td><td></td></tr>
<tr><td>lądy v3 (regiony z gęstości, greedy w regionach, makrotrasa)</td><td class="n">1 710 405</td><td class="n">1 542 833</td><td class="n">1 530 774</td><td>najlepszy krok 1</td></tr>
</table>
<div class="two" style="margin-top:6px">
<div>
<div class="lesson"><b>Po ILS wszystkie linie zbiegają do 1 528,7–1 531 k.</b> Greedy skraca krok 1 z godziny do 8 minut, lądy dają najlepszy krok 1, ale rekord wychodzi z procedury (ulga 0, uncross, orlong, frag), nie ze startu.</div>
<div class="lesson" style="margin-top:12px"><b>Krzyżowania między liniami nie działają.</b> GPX (podmiana segmentów o tym samym zbiorze miast i końcach, kara dokładna) Hilbert × greedy: 0 podmian z ~250 kandydatów. Trasy z różnych startów nie mają wspólnych segmentów; GPX działa tylko w obrębie jednej linii (−131…−197). OX z naprawą (segment 70 %): dziecko 1 528 687, poziom rodziców, nie przebicie.</div>
</div>
<div>
<div class="fig"><img src="../../images/art_tour_1.jpg"></div>
<p class="cap">Skąd ten projekt: wcześniejszy solver rysował trasy przez obrazy (art-TSP). Ten sam podgląd tras, który w dniu 3 pokazał szew i skrzyżowania, zaczął się od takich rysunków.</p>
</div>
</div>'''))

# 10 — plateau breakers
P.append(page(10, "Dzień 5–7 · twarde plateau", f'''
<h2>Trzy dni na 2 400 punktów: or-opt jako krok łańcucha, szersze okno, szybkie ścieżki bez kary.</h2>
<div class="two">
<div>
<ul>
<li><b>Or-opt w łańcuchu LK (E27).</b> Dotąd łańcuch = same odwrócenia (2-opt). Dodany krok „segment 1–3 za t3” w dwóch wariantach, wybór typu kroku po dokładnym zysku z karą. Na plateau zwykłego LK (0 łańcuchów) rekord Santa: 125 łańcuchów, 40/40 zgodnych z pełnym przeliczeniem. Bramka 20 min × 3 od rekordu: baza +0, or-opt −212. Bieg 4 h: 1 528 685 → 1 527 422.</li>
<li><b>Szybka ścieżka bez kary (E28).</b> Frag i LK liczyły koszt kandydata na odcinku O(c), bo kara zależy od pozycji. Przy karze 0 (faza ulgi, TSPLIB) delta to 4–6 odległości: frag 17× szybciej, LK z or-opt 18×, zyski identyczne co do bitu. Krok 1 z or-opt stał się tańszy niż bez niego dzień wcześniej.</li>
<li><b>Okno łańcucha 5000 zamiast 3000 (ostatnia noc).</b> Tempo na plateau −390 / 6 h wobec −80…−100 / 4 h. Łańcuchom z or-opt brakowało zasięgu, znów. 1 527 022 → 1 526 454 → 1 526 256.</li>
</ul>
<div class="lesson" style="margin-top:14px"><b>Na twardym plateau liczy się klasa ruchu, nie budżet.</b> Bramka „baza” od rekordu przez 20 minut bez frag: +0. Każdy z trzech zysków powyżej pochodził ze zmiany tego, co łańcuch może zrobić lub jak daleko sięga.</div>
</div>
<div>
<table>
<tr><th>bieg</th><th class="n">wynik</th><th>zmiana</th></tr>
<tr><td>rekord dnia 4 (ox1p)</td><td class="n">1 528 685</td><td></td></tr>
<tr><td>play8, 4 h</td><td class="n">1 527 422</td><td>or-opt w LK, frag 100</td></tr>
<tr><td>play9 / play10 / play11, po 4 h</td><td class="n">1 527 102 / 1 527 022 / 1 526 850</td><td>plateau 80–100 na rundę</td></tr>
<tr><td>play12, 6 h</td><td class="n">1 526 454</td><td>okno LK 5000</td></tr>
<tr class="hi"><td>play13, 4 h + post</td><td class="n">1 526 256</td><td>okno 5000, seed 113</td></tr>
</table>
<div class="grid3" style="margin-top:18px">
<div><div class="v">1 514 940</div><div class="k">długość plain rekordu</div></div>
<div><div class="v">11 317</div><div class="k">kara (LKH: 13 700)</div></div>
</div>
<p class="small" style="margin-top:14px">Pełna walidacja końcowa: 197 769 miast, permutacja poprawna, wynik przeliczony niezależnie od solvera. Do wzorca LKH z karą zostało 10 000 (0,66 %), do czołówki Kaggle 14 256 (0,94 %). Cała ta luka to długość trasy, nie kara.</p>
</div>
</div>'''))

# 11 — what didn't work
P.append(page(11, "Co nie zadziałało (z liczbami)", f'''
<h2>Czternaście chybień. Każde kosztowało bramkę, nie noc.</h2>
<div class="two">
<div>
<table style="font-size:14.5px">
<tr><th>hipoteza</th><th>wynik</th></tr>
<tr><td>głębszy łańcuch LK (8 zamiast 5)</td><td>−29, remis ze szumem</td></tr>
<tr><td>16 najbliższych kandydatów zamiast 10</td><td>+96, gorszy</td></tr>
<tr><td>mniejsze okna kopnięć (150)</td><td>−85, pod progiem</td></tr>
<tr><td>okna dokładne DP k=16 w złożeniu z frag</td><td>gorzej niż sam frag (zabiera budżet)</td></tr>
<tr><td>xswap segmentów równej długości</td><td>~10 kandydatów na rundę, za rzadki</td></tr>
<tr><td>frag „więcej tego samego” (100 kopnięć, F=256, Or-3opt)</td><td>&lt;0,1 pkt/s na plateau</td></tr>
<tr><td>długi bieg plain + naprawa na końcu</td><td>ubity po 5 min (wariant negatywny u doktora)</td></tr>
<tr><td>kara ujemna w fazie ulgi (0 / −0,03 / −0,10)</td><td>wszystkie ramiona bit w bit: pomiar bez rozstrzygnięcia</td></tr>
<tr><td>GPX między liniami z różnych startów</td><td>0 podmian z ~250 kandydatów</td></tr>
<tr><td>OX jako przebicie dwóch linii</td><td>1 528 687: poziom rodziców</td></tr>
<tr><td>Moore adaptowany do gęstości (mediany)</td><td>1 979 000+, gorszy od Hilberta</td></tr>
<tr><td>makro-TSP nad segmentami trasy</td><td>0 (kolejność już lokalnie optymalna)</td></tr>
<tr><td>lądy v1/v2 (porty + DP)</td><td>remis z greedy; dopiero v3 wyrównała</td></tr>
<tr><td>hybryda populacja + LK: kick / worst (TSPLIB 2–5 k)</td><td>0,54 % / 0,69 % vs 0,51 % czystej populacji</td></tr>
</table>
</div>
<div>
<div class="lesson"><b>Wzorzec chybień.</b> „Więcej tego samego” (głębiej, więcej sąsiadów, więcej kopnięć) prawie nigdy nie przechodziło progu. Przechodziło to, co zmieniało <i>klasę</i> ruchu albo <i>zasięg</i> w trasie.</div>
<div class="lesson" style="margin-top:12px"><b>Pomiary bez rozstrzygnięcia też się liczą.</b> Bramka kary ujemnej od trasy na plateau bez frag dała cztery identyczne wyniki: to nie „kara ujemna nie działa”, to „ten start nie ma rozdzielczości”. Zapisane jako otwarte, nie jako odrzucone.</div>
<div class="lesson" style="margin-top:12px"><b>Krzyżowanie potrzebuje rodziców z osobnych linii,</b> ale osobne linie nie mają wspólnych segmentów. Na tej instancji populacja tras to zły model; jedna trasa i operatory dalekie to dobry.</div>
<p class="small" style="margin-top:14px">Do tego wpadki procesu: siedem razy agent ubił własną sesję (wzorzec do zabicia zawarty we własnej komendzie; reguła: kill tylko po PID), dwa procesy na jeden log (dwa mechanizmy startu na jedno zdarzenie), best nadpisany gorszą trasą (−850), cichy segfault w numba (odczyt poza tablicą, 50 min), pierwszy GPX bez filtra = 10 GB RAM. Wszystko w dzienniku.</p>
</div>
</div>'''))

# 12 — TSPLIB one code
P.append(page(12, "Dzień 5–6 · jeden kod dla każdej instancji", f'''
<h2>41 instancji TSPLIB jednym wejściem, parametry i budżet z n: mediana luki 0,198 %.</h2>
<div class="two">
<div>
<p>Polecenie: „jeden kod, czas i parametry zależne od n”. Loader TSPLIB (EUC_2D, ATT, GEO), Santa jako przypadek szczególny z karą.
Fragmenty na 1 000 miast dały 3,11 %, bo iteracja ILS trwała 18–50 s i nic nie zostawało na perturbację. Decyzja Sebastiana: małe instancje populacja + OX, duże fragmenty.</p>
<table style="margin-top:12px; font-size:14.5px">
<tr><th style="width:24%">strefa</th><th style="width:46%">tryb</th><th class="n">luka</th></tr>
<tr><td style="white-space:nowrap">≤ 2 000 miast (31 inst.)</td><td>populacja 8 192 osobników na GPU, OX, kolonie, tabu; dogrywka LK/or-opt</td><td class="n">0,00–0,94 %, mediana ~0,16 %</td></tr>
<tr><td style="white-space:nowrap">2 000–5 000 (7 inst.)</td><td>populacja 4 096 / 2 048 + polerowanie elity naszym LK (hybryda)</td><td class="n">0,48–1,89 %</td></tr>
<tr><td style="white-space:nowrap">≥ 5 000 (3 inst.)</td><td>fragmenty: greedy → LS/LK → ILS z ulgą 0</td><td class="n">1,01–1,74 %</td></tr>
</table>
<table style="margin-top:12px; font-size:14.5px">
<tr><th>instancja</th><th class="n">n</th><th class="n">luka</th><th class="n">czas</th><th class="n">poprzedni solver</th></tr>
<tr><td>pr1002</td><td class="n">1 002</td><td class="n">0,20 %</td><td class="n">11 min</td><td class="n">1,19 %</td></tr>
<tr><td>pcb1173</td><td class="n">1 173</td><td class="n">0,16 %</td><td class="n">—</td><td class="n">2,05 %</td></tr>
<tr><td>rl1889</td><td class="n">1 889</td><td class="n">0,32 %</td><td class="n">21 min</td><td class="n">1,96 %</td></tr>
<tr><td>d2103 (hybryda)</td><td class="n">2 103</td><td class="n">0,48 %</td><td class="n">28 min</td><td class="n">0,66 %</td></tr>
<tr><td>fnl4461</td><td class="n">4 461</td><td class="n">1,89 %</td><td class="n">2,7 h</td><td class="n">2,47 %</td></tr>
<tr><td>gr9882 (Grecja)</td><td class="n">9 882</td><td class="n">1,01 %</td><td class="n">5 h</td><td class="n">3,57 %</td></tr>
</table>
</div>
<div>
<div class="fig"><img src="../../images/gr9882_303930.png"></div>
<p class="cap">gr9882, 303 930 (1,01 %). Wyspy i wybrzeża: stąd operatory „wyprawy” i starty „lądy”. Frag na 9 882 miastach hamował (154 bloki), LK z oknem 2 000 zamiast całej trasy: 2 s zamiast 17 s na przebieg bez straty.</p>
<div class="lesson" style="margin-top:12px"><b>Dlaczego fragmenty są słabe przy 2–5 k.</b> Bez kary kopnięcia ILS w oknach LK cofa natychmiast (12 iteracji, 0 okien gorszych, 0 poprawy). Na Santa ten schemat działa, bo kara „przykleja” kopnięcia. Bez kary perturbacją jest OX w populacji.</div>
<p class="small" style="margin-top:12px">16 instancji ≤ 0,1 % (12 na optimum), 35 ≤ 1 %, najgorsza fnl4461 1,89 %. Każda instancja lepiej niż poprzedni solver z 2025, zwykle kilkukrotnie. Po przetłumaczeniu kodu na angielski sweep regresyjny &gt;1 000 miast odtworzył wyniki (deterministyczne bit w bit).</p>
</div>
</div>'''))

# 13 — recipe
P.append(page(13, "Przepis od zera", f'''
<h2>Jeden skrypt, 5 godzin, 1 527 835. Wynik po 5 h ≈ wynik po 4 dniach.</h2>
<div class="two">
<div>
<p>Polecenie z dnia 3: zapisywać zwycięskie sekwencje, na koniec gotowa sekwencja bez bandyty. Zmierzona odtwarzalność:</p>
<table style="margin-top:12px">
<tr><th>krok</th><th>co</th><th class="n">wynik</th><th class="n">czas</th></tr>
<tr><td>0</td><td>greedy-edge + 2-opt długich krawędzi + uncross</td><td class="n">1 679 690</td><td class="n">2 min</td></tr>
<tr><td>1</td><td>LS + łańcuch LK z or-opt (kara pełna)</td><td class="n">1 536 885</td><td class="n">46 min</td></tr>
<tr><td>2</td><td>ILS z ulgą 0: w fazie szukania or-opt-LK, frag 100, xswap2, orlong, uncross, wyprawy; naprawa z pełną karą; ocena okien po pełnej cenie</td><td class="n">1 527 942</td><td class="n">3,5 h (13 iteracji)</td></tr>
<tr class="hi"><td>3</td><td>post: orlong → uncross bez kary → naprawa</td><td class="n">1 527 835</td><td class="n">~15 min</td></tr>
</table>
<div class="lesson" style="margin-top:14px"><b>Bez or-opt w łańcuchu krok 1 dawał 1 553 012.</b> Jeden krok w łańcuchu przesuwa optimum LS/LK o 16 000 i skraca resztę.</div>
<div class="lesson" style="margin-top:12px"><b>Dźwignie są w procedurze i operatorach, nie w czasie GPU.</b> Rekord 1 526 256 kosztował 7 dni; 1 527 835 z przepisu kosztuje 5 h na jednej karcie.</div>
</div>
<div>
<p class="kicker" style="margin-bottom:6px">Proces, który został</p>
<ul>
<li><b>Bramka przed biegiem:</b> 10 min × 3 seedy od tej samej trasy, mediana, próg ≈ 3× rozrzut ustalony przed biegiem. Remis = odrzucenie. Zwycięzca dostaje 1–4 h.</li>
<li><b>Plateau = stop.</b> ILS ma detektor zastoju; budżet GPU idzie na hipotezy, nie na mielenie krzywej, która stoi. Bieg 2 przerwany po 6 h, gdy ostatnie 2 h dawały ~8 na iterację.</li>
<li><b>Jedna zmienna naraz</b>, bandyta konfiguracji na noc (rollout 5 min × 2 seedy, gra 1 h ze stopem).</li>
<li><b>Dziennik co 12 h:</b> liczby, decyzja, wpadki. 16 cykli, żaden nie pominięty.</li>
<li><b>Dwa mechanizmy odbioru:</b> każdy operator ma test „zysk na GPU == różnica pełnego wyniku”; każdy zapisany rekord jest przeliczany niezależnie.</li>
<li><b>Człowiek:</b> kierunek (bez LKH, populacja vs fragmenty, plateau = stop), podgląd tras, pomysły operatorów (okna dokładne, xswap, uncross, orlong, wyprawy, OX). Agent: architektura, kod, testy, pomiar, decyzje w bramkach, łańcuchy nocne.</li>
</ul>
</div>
</div>'''))

# 14 — pictures of results
P.append(page(14, "Trasy", f'''
<h2>Ten sam kod, cztery instancje.</h2>
<div class="two">
<div><div class="fig"><img src="../../images/rl1889.png"></div><p class="cap">rl1889: 317 552 (0,32 %), populacja + dogrywka LK/or-opt, 21 min.</p></div>
<div><div class="fig"><img src="../../images/pr1002.png"></div><p class="cap">pr1002: 259 558 (0,20 %), 11 min. Poprzedni solver: 1,19 %.</p></div>
</div>
<div class="two">
<div><div class="fig"><img src="../../images/fl3795.png"></div><p class="cap">fl3795: 0,49 %, populacja 2 048, 43 min. Instancja „wiertarska” z długimi pustymi przelotami.</p></div>
<div><div class="fig"><img src="../../images/santa_day1_hilbert_ls.png"></div><p class="cap">Santa, dzień 1 (1 566 117) obok finału ze strony 1 (1 526 256): ta sama mapa, 40 000 mniej. Film z rekordów dzień po dniu: youtu.be/4v6lGGsZwS4</p></div>
</div>'''))

# 15 — conclusions
P.append(page(15, "Wnioski", f'''
<h2>Pięć rzeczy, które zabieram z tych siedmiu dni.</h2>
<ul style="font-size:19px; line-height:1.42">
<li style="margin-bottom:14px"><b>Człowiek z obrazkiem bije operatory lokalne.</b> Szew, filamenty, skrzyżowania, wyprawy na wyspy: każde z tych spostrzeżeń zajęło minuty i dało nowy operator. Agent liczył, człowiek patrzył. To nie jest podział „kto mądrzejszy”, tylko „kto co widzi”.</li>
<li style="margin-bottom:14px"><b>Każdy operator daleki liczy bez kary i naprawia po.</b> Pełna kara blokuje ruchy siatką pozycji, nie geometrią. To jedno zdanie odpowiada za ~20 000 z 316 000 zysku i za wszystkie skoki od dnia 3.</li>
<li style="margin-bottom:14px"><b>Plateau = stop, bramka przed biegiem, jedna zmienna naraz.</b> „Więcej tego samego” prawie nigdy nie przeszło progu. Przechodziła zmiana klasy ruchu albo zasięgu.</li>
<li style="margin-bottom:14px"><b>Dźwignie są w procedurze.</b> Przepis od zera w 5 h daje 1 527 835; rekord po 7 dniach 1 526 256. Różnica to 0,1 %. Wiedza z siedmiu dni mieści się w jednym skrypcie.</li>
<li style="margin-bottom:14px"><b>Jeden kod, dwa reżimy.</b> Populacja + OX poniżej ~5 000 miast, jedna trasa + operatory dalekie powyżej. 41 instancji TSPLIB, mediana 0,198 %, i Santa tym samym wejściem.</li>
</ul>
<div class="spacer"></div>
<div class="two" style="align-items:flex-end">
<div>
<p class="kicker" style="margin-bottom:6px">Czego nie zrobiliśmy</p>
<p class="small">Bez LKH i jego serca (π/α Helda–Karpa, 5-opt sekwencyjny) z wyboru: chodziło o uczciwe porównanie z solverem pisanym od zera. Do wzorca LKH z karą zostało 0,66 %, do czołówki Kaggle 0,94 %; cała luka to długość trasy, nie kara. To jest następny eksperyment.</p>
</div>
<div style="text-align:right">
<div class="big" style="font-size:52px">1 526 256</div>
<p class="small" style="margin-top:8px">7 dni · od zera · 1 × RTX 3090 (dzielona)<br>agent Claude Fable 5.1 (Claude Code) · człowiek: Sebastian Gruza<br>film: youtu.be/4v6lGGsZwS4</p>
</div>
</div>'''))

html = f"<!doctype html><html lang='pl'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(P)}</body></html>"
(HERE / "carousel.html").write_text(html)
HTML(string=html, base_url=str(HERE)).write_pdf(OUT)
print("OK", OUT, OUT.stat().st_size // 1024, "kB")
