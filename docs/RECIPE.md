# Santa 2018 Prime Paths — przepis od zera (bez bandyty, bez eksperymentów)

Cel: jedna sekwencja, którą uruchamia się raz i po 24–48 h daje wynik zbliżony do najlepszego, jaki osiągnęliśmy w projekcie.
Każdy krok = gotowa komenda `run.py` (repo `memetic-framework`, branch `santa`, `PYTHONPATH=~/mf-santa`, `CUDA_HOME=~/cudashim`,
RTX 3090). Status: **Z** = zmierzone w projekcie (liczby z dziennika), **P** = przypuszczenie do potwierdzenia.
Skrypt wykonujący całość: `scripts/santa_recipe.sh` (aktualizowany razem z tym plikiem).

Wspólne argumenty (baza G + operatory dnia 2, kara 0 w szukaniu z naprawą):
```
COMMON="--lk 3 --Lblk 30 --ils-margin 300 --lk-depth 5 --lk-br 8,5,3 --ils-w 400 --K 8 --lk-W 3000 --cand alpha --kick uniform --uncross 300 --orlong 3"
```

| # | krok | komenda (poza COMMON) | budżet | wynik oczekiwany | status |
|---|---|---|---|---|---|
| 1 | Start: **krzywa Moore'a** (zamknięty Hilbert, bez szwu z rogu do rogu) → LS (Or-opt L≤3, bloki 10/20/30, 2-opt, swap) → LS+LK naprzemiennie do zbieżności | `--start moore --time 3600 --pen 0.03 --relax-time 600 --full-time 600 --ils 0 --tag r1` | 1 h | Moore 1 835 476 (Hilbert 1 842 660) → ~1 570 000 | Z dla Hilberta (LS 1 590 816, LS+LK 1 576 303); P dla Moore'a (linia moore_r1/r2 w kolejce 23.09) |
| 2 | ILS z ulgą 0 + naprawa, frag100, xswap2 D=20 (jak lin_s1) | `--start best_r1.csv --time 600 --relax-time 300 --full-time 300 --pen 0.0 --frag 100 --xswap2 20 --ils 1000000 --ils-time 21600 --ils-rounds 2 --stall-iters 40 --stall-gain 150 --tag r2` | 6 h | −9 000 w 3 h od trasy 1 545 878 (lin_s1); od ~1 576 000 spodziewane ~1 540 000 | Z dla startu G, **P** dla startu z kroku 1 |
| 3 | Trzy linie: krok 2 powtórzony z seedami 21/22/23 od tej samej trasy (różnorodność pod GPX) — sekwencyjnie | `--seed 21|22|23 --ils-time 10800 --tag r3_sN` | 3 × 3 h | od tras G (1 545 8xx): 1 536 421 / 1 536 242 / 1 536 785 (−9.1…−9.6k każda) | Z (noc 4, 23.09) |
| 4 | GPX: krzyżowanie najlepszej trasy z trzema liniami (segmenty o tym samym zbiorze miast, kara dokładna) | `python ~/santa/gpx_merge.py child.csv best.csv lin1.csv lin2.csv lin3.csv` | 5 min | −131 (baza lin_s2 1 536 242 → 1 536 112, 131 podmian z 3 linii + gpx1) | Z (23.09) |
| 5 | Dogrywka: krok 2 od dziecka GPX, 4 h, stall-stop | `--start child.csv ... --ils-time 14400 --stall-iters 60 --stall-gain 100 --tag r5` | 4 h | ? | P (play7 nocy 4) |

Suma: ~1 + 6 + 9 + 4 = 20 h. Zapas do 48 h: powtórzenie kroków 3–5 drugi raz (druga generacja GPX).

## Co NIE wchodzi do przepisu (zmierzone jako bez efektu lub gorsze)
- K=16 kandydatów (+96), głębokość LK 8 (−29 = remis), okna ILS 150 (−85 pod progiem) — dzień 1.
- DP okna dokładne k=12/14/16 w złożeniu z frag: DP zabiera budżet słabszemu operatorowi (dp16_frag10 < frag10).
- xswap równej długości (E15) na trasie po LK: ~10 kandydatów/rundę; zastąpiony przez xswap2.
- frag v2 (F=256, Or-3opt, 100 kopnięć jako jedyny operator na plateau): <0.1 pkt/s.
- Jeden długi bieg bez kary z naprawą na końcu (`--relax-current`): u doktora bez efektu; u nas nie mierzony do końca.
- Ulga 0.03: −293/10 min vs ulga 0: −523/10 min (bramka nocy 4) → w przepisie ulga 0.
- Kopnięcia celowane (`--kick bad`): ten sam zysk co losowe, rozrzut 5× mniejszy — opcjonalne (stabilność, nie tempo).

## Szew Hilberta (23.09, obserwacja Sebastiana z podglądu trasy)
Trasa od Hilberta ma JEDNĄ krawędź 3 525 (mediana 6.6): ścieżka z rogu do rogu zamknięta w cykl przez miasto 0. Żaden ruch
lokalny jej nie zdejmie (do 2-opt potrzebna druga długa krawędź; łańcuch LK przesuwa wolny koniec ~7/poziom). Koszt ≈0.23%
= połowa luki do doktora. Lekarstwo: nie tworzyć — start z krzywej Moore'a (max krawędź 324). Podgląd: `tour_view.py`.

## Uncross (23.09, pomysł Sebastiana: operator z jego wersji na aparapi)
`domains/santa/uncross.py`: dla N najdłuższych krawędzi znajdź geometrycznie krawędzie, które je przecinają, i rozplącz
2-optem bez limitu odległości w trasie; kara pełnym przeliczeniem. Greedy: −26k w 86 s; trasa po pełnym pipeline (gpx2):
−869 w 3 s. Wchodzi do przepisu: w `--start greedy` (po greedy + 2-opt długich) i jako `--uncross 300` w każdym kroku.

## Orlong (23.09, pomysł Sebastiana: „wszyć" filament w region jego sąsiadów)
`domains/santa/orlong.py`: Or-opt segmentów o długości 3…5000 wstawianych przy geometrycznym sąsiedzie końca segmentu
(obie strony, obie orientacje), bez limitu odległości w trasie; kara w O(1) przez sumy prefiksowe po resztach mod 10.
Na rekordzie po pełnym pipeline: −1 744 w 37 s. Wchodzi do przepisu jako `--orlong 3` w każdym kroku (+ CPU post-processing
`orlong.py` → `uncross.py` po każdym kroku).

## Uncross bez kary (23.09 wieczór, obserwacja Sebastiana: skrzyżowania zieleni z fioletem)
Skrzyżowania krawędzi odległych w trasie nie znikają przy pełnej karze: odwrócenie długiego odcinka przetasowuje siatkę
kary. `uncross.py --all --pen 0` (wszystkie krawędzie, kandydaci z KNN, kryterium czystej długości) + naprawa kary
LS/LK (`run.py --ils 0 --time 600`): greedy_r2 1 539 838 → 1 532 713 w 4 min. W ILS: uncross_all/orlong w rundzie ulgi
liczą się z karą fazy (0), naprawa w rundzie pełnej kary.

## ZMIERZONE 25.09: przepis od zera jednym skryptem (`~/santa/chain_recipe_fast.sh`)
greedy + 2-opt długich + uncross (2 min) → LS/LK z or-opt w łańcuchu (46 min: 1 536 885) → ILS ulga 0 z or-opt, frag100,
xswap2, orlong, uncross, wyprawy (4 h: 1 527 869) → post orlong/uncross/naprawa (4 min: **1 527 835**). Razem 5 h,
wynik poniżej doktora (1 529 000) i o 413 od rekordu po 4 dniach (1 527 422). Dla TSPLIB: `solve.py` (populacja ≤5000,
fragmenty powyżej) — 41 instancji, mediana 0.198% (TSPLIB_RESULTS.md).

## Historia zmian
- 23.09 08:40: pierwsza wersja (po nocy 4: ulga 0 + naprawa, linie w toku).
- 23.09 19:45: po każdym kroku post-processing: orlong → uncross --all --pen 0 → naprawa run.py 10 min (Z: −7.1k na greedy_r2).
- 23.09 17:50: `--orlong 3` we wszystkich krokach + post-processing orlong→uncross po kroku (P: noc 5b).
- 23.09 16:35: uncross w kroku 1 (start greedy) i `--uncross 300` we wszystkich krokach (P: linia greedy nocy 5).
- 25.09 20:45: przepis zmierzony końcowo (chain_recipe_fast.sh, 5 h → 1 527 835); or-opt w LK we wszystkich krokach.
- 24.09 09:50: nowy kandydat kroku 1: `--start cell --cell-C 1024` (partycja Karpa: komórki wzdłuż Moore, ścieżka w komórce): start 1 576 699 = Hilbert po 1 h LS/LK, filamentów 884 vs 8889 (greedy). Test kroków 1–2 w łańcuchu cell (24.09).
- 23.09 16:10: kandydaci na krok 1: `--start greedy` (greedy-edge + 2-opt długich krawędzi: 1 705 742, max 730) vs `--start moore` (1 835 476, max 324); linie obu w kolejce (chain_starts.sh), decyzja po wynikach r2. Wariant Moore z medianami (amoore) odrzucony: gorszy na starcie.
- 23.09 15:30: krok 1 → start moore (szew Hilberta), status P do czasu linii moore.
- 23.09 14:40: kroki 3 i 4 zmierzone (linie −9.1…−9.6k; GPX −131). Uwaga: linie z osobnych startów G, nie z jednej trasy — w skrypcie linie startują z R2 (jedna trasa); różnorodność z seedów jest słabsza (sib5/6: −17 w GPX) → P: w kroku 3 lepiej użyć 3 różnych tras z kroku 1 (3 seedy LS/LK) niż jednej R2.
