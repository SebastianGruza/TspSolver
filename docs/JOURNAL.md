# Dziennik eksperymentu: „Co, jeżeli dać agentowi klasy Fable 5.1 dostęp do VM z RTX 3090 i sporą autonomię w zadaniu TSP?"

Zadanie: Kaggle Traveling Santa 2018 – Prime Paths (197 769 miast, co 10. krok +10%, chyba że z liczby pierwszej).
Punkty odniesienia: czołówka Kaggle ≈1 512 000; trasa LKH z karą 1 516 256 (wzorzec dr. Jakuczuna); dr Jakuczun po 9 dniach
i 45 eksperymentach: 1 529 000 (od zera, własny solver w Julii). Sprzęt: 1× RTX 3090 (24 GB), 12 rdzeni, VM w homelabie.
Agent: Claude Fable 5.1 (Claude Code), człowiek podejmuje decyzje kierunkowe, agent projektuje, koduje, testuje, mierzy.
Zasada z karuzeli doktora przyjęta od pierwszej minuty: każdy zapisany wynik przeliczony od zera po pełnej cenie.

Wpisy co ~6 h. Format: czas · co zrobiono · liczby · decyzja · co dalej.

---

## 21.09.2026, 00:30–01:40 (T+0 … T+70 min)

**Start.** Zadanie od Sebastiana: branch dedykowany, zmieścić się w 20 GB VRAM, zbliżyć się do 1.529M.

**E1 · Architektura (decyzja agenta, 00:35).** Kernel frameworku (wątek na osobnika, tablicowe 2-opt/Or-opt, O(n) na ruch)
jest bezużyteczny przy n=197k. Nowy branch `santa`: jedna trasa, GPU ocenia równolegle ruchy (wątek na pozycję), partie
ruchów o rozłącznych oknach na CPU. VRAM przestaje być ograniczeniem (trasa = 800 kB).

**E2 · Harness (00:40).** data/score/validate/submission, KNN, Hilbert, sqlite z biegami, test „zysk ruchu na GPU ==
różnica pełnego wyniku" (1e-6). LKH-3.0.13 skompilowany jako punkt odniesienia na CPU.

**E3 · GPU LS (00:45).** Or-opt L≤3 (przed/po, ±odwrócenie), 2-opt, okno W=1000, dokładna delta z karą.
Syntetyk 197k: Hilbert 1.772M → 1.371M w 50 s. Prawdziwe dane (cities.csv od 00:48): Hilbert 1 842 660 → 1 590 816.

**E4 · Trasa LKH + nasz LS (01:07).** LKH po 4 min: plain 1 503 177, z karą 1 516 760; nasz LS na niej: 1 516 183 po 23 s.
Poniżej wzorca doktora (1 516 256) i 0.84% poniżej jego wyniku. Ale to „nie jego dyscyplina" (on szedł od zera).

**E5 · Diagnoza kary (01:10).** 19 776 pozycji karnych, 17 802 liczb pierwszych (90%), na pozycjach karnych 1 852 (losowo).
7 755 karnych pozycji ma prime w ±3. Celowany ruch „prime na pozycję karną z ±10": poprawny (5/5 exact), ale tylko 5
opłacalnych ruchów na trasie LKH. Wniosek: kara nie zdejmuje się tanio; luka = długość+kara łącznie → potrzebna klasa LK.

**E6 · Wskazówki z karuzeli doktora (01:00).** Wdrożone: bloki Or-opt 10/20/30 (neutralne dla kary), kara jako parametr
kernela (relaksacja 0.03 z best po pełnej cenie), swap, bramki bench.py (10 min × 3 seedy, mediana, próg przed biegiem).

**E7 · DECYZJA (Sebastian, 01:13): implementujemy łańcuch LK na GPU; LKH tylko jako wzorzec.**

**E8 · LK na GPU (01:15–01:30).** LK jako sekwencja 2-opt od ustalonego t1, kopia okna per wątek, DFS 5×3×2, głębokość 5,
oba kierunki, delta z karą dokładnie. Pierwsza wersja: 27 łańcuchów na optimum Or-opt — BŁĄD granicy odwrócenia
(odwracano [1..c] zamiast [1..c-1], t4 = pred(t3)). Po naprawie: 5 723 łańcuchy, zysk 11k/rundę, głębokość śr. 3.5;
na trasie LKH 947 łańcuchów (bo liczy karę). Exactness 31/31.

**E9 · Bieg od zera v3 z LK (01:24).** LS 1 590 816 → LS+LK 1 576 303 (LK: −14.5k, 0.9%, gaśnie po 6 rundach)
→ relaksacja + ILS z LK 40 min (w toku). Po nim: `chained_lk_6h` (kopnięcia w oknach + LK, 6 h).

**Wpadki agenta:** dwa razy zerwana sesja ssh przez `pkill -f` z wzorcem obecnym we własnej komendzie (reguła: kill po PID).
Czujka odpaliła LS na trasie pośredniej LKH zamiast końcowej (LKH zapisuje trasę po każdej poprawie).

**Stan na 01:40:** od zera 1 576 303 (przed ILS); LKH plain 1 503 126 (29 min z 180); LKH+nasz LS 1 516 164.
Dashboard: http://192.168.80.220:8902.

## 21.09.2026, 02:13 (T+1 h 43 min) · wpis cyklu 1

**Biegi.** v3 od zera: ILS z LK 37 min, best **1 566 788** (od 1 576 303 po LS+LK: −9.5k w 44 iteracjach, ~200 okien
kopnięć na iterację, 5–10% okien cofanych jako gorsze). Za chwilę kończy 40-minutowy budżet, potem automatycznie
`chained_lk_6h` (ILS z LK, 6 h, od najlepszej trasy v3). LS na pośrednich trasach LKH zakończony: seed 1 **1 516 165**
(kara 12 530), seed 2 1 516 249 — oba poniżej wzorca doktora (1 516 256). LKH liczy dalej: plain 1 503 106 po 71 min.

**Obserwacja.** Na trasie LKH nasz LS+ILS zdejmuje z kary tylko ~1.1k z 13.6k (do 12.5k). Potwierdza E5: kara siedzi
w geometrii ciasnej trasy, nie w rozstawieniu liczb pierwszych; czołówka (1.512M) musiała optymalizować karę i długość
razem od początku. Dla toru „od zera" to argument, żeby kara była w LK od pierwszej rundy (jest).

**Tempo toru od zera.** −9.5k/37 min w ILS to za wolno na noc: potrzeba ~35k do doktora i ~50k do LKH-plain+kara.
Kandydaci do bramek po nocy (bench.py, 10 min × 3 seedy): (a) rozgałęzienie LK 5×3×2 → 8×5×3, (b) głębokość 5 → 8,
(c) K 10 → 16, (d) okna kopnięć 400 → 150 (więcej, mniejszych), (e) więcej rund LK na iterację ILS.

**Wpadki:** brak nowych. Dashboard :8902 i /journal działają.

**Stan:** od zera 1 566 788 (liczy) · LKH+nasz LS 1 516 165 · LKH plain 1 503 106 (liczy) · cele: 1 529 000 / 1 516 256 / 1 512 000.

## 21.09.2026, 08:13 (T+7 h 43 min) · wpis cyklu 2

**LKH skończył (3 h × 2 seedy).** plain 1 503 070 / 1 503 076, z karą 1 516 795 / 1 516 831. Trzy godziny dały ~100 więcej
niż trasa z 4. minuty — wzorzec jest stabilny. Tor „LKH + my" (LS+LK+ILS 20 min na końcowych trasach): **1 516 106**
(kara 12 509) i 1 516 136. LK dołożył ~60 względem samego LS (1 516 165). Poniżej wzorca doktora o 150, poniżej
doktora o 12.9k (0.84%).

**Tor od zera: `chained_lk_6h` (ILS z LK, 200 okien kopnięć/iterację).** Krzywa: 1 566 117 (02:17) → 1 556 411 (04:07)
→ 1 553 040 (06:20) → **1 550 502** (08:10, iteracja 755). Tempo spadło z ~200/iter do ~7/iter: plateau. Luka do
doktora 21.5k (1.4%), do LKH+kara 33.7k (2.2%), do LKH-plain 3.2%.

**E10 · Diagnoza plateau (agent).** Kopnięcia double-bridge w oknach 400 przy 200 oknach/iterację dają ~1 poprawę na
15 okien i cofanie ~6% okien. LK ma rozgałęzienie 5×3×2 i głębokość 5 z kandydatami = 10 najbliższych; LKH używa
5 kandydatów alfa (1-drzewo) i 5-opt bez ograniczenia okna. Hipotezy do bramek (10 min × 3 seedy od tej samej trasy,
mediana, próg = 3× rozrzut): A rozgałęzienie 8×5×3; B głębokość 8; C K=16; D okna kopnięć 150 (więcej, mniejszych).

**E11 · Decyzja procesu.** Bieg 6 h kończy o 08:17; GPU idzie na bramki (2.5 h), potem drugi bieg nocny ze zwycięską
konfiguracją. Zgodnie z karuzelą: „gdy krzywa stoi, kup pomiar, nie pomysł".

**Wpadki:** stary waiter nie wygasł i uruchomił drugi egzemplarz `chained_lk_6h` równolegle z nową czujką (2 procesy na
jeden log); ubite po PID, start od nowa, strata 2 min. Lekcja: jeden mechanizm startu na zdarzenie.

**Stan:** od zera **1 550 502** · LKH+my **1 516 106** · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

## 21.09.2026, 11:10 (T+10 h 40 min) · wynik bramek E10

Start wspólny: trasa z końca `chained_lk_6h` (1 550 5xx). Budżet 10 min, 3 seedy, mediana; próg = 3× rozrzut ≈ 280.

| wariant | mediana | vs baseline | rozrzut | werdykt |
|---|---|---|---|---|
| baseline (LK 5×3×2, gł. 5, K=10, okna 400) | 1 550 230 | — | 92 | — |
| **A: rozgałęzienie LK 8×5×3** | **1 549 722** | **−509** | 72 | **PRZYJĘTY** |
| B: głębokość 8 | 1 550 202 | −29 | 93 | remis, odrzucony |
| C: K=16 | 1 550 326 | +96 | 64 | gorszy, odrzucony |
| D: okna kopnięć 150 | 1 550 145 | −85 | 76 | poniżej progu, odrzucony |

Lekcje: (1) szerokość na wczesnych poziomach łańcucha bije głębokość — głębsze łańcuchy bez szerszego wyboru nic
nie dają; (2) więcej najbliższych sąsiadów szkodzi (czas oceny rośnie, dalsi sąsiedzi nie wchodzą do dobrych
łańcuchów) → następna hipoteza to INNY zbiór kandydatów (alfa z 1-drzewa, jak w LKH), nie większy; (3) mniejsze okna
kopnięć to szum. Uruchomiony bieg nocny 2: `chained2_br853_8h` (A, 8 h, seed 2) od najlepszej trasy.

**Stan:** od zera 1 549 7xx (liczy) · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

## 21.09.2026, 14:13 (T+13 h 43 min) · wpis cyklu 3

**Bieg nocny 2 (`chained2_br853_8h`, LK 8×5×3, seed 2) po 3 h:** 1 549 906 → **1 547 632** (iteracja 345, ~31 s na
iterację, ~8/iter). Cofanych okien 2–4%, czyli kopnięcia prawie zawsze się „naprawiają" do tego samego poziomu — ILS
jest w reżimie dyfuzji, nie schodzenia. Prognoza na koniec (19:10): ~1.545M. Luka do doktora 18.6k (1.2%).

**E12 · Co ogranicza LK teraz (analiza agenta).** Trzy różnice względem LKH, które zostały: (1) okno pozycji W=1000
w łańcuchu — LKH nie ma limitu zasięgu, u nas łańcuch nie może dotknąć miast odległych w trasie; doktor pisał, że
horyzont przestał być przeszkodą dopiero po przejściu na LK bez limitu; (2) kandydaci = 10 najbliższych, a K=16 pogorszył
wynik → kandydaci alfa z 1-drzewa (LKH: 5 alfa-najbliższych bije 10 najbliższych geometrycznie); (3) brak „don't-look
bits" i kolejki aktywnych miast (doktor: dwie próby, pomogło za mało). Do bramek po biegu: E = W_lk 3000, F = alfa.

**Decyzja:** nie przerywać biegu 2 (daje dane o plateau przy szerokim LK); w tym czasie implementacja alfa-kandydatów
na CPU (MST po grafie KNN, alfa(i,j) = d(i,j) − maks. krawędź na ścieżce MST) i test poprawności bez GPU.

**Wpadki:** brak. Dashboard i dziennik na żywo.

**Stan:** od zera **1 547 632** (liczy) · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

**E13 · Kandydaci alfa, pomiar przed testem (14:40).** MST na grafie KNN(16), alfa(i,j) = d − maks. krawędź na ścieżce
MST (drzewo Kruskala + LCA), 4 s na CPU. Pokrycie krawędzi trasy LKH: knn5 99.87%, knn8 99.99%, knn10 99.99%;
alfa5 99.89%, alfa8 100.00%. Wniosek przed biegiem: na tej instancji alfa nie zmienia POKRYCIA, może zmienić tylko
KOLEJNOŚĆ kandydatów w łańcuchu (LK bierze pierwsze spełniające kryterium). Oczekiwany efekt: mały. Bramki 2 po biegu
nocnym 2: base2 vs E (okno LK 3000) vs F (alfa8) vs G (E+F), 10 min × 3 seedy.

**E14 · Okna dokładne (pomysł Sebastiana, 17:24–17:45).** Podział TRASY na fragmenty po k pozycji z zamrożonymi
końcami; każdy wątek rozwiązuje fragment optymalnie (Held–Karp po podzbiorach, m=k−2 miast wewnętrznych) z karą liczoną
dokładnie, bo pozycje we fragmencie są stałe. Fragmenty rozłączne → poprawy aplikowane naraz; przebieg = k offsetów.
Bez procedur naprawczych: dzielimy trasę, nie mapę, permutacja z konstrukcji. Test: optimum DP == brute force
(720 permutacji) na 30 oknach, zysk == różnica pełnego wyniku; na małym syntetyku od Hilberta: 2 089 w jednym
offsecie, 2 712 w pełnym przebiegu. Ruch, którego LK nie ma (optimum globalne w oknie + idealne primes lokalnie).
Do bramek jako ramię H (`--dp 12`/`--dp 14`) po bench 2. Równolegle: zasada „plateau = stop" (ILS stall-stop) po
uwadze Sebastiana, bieg 2 przerwany na 1 546 271 po 6 h, bramki 2 ruszyły od tej trasy.

**E15 · Wymiana między odległymi częściami trasy (pomysł Sebastiana, 17:49–18:05).** Obserwacja, którą przeoczyłem:
zamiana dwóch miast (lub segmentów RÓWNEJ długości) nie przesuwa żadnej innej pozycji, więc delta z karą to
O(L) — 2 krawędzie graniczne na segment plus kroki wewnątrz — bez żadnego ograniczenia odległości w trasie. Dotychczasowy
swap liczył to na całym oknie O(W) i tylko w promieniu 1000 pozycji. Nowy kernel `xswap`: dla miasta i jego 10 sąsiadów
geometrycznych (oraz ich poprzedników/następników) segmenty L=1..3, 4 warianty odwróceń, partia z dwoma oknami
konfliktów. Test: 15/15 ruchów zgodnych z pełnym przeliczeniem, 13 z nich „dalekich" (>200 pozycji) — czyli ruchy,
których LK w oknie ani DP nie widzą. Ramię I w bramkach 3 i w bandycie.

## 21.09.2026, 19:35 (T+19 h) · wpis cyklu 4 · bramki 2

Start: trasa z końca biegu 2 (1 546 271, przerwanego ręcznie na plateau). 10 min × 3 seedy, mediana, próg ≈ 3× rozrzut ≈ 145.

| wariant | mediana | vs baseline | rozrzut | werdykt |
|---|---|---|---|---|
| base2 (LK 8×5×3, okno 1000, K=10 knn) | 1 546 241 | — | 48 | plateau: +30 w 10 min |
| **E: okno łańcucha LK 3000** | **1 545 970** | **−271** | 58 | PRZYJĘTY |
| F: kandydaci alfa (K=8) | 1 546 134 | −107 | 43 | pod progiem sam, ale… |
| **G: E + F** | **1 545 879** | **−362** | 52 | **PRZYJĘTY jako nowa baza** |

Lekcje: (1) zasięg łańcucha był realnym ograniczeniem (jak u doktora: „horyzont"); (2) alfa sama nie przechodzi, ale
w złożeniu z szerszym oknem dokłada 91 — kolejność kandydatów ma znaczenie dopiero, gdy łańcuch ma gdzie sięgnąć;
(3) baseline w 10 min daje 30, więc każdy przyjęty wariant to realny skok, nie szum.

**Zmiana procesu (Sebastian, 17:13): plateau = stop.** Bieg 2 przerwany po 6 h (ostatnie 2 h to ~8/iter), ILS ma
detektor plateau, budżet GPU idzie na bramki i nowe operatory. Zaimplementowane w tym cyklu i czekające na pomiar
(bramki 3, od 19:31): E14 okna dokładne DP k=12/14/16 (Held–Karp na fragmentach trasy, kara dokładna), E15 xswap
(wymiana segmentów równej długości L≤3 między dowolnymi częściami trasy, delta O(L), bez limitu odległości).
Na noc: bandyta konfiguracji (`bandit.py`) na bazie G, ramiona dp12/14/16, xswap3, relaksacja 0.06, okno 5000.

**Stan:** od zera **1 545 839** (najlepszy seed G) · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.
Luka do doktora: 16.8k (1.1%).

## 21.09.2026, 20:15 (T+19.5 h) · wpis cyklu 5 · bramki 3 w toku, cykl okien DP

**Bramki 3 (od 19:31, ta sama baza co bramki 2: trasa 1 546 271, okno 1000, K=10 knn — pomiar izolowany).**
Próg ≈ 145. Pierwszy wynik:

| wariant | mediana | vs base2 | rozrzut | werdykt |
|---|---|---|---|---|
| H_dp12: okna dokładne k=12 | 1 546 116 | −125 | **27** | tuż pod progiem, ale najstabilniejszy ze wszystkich wariantów |
| H_dp14 | w toku (s1 po 7.5 min: 1 546 050) | | | zapowiada się lepiej niż dp12 |
| H_dp16, I_xswap3 | kolejka do ~21:50 | | | |

Co widać w logu dp14: pierwszy offset daje 37 popraw / 20 jednostek, kolejne offsety po kilka–kilkanaście — DP zbiera
drobne, deterministyczne zyski, których LK nie widzi (optimum globalne w oknie z idealnym ułożeniem primes). Rozrzut 27
(vs 48–58 dla LK-wariantów) potwierdza: to operator „porządkujący", nie losujący.

**Sebastian (20:15):** przesuwać podział systematycznie i cyklicznie powtarzać k=12 → 14 → 16 → 12…; jeśli k=16 da
ładne poprawy, pomyśleć o oknach rzędu 100 pozycji z heurystyką (nie pełny przegląd — wykładniczy). Stan: przesuwanie
podziału już jest (każdy przebieg DP = wszystkie k offsetów). Cykl k dodany: `--dp 12,14,16` (kolejna iteracja ILS =
kolejne k), ramię `dpcyc` w bandycie. Okna ~100 z heurystyką = „forma 2" z kolejki: blok GPU na fragment, pełny
Or-opt/2-opt/LK w pamięci współdzielonej na fragmencie o zamrożonych końcach — to jest naturalny następny krok po DP,
bo DP k=16 to sufit dokładności (2^14 stanów na wątek), a fragment 100 to sufit heurystyki.

**Decyzja procesowa:** bandyta `night3` wystartuje po bramkach 3 z NAJLEPSZEJ trasy G (1 545 839), nie z trasy sprzed
bramek 2 — ramiona są już na bazie G, start z gorszej trasy o 432 marnowałby pierwszą godzinę gry. Ramiona:
G_base, dp12, dp14, dp16, dpcyc, xswap3, relax6, W5000; rollout 300 s × 2 seedy, gra 1 h ze stall-stopem, 6 cykli.

**Stan:** od zera **1 545 839** · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.
Luka do doktora: 16.8k (1.1%). Wpadki cyklu: brak.

**Bramki 3 — komplet (22:51):** H_dp12 −125 (rozrzut 27), H_dp14 −193 (33), **H_dp16 −321 (18)**, I_xswap3 −106 (37).
Okna dokładne rosną z k i mają najmniejszy rozrzut ze wszystkich operatorów; samo k=16 ≈ cała paczka G. xswap na trasie
po LK znajduje ~10 kandydatów/rundę — za rzadki jako samodzielny operator (dalej: nierówne długości, bloki ×10).

**E16 · Forma 2: fragment 128 pozycji z mini-ILS w bloku GPU (pomysł Sebastiana „okna ~100 z heurystyką", 22:55–23:20).**
Po tym, jak k=16 potwierdziło trend, sufit dokładności (2^14 stanów) zamieniamy na sufit heurystyki: jeden blok GPU na
fragment F=128 z zamrożonymi końcami, fragment w pamięci współdzielonej, w środku ILS: kopnięcie double-bridge wewnątrz
fragmentu → 2-opt + Or-opt (L≤3, oba kierunki, okno 32) do lokalnego optimum z karą DOKŁADNĄ (pozycje stałe) → powrót do
best. 128 wątków ocenia ruchy równolegle (wątek = pozycja), wątek 0 redukuje, wszystkie wątki budują nową kolejność przez
indeks wirtualny (`_slot_after`). 1545 fragmentów naraz, 4 przesunięcia podziału. Pierwszy test na NAJLEPSZEJ trasie G
(optimum LS+LK+ILS): **−338 w 97 s** przy 10 kopnięciach (bramka G dawała −362 w 10 min). Wynik zgodny z pełnym
przeliczeniem. Wpięty jako `--frag KICKS`; bandyta night3 zatrzymany po 5 min (nic nie stracił), bramki 4 (J_frag10,
K_dp16+frag10) na bazie bramek 3, potem bandyta night3b z ramionami frag10 i dp16_frag10 od trasy G.
Wpadki: bandyta padł na `No module named domains` (skrypt w podkatalogu → PYTHONPATH); PIĄTY self-kill: `pgrep -f
"bandit.py --start"` w tej samej komendzie ssh, w której stał tekst uruchomienia — zasada „PID z osobnej komendy" znów.

## 22.09.2026, 08:13 (T+31 h) · wpis cyklu 6 · frag zmienia skalę, potem ostry plateau

**Noc w liczbach.** Start nocy: 1 545 839 (trasa G). Bieg J_frag10 (bez limitu zbieżności, stary kod bez zapisu):
1 546 271 → 1 541 716 w 80 min, ubity o 00:25 na ~15/min (trasa stracona — wada naprawiona: zapis best co iterację,
1fcecee). Bandyta night3b od trasy G, 9 ramion, rollout 5 min × 2 seedy:

| ramię | mediana | zysk/s |
|---|---|---|
| **frag30_relax6** | **1 542 755** | 10.3 |
| frag30 | 1 542 781 | 10.2 |
| frag10 | 1 543 605 | 7.4 |
| frag10_dpcyc | 1 543 717 | 7.1 |
| dp16_frag10 | 1 543 931 | 6.4 |
| relax6 / dpcyc / dp16 | ~1 545 480 | 1.2 |
| G_base | 1 545 790 | 0.2 |

Gra frag30_relax6, 70 min: **1 545 839 → 1 541 244** (rekord od zera, zapisany). Cykl 1 (od 1 541 244): frag10 −82,
dp16_frag10 −63, dpcyc −27 w 5 min — zyski spadły 30×. Frag z 10–30 kopnięciami wyczerpał się na tej trasie.

**Lekcje.** (1) Operator fragmentowy to zmiana skali: −4 600 w jedną noc vs −362 za cały dzień bramek; różnica bierze
się z liczby kopnięć na sekundę (1545 fragmentów × 30 kopnięć naraz vs 200 kopnięć/iterację w ILS na całej trasie).
(2) Więcej kopnięć = szybciej (30 vs 10: +40%), a DP w złożeniu z frag przeszkadza (zabiera budżet słabszemu).
(3) Plateau przychodzi ostro po ~1.5 h — jak u doktora: „gdy krzywa stoi, kup pomiar, nie pomysł".

**Decyzje.** Dziennik co 12 h (Sebastian). Liczenie autonomiczne: łańcuch nohup na gpu-node (chain_night3b.sh), sesja
agenta tylko obserwuje. Po grze cyklu 1 (~10:30) bandyta stop; wchodzą wersje 2 operatora: kopnięcia 100/300, F=256
(okno Or-opt 64), Or-3opt w zejściu (wymiana dwóch segmentów), kopnięcie z odwróceniem (zmienia siatkę kary).

**Wpadki.** Budżet bramki ograniczał tylko ILS — z frag zbieżność trwała godzinami; teraz --ils-time = budżet ścienny
(0e580b2). Rollout trwa ~13 min zamiast 5 (minimum 30 s na fazę + alfa + JIT) — cykl bandyty 5 h zamiast 2.5 h.
Bandyta padł raz na PYTHONPATH. Piąty self-kill pgrep.

**Stan:** od zera **1 541 244** · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.
Luka do doktora: 12.2k (0.8%), do wzorca LKH-z-karą 25k.

## 22.09.2026, 20:13 (T+43 h) · wpis cyklu 7 · dzień 2: xswap2, GPX, hipoteza kary, decyzja „bez serca LKH"

**Liczby dnia.** Rano 1 541 244. play2 (4 h, LK+LS+frag100+xswap2 D=20): **1 538 779**. play3 (relaks 0.03 + pełna
kara): **1 538 476**. Rodzeństwo sib5/sib6 od play2 (108 min równolegle): 1 538 727 / 1 538 735 — za blisko siebie;
GPX(play3, s5, s6) = **1 538 458** (−17). play6 (ILS naprzemienny z karą 0) w toku: po 64 min 1 538 405, jeszcze w fazie
naprawy. Luka do doktora: 9.4k (0.62%); do wzorca LKH z karą 21.6k. Rozkład: nasz plain 1 527 774 vs LKH 1 503 070
(+24.7k), nasza kara 10.7k vs LKH 13.7k (−3k) — luka to długość, nie kara.

**E17 · Uogólniony xswap (xswap2, ab8f2ac).** Frag v2 (100 kopnięć, F=256, Or-3opt, kopnięcie z odwróceniem) na trasie
z plateau: <0.1 pkt/s — klasa optimów wyczerpana. Zamiast „więcej tego samego": wymiana miasta u z segmentu A za miasto v
z segmentu B, każde wstawione w NAJLEPSZE miejsce w oknie ±D drugiego segmentu (stary xswap: ta sama pozycja, ~10
kandydatów). Długości segmentów zachowane → siatka kary przesuwa się tylko w dwóch krótkich zakresach, delta O(D).
Na trasie 1 541 244: D=5 −350 w 3 s (237 kandydatów), D=10 −573, **D=20 −889** (654 kand., ocena 28 s), D=40 za wolno
(D³). K=8 vs 16 bez różnicy. Ruchy na dystansach 1 300–196 000 pozycji: niewidoczne dla LK w oknie, DP i frag.

**E18 · Krzyżowanie tras z dokładną karą (gpx.py, 9973895).** Segment A[i..j] o tym samym zbiorze miast i tych samych
końcach co segment B → podmiana nie rusza pozycji poza segmentem, kara dokładna O(długości); dziecko bierze lepszą wersję
każdego segmentu, gwarancja ≤ rodzic. Test na 3 trasach G (rodzeństwo z bramek): 44+25 podmian z ~400k kandydatów,
dziecko −197. Wpadka: pierwsza wersja bez filtra segmentów identycznych = O(n·maxlen) kandydatów, 10 GB RAM, ubita po
PID. Lekcja z sib5/sib6: GPX potrzebuje rodziców z osobnych linii, nie z tego samego startu po 100 min.

**Hipoteza Sebastiana (19:00): „kara trzyma nas w minimach, liczyć plain, karę włączyć na końcu".** Doktor w karuzeli:
zadziałało NAPRZEMIENNE szukanie bez kary z uczciwą wyceną (best zawsze po pełnej cenie), „im bliżej pełnej ceny, tym
mniejszy zysk"; NIE zadziałał jeden długi bieg bez kary z naprawą na końcu. Nasz ILS jest naprzemienny, ale z ulgą 0.03
(najmniejszą). play5z (długi plain, `--relax-current`) ubity po 5 min jako wariant negatywny u doktora; **play6 = ILS
z karą 0** (4 h od 19:09) — brakujący pomiar poziomu ulgi.

**Decyzja Sebastiana (19:42): świadomie bez metod LKH i jego „serca"** (π subgradient / alfa Helda–Karpa, 5-opt
sekwencyjny) do 5–6 dnia, żeby porównanie z doktorem było uczciwe. Kolejność: kopnięcia celowane, potem linie pod GPX.

**E19 · Kopnięcia celowane (8200f0c).** Zło krawędzi = nadwyżka nad odległością do 1-NN + kara; okna do kopania
losowane ∝ zło okna, cięcia double-bridge ∝ zło krawędzi (`--kick bad --kick-frac`). Bramka w nocy.

**Łańcuch nocy 4 (autonomiczny, po play6):** kara 0 lub 0.03 wg wyniku play6 → bramka kopnięć (uniform / bad 0.5 /
bad 1.0, 10 min × 3) → 3 linie od tras G po 3 h ze zwycięskimi kopnięciami → GPX(najlepsza, linie) → play7 4 h.
Koniec ~14:00 23.09.

**Wpadki.** Szósty self-kill (nazwa skryptu w komendzie z kill → reguła: komenda z kill = tylko PID-y). Budżet ścienny
sprawdzany po iteracji Combo: przy 2 biegach naraz iteracja 15 min, sib5/6 przekroczyły 75 min o 33. Panel pokazuje
czas fazy, nie biegu (do poprawy przy restarcie).

**Stan:** od zera **1 538 458** (gpx1) · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

## 23.09.2026, 08:13 (T+55 h) · wpis cyklu 8 · noc 4: ulga w karze naprawdę włączona, −9k w 3 h

**Główna lekcja dnia: „tryb naprzemienny" nigdy u nas nie działał.** Bramka nocna pokazała P0_uniform (kara 0) identyczne
co do bitu z K_uniform (0.03): run.py nie przekazywał `pen` do ILS, więc ILS zawsze szukał z pełną karą; ulga działała
tylko w krótkiej fazie relax. Wszystkie wcześniejsze wnioski „relaksacja daje mało" mierzyły inną rzecz. Po fixie
(f07a6a6) drugi błąd: kara 0 w ILS = 0 zysku, bo 118/197 okien wychodziło gorzej po prawdziwej cenie i nie dało się ich
cofnąć (LK bez kary miesza miasta między oknami) → bezpiecznik odrzucał całe iteracje. Brakowało kroku „naprawy": szukaj
jakby kary nie było → wróć do pełnej ceny (LS/LK z karą 0.1) → dopiero potem uczciwa ocena okien (ea9b195). To jest pełne
zdanie doktora: „przez chwilę szuka, jakby kary nie było, a potem wraca do pełnej ceny".

**Bramka nocy 4 (od gpx1 1 538 458, bez frag, 10 min × 3 seedy, ILS jako główna faza):**

| ramię | mediana | zysk | rozrzut |
|---|---|---|---|
| pełna kara w ILS (stan sprzed fixu) | 1 538 390 | −68 | 74 |
| K_uniform: ulga 0.03 + naprawa | 1 538 165 | −293 | 106 |
| **P0_uniform: ulga 0 + naprawa** | **1 537 936** | **−523** | 76 |
| K_bad05: celowane 50% okien (0.03) | 1 538 195 | −263 | 17 |
| K_bad10: celowane 100% (0.03) | 1 538 167 | −292 | 22 |

Wzór doktora („trzy poziomy ulgi, im bliżej pełnej ceny, tym mniejszy zysk") powtórzony co do kierunku: 0 > 0.03 > 0.1.
Kopnięcia celowane (E19): ten sam zysk, rozrzut 5× mniejszy — stabilizują, nie przyspieszają. Hipoteza Sebastiana
(„kara trzyma nas w minimach") potwierdzona w wersji naprzemiennej, nie w wersji „długi bieg plain + naprawa na końcu".

**Linie od G (kara 0, naprawa, frag100, xswap2 D=20).** lin_s1 od 1 545 878 po 2 h 55 min: **1 536 727** (−9 151),
6 iteracji ILS po ~25 min (frag100 ×2 rundy ×2 fazy). Start gorszy o 7.4k od dotychczasowego rekordu, wynik lepszy o
1.2k — dźwignia jest w procedurze, nie w trasie startowej. Luka do doktora: 7.7k (0.50%); do wzorca LKH z karą 20.5k.

**Co dalej (łańcuch autonomiczny):** lin_s2, lin_s3 (po 3 h) → GPX(najlepsza, 3 linie) → play7 4 h (~18:30). Potem:
ILS-iteracja jest za ciężka (25 min) — frag100 w obu fazach; do zmierzenia frag tylko w naprawie, lżejszy frag (30) i
`--ils-rounds 1`, żeby ILS robił 3–5× więcej iteracji na godzinę.

**Wpadki.** Trzy restarty bramki (bez frag; fix pen; fix naprawy) — łącznie ~2 h GPU; panel pokazywał setki starych
biegów i nie parsował frag/xswap2 (naprawione, 9c3c462: 24 h, żywe na górze, kolumna start).

**Stan:** od zera **1 536 727** (lin_s1, w toku) · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

## 23.09.2026, 20:13 (T+67 h) · wpis cyklu 9 · dzień 3: podgląd trasy zmienia wszystko

**Liczby dnia.** Rano 1 536 421 (lin_s1). Wieczorem **1 531 947** (−4.5k). Luka do doktora: 2.9k (0.19%); do wzorca LKH
z karą 15.7k. Przepis od zera (`RECIPE.md` + `scripts/santa_recipe.sh`) — polecenie Sebastiana: na koniec gotowa
sekwencja 24–48 h bez bandyty.

**Rano: linie i GPX.** lin_s2 1 536 242, lin_s3 1 536 785 (3 h każda od tras G), GPX(play3, 3 linie) 1 536 112 (−131,
131 podmian); play7 od dziecka 4 h → 1 535 227 w logu, ale plik best miał 1 536 074: BUG zapisu (best nadpisywany gorszą
trasą po kopnięciach) — naprawiony (199ef61). Pomysł Sebastiana „kara ujemna w fazie ulgi": bramka równolegle z linią
= 1 iteracja/10 min, zero rozdzielczości (ubita); wersja 2 w kolejce nocy.

**Po południu: Sebastian ogląda trasę (E20 · tour_view.py, :8903).** Pełna trasa na canvasie, kolor = pozycja w trasie,
czerwone kroki karne, żółte krawędzie ≥ k× odległość do 1-NN. Pierwsze spojrzenie: **jedna krawędź 3 525** (mediana 6.6)
przez pół mapy — szew Hilberta (ścieżka z rogu do rogu zamknięta przez miasto 0), nieusuwalny lokalnie (głęboki LK bez
okna: 0). Koszt 0.23% = pół ówczesnej luki. Sebastian: krzywa musi uwzględniać gęstość → warianty startu:

| start | wynik | max krawędź |
|---|---|---|
| Hilbert | 1 842 660 | 5 050 |
| Moore (zamknięty Hilbert) | 1 835 476 | 324 |
| Moore z medianami (gęstość) | 1 979 000+ | 354+ |
| **greedy-edge + 2-opt długich + uncross** | **1 679 690** | 404 |

**E21 · uncross (operator Sebastiana z jego wersji na aparapi).** 2-opt rozplątujący skrzyżowania długich krawędzi z
dowolnie odległymi w trasie (kd-tree wzdłuż odcinka). Greedy −26k w 86 s; trasa po pełnym pipeline −869 w 3 s.
**E22 · orlong (Sebastian: „wszyć filament w region sąsiadów").** Or-opt segmentów 3…5000 bez limitu odległości, kara
w O(1) przez prefiksy po resztach mod 10 (398/398 exact), 12 rdzeni: rekord −1 744 w 37 s. Filamentów greedy nie
rozpuszcza (7981→7856): to artefakt łączenia fragmentów, lek u źródła (start gmoore: fragmenty wzdłuż Moore — test nocą).

**Wieczór: „skrzyżowania zieleni z fioletem" (Sebastian z podglądu).** Greedy po 2.5 h ILS miało 1 789 skrzyżowań, a
uncross z pełną karą nie zdejmował żadnego: odwrócenie długiego odcinka przetasowuje siatkę kary. **Kryterium bez kary +
naprawa** (jak tryb ulgi doktora): 872 ruchy w 4 s, skrzyżowań 1789→11, po naprawie LS/LK **1 539 838 → 1 532 713**;
ta sama ścieżka na linii Hilberta: 1 533 419 → **1 531 947**. Kara fazy przekazana do uncross/orlong w ILS (6853039).

**Lekcje.** (1) Człowiek patrzący na obraz znalazł w 2 minuty to, czego 60 h operatorów lokalnych nie widziało: szew,
filamenty, skrzyżowania. (2) Każdy operator „daleki" musi liczyć bez kary i naprawiać po — pełna kara blokuje ruchy
przez siatkę, nie przez geometrię. (3) Greedy po LS/LK w 8 min = 1 553 012 vs Hilbert 1 576 303 po godzinie; ale po
ILS obie linie zbiegają do tego samego poziomu — start decyduje o czasie, operatory dalekie o wyniku.

**Noc 5c (autonomicznie):** g3 = ILS ulga 0 3 h od 1 531 947 → post (orlong → uncross bez kary → naprawa) → gmoore_r1
(test startu) → bramka kary ujemnej v2 (0/−0.03/−0.10/0.03, 20 min × 3) → linia Moore → GPX warunkowo. Animacja:
`frames.py` (33 klatki rekordów + co godzinę), `santa.mp4` na :8903; `best.html` co 10 min.

**Wpadki.** Bug zapisu best (−850 stracone); bramka ujemna bez rozdzielczości; bug LS po dużych zmianach z zewnątrz
(partia z dodatnim zyskiem pogarsza wynik — obejście: cofnięcie partii, do wyjaśnienia); 3 iteracje pętli best_view
(syntetyk, plik LKH); pierwszy GPX bez filtra = 10 GB RAM; panel: „liczy" z listy procesów (ILS loguje raz na 25 min).

**Stan:** od zera **1 531 947** · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 529 000 / 1 516 256 / 1 512 000.

## 24.09.2026, 08:13 (T+79 h) · wpis cyklu 10 · noc 5: poniżej doktora

**Wynik nocy: 1 528 697 od zera, bez LKH, po 3 dniach — poniżej 1 529 000 doktora (9 dni).** Ścieżka: greedy_r2
(1 539 639 na plateau, ubity) → uncross bez kary + naprawa (1 532 713) → ILS ulga 0 z uncross/orlong w fazie ulgi, 3 h
(gg3) → **1 528 697** → post 1 528 710. Linia Hilberta tą samą procedurą: g3 1 530 532 → 1 530 522. Luka do wzorca
LKH-z-karą (1 516 256): 12.5k (0.8%); do czołówki Kaggle 16.7k. Kara na rekordzie 11.7k (LKH: 13.7k), plain 1 517 033
(LKH plain 1 503 070): cała luka to długość, nie kara.

| linia | start | po LS/LK | po ILS ulga 0 + post |
|---|---|---|---|
| Hilbert (dzień 1–3) | 1 842 660 | 1 576 303 (1 h) | 1 530 522 |
| greedy + 2-opt długich + uncross | 1 679 690 | 1 553 012 (8 min) | **1 528 710** |
| gmoore (fragmenty greedy wzdłuż Moore) | 1 806 383 | **1 543 388** (1 h) | w kolejce |

**Co się nie udało w nocy.** (1) g3 zniknął bez tracebacku po 2 h: cichy segfault w numba (bez bounds-check) — orlong
czytał `R[n]` przy i+L==n; fix, wznowienie od zapisanego rekordu, strata 50 min. (2) Bramka kary ujemnej v2 (0 / −0.03 /
−0.10 / 0.03, 20 min × 3 od 1 528 710): wszystkie ramiona 1 528 709 bit w bit — trasa w plateau, ILS bez frag nic nie
robi w 20 min; pomiar bez rozstrzygnięcia, do powtórki z frag od trasy sprzed plateau. (3) GPX Hilbert × greedy: 0 podmian
z ~250 kandydatów — trasy z różnych startów nie mają wspólnych segmentów; GPX działa tylko w obrębie linii.

**E23 · OX (Sebastian, 06:46): „zamiast GPX krzyżuj OX, na najlepszym dziecku pełna kombinacja".** Order crossover:
segment A + reszta w kolejności B; dzieci mają dziury tam, gdzie B przechodził przez region segmentu (surowe +25…45k).
Tania naprawa CPU (uncross bez kary → orlong → uncross) do wyboru dziecka: 12 dzieci, segment 30/50/70% trasy → najlepsze
**segment 70%: 1 537 418** (50%: 1 538 074, 30%: 1 540 068). Od 07:31 bieg ox1 (LS/LK + ILS ulga 0, 3 h); test:
czy dziecko dwóch linii przebije 1 528 710. Podgląd: :8903/ox_kid.html.

**Lekcje dnia 3→4.** Start decyduje o czasie kroku 1 (greedy 8 min vs Hilbert 1 h), ale o wyniku decydują operatory
dalekie liczone bez kary z naprawą (uncross, orlong); trasy z różnych startów zbiegają do podobnego poziomu (1 528.7k vs
1 530.5k). Przepis (`RECIPE.md`): krok 1 = gmoore lub greedy (do potwierdzenia kroku 2), post-processing po każdym
kroku, kara 0 w fazie szukania.

**Co dalej (łańcuch):** ox1 → post → linia Moore (r1/r2 + post) ~17:00. Potem: bramka „lżejsza iteracja" (frag tylko w
naprawie / frag30 / 1 runda), kara ujemna z frag, dryf LS po dużych zmianach (obejście działa, przyczyna nieznana).

**Stan:** od zera **1 528 710** · doktor 1 529 000 ✔ · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 516 256 / 1 512 000.

## 24.09.2026, 20:13 (T+91 h) · wpis cyklu 11 · dzień 4: starty, Grecja, or-opt w łańcuchu

**Rekord Santa: 1 528 685** (ox1p, rano) — bez zmiany w ciągu dnia; dzień poszedł na starty, drugą instancję i nowy krok
w LK. Rano decyzja Sebastiana: **nie wdrażamy LKH** (π/alfa Helda–Karpa, 5-opt) — cel: zbliżyć się do rekordu bez jego serca.

**Starty (Sebastian: „bliskie sąsiedztwo punktów = blisko na ścieżce", „strategia przejścia przez długie krawędzie na
starcie, potem każdy segment niezależnie").** Cztery konstrukcje, wszystkie po tym samym LS/LK (krok 1) i 4 h ILS ulga 0:

| start | surowy | po LS/LK | po ILS 4 h + post |
|---|---|---|---|
| Hilbert | 1 842 660 | 1 576 303 | 1 530 522 |
| greedy + 2-opt długich + uncross | 1 679 690 | 1 553 012 | **1 528 710** |
| komórki (partycja Karpa, C=1024) | 1 576 699 | 1 552 096 | 1 535 860 (2.5 h, ubity) |
| gmoore (fragmenty greedy wzdłuż Moore) | 1 806 383 | 1 543 388 | — |
| **lądy v3** (regiony z gęstości, greedy w regionach, makro-trasa) | 1 710 405 | **1 542 833** | 1 530 774 |

Lekcja: start decyduje o kroku 1 (lądy najlepsze), ale po ILS wszystkie zbiegają do 1 528.7–1 531k. Filamenty greedy nie
przeszkadzają ILS z operatorami dalekimi. OX (E23) dziecko dwóch linii: 1 528 687 — poziom rodziców, nie przebicie.

**gr9882 (Grecja, 9 882 miast, optimum 300 899; stary solver Sebastiana 311 626 = 3.57%).** Zadanie na boku, kod bez
zmian (`--no-penalty`). Ścieżka: greedy 341k → LS/LK 308k → ILS 30 min 305 915 → 2 h (z frag) 304 882 → 1 h bez frag
304 335 → 1 h LK W2000 **303 930 (1.01%)**. Lekcje: (1) frag na małej instancji hamuje (154 bloki, sekwencyjne
kopnięcia; 43 iteracje/2 h), bez frag 234 iter/h; (2) LK z oknem = cała trasa i 16×8×4 to 17 s/przebieg, W2000 8×5×3 —
2 s bez straty; (3) prognoza 0.3–0.7% nie trafiona: sufit tej klasy ruchów ≈1%.
Operatory z podglądu Grecji (Sebastian): E24 **excursion_merge** (dwie wyprawy na wyspę → jedna; pełny przegląd k w numba,
kandydaci KNN przegapiali ruch +459); E25 **macro-TSP** nad segmentami (kolejność lokalnie optymalna: 0); E26 **lądy v1/v2/v3**
(v2 porty + DP; v3 greedy w regionach: Grecja po LS/LK 308 480 vs greedy 308 118 — remis).

**E27 · Or-opt w łańcuchu LK (wieczór; Sebastian: „pisz i działaj").** Dotąd łańcuch = same odwrócenia. Dodane kroki
„segment 1–3 za t3" w dwóch wariantach, wybór typu kroku po dokładnym zysku z karą, typ w wyniku, apply z rotacjami.
Na plateau zwykłego LK (0 łańcuchów): Grecja 14 łańcuchów, Santa rekord **125 łańcuchów, 40/40 exact, −105 w 3 rundach**.
Bramka Grecja (10 min × 3 od 303 930): baza −75 (rozrzut 180, 30 iter) vs **or-opt −145 (rozrzut 11, 8 iter)** — wygrywa
mimo 4× mniej iteracji, prawie deterministycznie. Bramka Santa w toku (SO_base = +0 w 20 min bez frag: rekord jest w
twardym plateau; SO_oropt do ~21:15) → play8 4 h z or-opt + frag → post.

**Wpadki.** cell_r2 i land3_r2 liczyły przez 2 h z dwoma biegami greckimi na karcie (wolniej ~3×); makro-TSP i lądy
v1/v2 — trzy wersje bez zysku zanim v3 (greedy w regionach) wyrównała greedy; excursion z kandydatami KNN = 1 ruch,
dopiero pełny przegląd k znalazł +459.

**Stan:** od zera **1 528 685** · doktor 1 529 000 ✔ · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 516 256 / 1 512 000.
Grecja **303 930** (1.01%). Podglądy :8903: /best.html, /land3*.html, /gr9882*.html.

## 25.09.2026, 08:13 (T+103 h) · wpis cyklu 12 · noc 6: or-opt w LK daje rekord, jeden kod dla wszystkich instancji

**Santa: 1 528 685 → 1 527 422.** Bramka or-opt w łańcuchu LK (20 min × 3 od rekordu, bez frag): baza +0, or-opt −212
(rozrzut 101). play8 (4 h, or-opt w LK, frag100, szybkie ścieżki bez kary w fazie ulgi): **1 527 424**, post 1 527 422.
Luka do wzorca LKH z karą: 11.2k (0.74%); do doktora −1.6k. Wieczorem to jedyny od dwóch dni operator, który dawał zysk
na plateau bez zmiany startu — noc to potwierdziła.

**E28 · Szybka ścieżka bez kary (Sebastian: „czemu tak wolno na małych?").** Frag i LK liczyły koszt każdego kandydata
na odcinku O(c), bo kara zależy od pozycji. Przy karze 0 (TSPLIB i faza ulgi Santa) delta 2-opt/Or-opt to 4–6 odległości:
frag 17× szybciej (15 s → 0.9 s), LK z or-opt 18× (16.2 s → 0.9 s), zyski identyczne co do bitu, ścieżka z karą nietknięta.

**E29 · Jeden kod dla każdej instancji (`solve.py`, polecenie Sebastiana).** Loader TSPLIB (EUC_2D/ATT/GEO), Santa CSV,
parametry i budżet z n, raport w metryce TSPLIB z luką do optimum (tabela z README TspSolver). Pierwsze testy:
berlin52/kroA100 optimum, ale pr1002 fragmentowo 3.11% w 7 min — na 1000 miast iteracja ILS trwała 18–50 s (DP k=16:
16 s niezależnie od n; or-opt LK O(W)). **Decyzja Sebastiana (21:07): < ~1500–2000 miast populacja + OX, duże →
fragmenty.** W memetic-framework jest gotowy populacyjny memetyk GPU (`solve_ga`: 8192 osobników, kolonie, OX, tabu,
drabina operatorów) — wpięty jako tryb dla n ≤ próg + dogrywka LK/or-opt (przyjmowana tylko, gdy lepsza w metryce TSPLIB).
pr1002: populacja 259 646 → dogrywka **259 574 (0.20%)** vs fragmenty 3.11%, README 1.19%.
rl1889 (Sebastian: „puść oba"): populacja 318 038 → **317 552 (0.32%)** w 46 min (karta dzielona z play8) vs fragmenty
323 714 (2.27%), README 1.96%.

**Przelot nocny po instancjach TspSolver (od 01:29, po play8):** 31 instancji populacyjnie (52–1889 miast) —
**wszystkie ≤ 0.95%**, 12 na optimum lub < 0.03%; mediana ~0.16%; czas 0.5–36 min. Najsłabsze: nrw1379 0.94%, u1432
0.83%. Od 2103 tryb fragmentowy: d2103 1.53%, u2152 2.72% → przelot zatrzymany.
**Dlaczego fragmenty słabe przy 2–5k (Sebastian: „zobacz czemu"):** na trasie po LS/LK frag daje 0 nawet przy 2000
bloków × 100 kopnięć (optimum 128-fragmentów wyczerpane), a kopnięcia ILS w oknach przy czystej długości LK cofa
natychmiast (12 iteracji, 17 okien, 0 okien gorszych, 0 poprawy). Na Santa ten schemat działa, bo kara „przykleja"
kopnięcia. Bez kary brakuje perturbacji — tę rolę pełni OX w populacji. **Próg populacji → 5000**; `sweep2` (2152–5934)
w kolejce po d2103 populacyjnie (w toku).

**Grecja (gr9882) — zadanie na boku:** frag hamuje (154 bloki), LK z oknem = cała trasa 17 s/przebieg → W2000: bieg d
**303 930 (1.01%)**, bramka or-opt −145 (rozrzut 11) vs baza −75 (rozrzut 180). Stary solver Sebastiana: 3.57%.

**Wpadki.** Raport TSPLIB zaokrąglał `rint` (połówki do parzystych) zamiast `floor(x+0.5)` → d493 „35 001 < optimum"
(poprawnie 35 005); memetyk liczył dobrze. Kopie fragmentów (`--frag-copies`): 3× zysk za 13× czasu — zostają opcją.
u724 i nrw1379 to instancje, gdzie populacja potrzebowała najdłużej (12–28 min).

**Stan:** od zera **1 527 422** · doktor 1 529 000 ✔ · LKH+my 1 516 106 · LKH plain 1 503 070 · cele 1 516 256 / 1 512 000.
TSPLIB: 31/31 ≤ 0.95% (populacja), gr9882 1.01% (fragmenty). Przepis: `solve.py` = jeden punkt wejścia.

## 25.09.2026, 20:13 (T+115 h) · wpis cyklu 13 · dzień 5: 41 instancji jednym kodem, przepis od zera działa

**Przelot TSPLIB zakończony (`docs/santa/TSPLIB_RESULTS.md`).** 41 instancji (52–9882 miast) jednym wejściem `solve.py`,
parametry i budżet z n: mediana luki **0.198%**, 16 instancji ≤ 0.1% (12 na optimum), 35 ≤ 1%, max 1.89% (fnl4461).
Na każdej instancji z README wynik lepszy od starego solvera Sebastiana, zwykle kilkukrotnie (pr1002 0.20% vs 1.19%,
pcb1173 0.16% vs 2.05%, rl1889 0.32% vs 1.96%, gr9882 1.01% vs 3.57%).

| strefa | tryb | luki |
|---|---|---|
| ≤ 2000 miast (31 inst.) | populacja 8192 + OX + dogrywka LK/or-opt | 0.00–0.94%, mediana ~0.16% |
| 2000–5000 (7 inst.) | populacja 4096/2048 | 0.49–1.89% (rośnie z n) |
| ≥ 5000 (3 inst.) | fragmenty (greedy → LS/LK → ILS ulga 0) | 1.01–1.74% |

**Dlaczego fragmenty słabe przy 2–5k (Sebastian: „zobacz czemu").** Na trasie po LS/LK frag daje 0 nawet przy 2000
bloków × 100 kopnięć; kopnięcia ILS w oknach przy czystej długości LK cofa natychmiast (12 iteracji, 17 okien, 0 okien
gorszych, 0 poprawy). Na Santa ten schemat działa, bo kara „przykleja" kopnięcia. Bez kary perturbacją jest OX w populacji
→ próg populacji 5000. Strefa 2–5k słaba z obu stron: populacja od góry (fnl4461 2.7 h, 1.89%), fragmenty od dołu
(d2103 1.53%). Kandydat: LK z or-opt jako zejście lokalne wewnątrz osobnika populacji.

**Przepis od zera na Santa (`recipe_fast`, pomiar odtwarzalności).** greedy → LS/LK z or-opt w łańcuchu: **1 536 885
w 46 min** (bez or-opt ten krok dawał 1 553 012 — or-opt przesuwa optimum LS/LK o 16k) → ILS ulga 0 (or-opt, frag100,
xswap2, orlong, uncross, wyprawy): po 13 iteracji / 3.5 h **1 527 942** — poniżej doktora (1 529 000) i o 500 od
rekordu (1 527 422, który kosztował 4 dni). Koniec kroku 2 ~20:05, post ~20:20. Przepis = ~5 h od zera, jeden skrypt.

**Lekcje dnia 5.** (1) Jeden kod, dwa tryby: decyzja Sebastiana „populacja + OX dla małych, fragmenty dla wielkich"
potwierdzona liczbami na 41 instancjach. (2) Szybkie ścieżki bez kary (17–18×) zmieniły ekonomię: krok 1 z or-opt jest
tańszy niż bez or-opt wczoraj. (3) Wynik od zera po 5 h ≈ wynik po 4 dniach: dźwignie są w procedurze i operatorach,
nie w czasie — dokładnie teza przepisu.

**Wpadki.** Zaokrąglanie raportu TSPLIB (rint → floor(x+0.5)); kopie fragmentów bez zysku na czas; fnl4461 populacyjnie
2.7 h (populacja 2048 za mała, ale fragmenty byłyby gorsze).

**Stan:** od zera **1 527 422** (przepis w toku: 1 527 942) · doktor 1 529 000 ✔ · LKH+my 1 516 106 · cele 1 516 256 / 1 512 000.

## 26.09.2026, 08:13 (T+127 h) · wpis cyklu 14 · dzień 6: hybryda dla strefy 2–5k, GPU noc bez biegu

**Santa: bez zmian, 1 527 422** (rekord) / przepis od zera 1 527 835 w 5 h. GPU stało od 23:00 (wieczór poszedł na
projekt poboczny: tłumacz głosowy PL↔EN dla Meet na tej samej karcie — dwie instancje WhisperLiveKit zajmują teraz
~4 GB VRAM na stałe; Santa ma nadal >14 GB). Od 08:15 rusza play9: ILS ulga 0 z or-opt 4 h od rekordu + post.

**E30 · Hybryda populacja + LK/or-opt (Sebastian: „rób LK z or-opt dla 2–5k").** W pętli chunków memetyka
(`solve_ga`) najlepsze wyspy (16–32) dostają po każdym chunku zejście naszym LK z or-opt + LS + uncross (kara 0) i wracają
do populacji (najsłabszy slot wyspy, aktualizacja rekordu wyspy). d2103: 0.479% w 28 min vs 0.512% w 34 min (populacja);
pcb3038 (32 wyspy): **1.368% vs 1.451%**. Działa, ale skromnie: elita memetyka jest już blisko optimum LK (0–3 z 16 wysp
poprawianych na chunk; raz 16 wysp po scaleniu, +269). Wniosek: nasz LK najwięcej daje na świeżo zaburzonych trasach,
nie na elicie → następny wariant: polerowanie dzieci po OX w kernelu, nie elity z hosta.

**Tabela TSPLIB (`TSPLIB_RESULTS.md`, 41 instancji):** mediana 0.198%, 16 ≤ 0.1%, 35 ≤ 1%; strefa 2–5k 0.5–1.9%
(fnl4461 1.89% w 2.7 h — populacja 2048 za mała), ≥5k fragmentowo 1.0–1.7%.

**Wpadki.** Brak biegu na Santa przez 9 h (kolejka nie miała „następnego kroku" po recipe_fast — poprawione: play9).

**Stan:** od zera **1 527 422** · przepis 5 h 1 527 835 · doktor 1 529 000 ✔ · LKH+my 1 516 106 · cele 1 516 256 / 1 512 000.

## 26.09.2026, 20:13 (T+139 h) · wpis cyklu 15 · dzień 6 wieczór: 1 527 000 pęka, hybryda rozstrzygnięta

**Santa: play9 1 527 102 → play10 1 527 022 → play11 (w toku, po 8 iteracjach) 1 526 924.** Granica 1 527 000 (2 000
poniżej doktora) przekroczona w trakcie biegu; zapis po DONE ~21:20 i post ~21:35. Tempo plateau z or-opt: 80–100 na
4-godzinną rundę. Rekord od zera po post: 1 527 022 (plik), 1 526 924 (live).

**E30/E31 · Hybryda populacja + LK/or-opt, bramka na d2103 (opt 80 450) zamknięta:**

| wariant | luka | czas |
|---|---|---|
| **hybryda best** (polerowanie rekordów wysp naszym LK/or-opt) | **0.479%** | 28 min |
| czysta populacja (solve_ga, OX, kolonie) | 0.512% | 34 min |
| hybryda kick (double-bridge + LK od rekordu wyspy, wstrzyknięcie) | 0.542% | 77 min |
| hybryda worst (polerowanie dzieci OX) | 0.685% | 49 min |
| fragmenty | 1.528% | 16 min |

Lekcja: wstrzykiwanie LK-optymalnych tras w środku ewolucji (kick, worst) obniża różnorodność, mimo że 29–32 z 32
tras wchodzi do wysp na chunk; pomaga tylko polerowanie elity, i to skromnie. W strefie 2–5k limitem jest sam memetyk
(populacja 4096, czas), nie brak siły lokalnej. `solve.py --hybrid` domyślnie best.

**Poboczne (ta sama karta):** tłumacz Meet — dwie instancje WLK 4.6 GB VRAM na stałe, orkiestrator działa; klient
laptopowy w testach u Sebastiana. Bielik v2.6 bnb-4bit odtworzony (6.15 GB VRAM, kod bielik-rng + `guess.py`: to samo
pytanie N× PRNG vs N× random.org); dane starego eksperymentu z 10.09 stracone z VM. Dysk: 55 → 38 GB wolne.

**Wpadki.** Pierwsza bramka kick ubita przez timeout 1 h (GPU dzielona z play10) — wynik stracony, powtórka 77 min.

**Jutro (27.09, dzień 7 = koniec wyzwania):** dziennik zamykający, repo z nowym README (obecne → branch legacy),
tabela wyników i czasów (Santa: dziennik + `sub_play*p_*.csv`; TSPLIB: `universal.jsonl` + `TSPLIB_RESULTS.md`),
animacja `santa.mp4`, materiał do karuzeli.

**Stan:** od zera **1 527 022** (post) / 1 526 924 (live) · doktor 1 529 000 ✔ (−2 076) · LKH+my 1 516 106 · cele 1 516 256 / 1 512 000.

## 27.09.2026, 08:13 (T+151 h) · wpis cyklu 16 · dzień 7: ostatnia noc, 1 526 259 na żywo

**Noc:** play11 → post **1 526 850**; play12 (6 h, okno LK 5000 zamiast 3000) → **1 526 460**, post **1 526 454**;
play13 (4 h, W5000, seed 113) po 12 iteracjach **1 526 259** (live), DONE ~08:30 + post. Szersze okno LK na plateau
daje ~4× większe tempo (−390/6 h vs −80…−100/4 h przy 3000): łańcuchy z or-opt, którym brakowało zasięgu.
Rekord od zera po post: **1 526 454** → −2 546 vs doktor (1 529 000); luka do wzorca LKH z karą 10.2k (0.67%).

**Krzywa całego wyzwania (od zera, bez LKH):** Hilbert 1 842 660 → LS 1 590 816 (d1) → LS+LK 1 576 303 → ILS 1 566 117
→ chained-LK 1 550 414 → bramki E/G 1 545 839 (d2) → frag 1 541 244 → xswap2/orlong 1 538 779 → uncross bez kary
1 532 713 → ulga 0 + naprawa 1 528 697 (d4, poniżej doktora) → or-opt w LK 1 527 422 (d5) → W5000 **1 526 454** (d7).
Przepis od zera jednym skryptem: 5 h → 1 527 835.

**Wpadki nocy:** brak. **Co dalej (dziś, koniec wyzwania):** dziennik zamykający po play13p, repo z nowym README
(obecny stan → branch legacy), tabela wyników i czasów Santa + TSPLIB (41 instancji, mediana 0.198%), `santa.mp4`,
materiał do karuzeli LinkedIn.

**Stan:** od zera **1 526 454** (post) / 1 526 259 (live) · doktor 1 529 000 ✔ · LKH+my 1 516 106 · cele 1 516 256 / 1 512 000.

## 27.09.2026, 08:50 (T+152 h) · WPIS ZAMYKAJĄCY · 7 dni: 1 526 256

**Wynik końcowy od zera, bez LKH i jego serca: 1 526 256** (plain 1 514 940, kara 11 317; pełna walidacja: 197 769 miast,
permutacja poprawna, wynik przeliczony niezależnie). `docs/santa/submission_1526256.csv`.

| punkt odniesienia | wynik | my vs |
|---|---|---|
| doktor (LinkedIn, 9 dni) | 1 529 000 | **−2 744** |
| wzorzec doktora: LKH z karą | 1 516 256 | +10 000 (0.66%) |
| LKH plain (bez kary) | 1 503 070 | +11 870 plain |
| czołówka Kaggle 2018 | ~1 512 000 | +14 256 (0.94%) |
| LKH + nasze operatory (linia pomocnicza) | 1 516 106 | — |

**Co zadziałało (w kolejności zysku):** ulga 0 w fazie szukania z naprawą po (−9.5k na trasę, dzień 4); frag =
fragment 128 pozycji jako mini-ILS w bloku GPU (−4.6k w noc); uncross bez kary (skrzyżowania między odległymi częściami
trasy, −7k na greedy); or-opt w łańcuchu LK (jedyny operator dający zysk na twardym plateau, −1.3k + tempo);
okno LK 5000 (−0.6k ostatniej nocy); start greedy/lądy zamiast Hilberta (szew 3.5k). Co nie zadziałało: DP okien
dokładnych w złożeniu z frag, xswap równej długości, kara ujemna (bez rozstrzygnięcia), GPX między liniami, OX jako
przebicie, hybrydy kick/worst, makro-TSP nad segmentami.

**Czego się nauczyliśmy:** (1) człowiek patrzący na rysunek trasy znajdował w minuty to, czego operatory nie widziały
przez dni (szew, filamenty, skrzyżowania, wyprawy); (2) każdy operator „daleki" musi liczyć bez kary i naprawiać po —
pełna kara blokuje ruchy siatką, nie geometrią; (3) plateau = stop, bramki 10 min × 3 seedy, jedna zmienna naraz;
(4) wynik po 5 h przepisem ≈ wynik po 4 dniach: dźwignie są w procedurze; (5) na małych instancjach populacja + OX,
na wielkich fragmenty — jeden kod, 41 instancji TSPLIB, mediana luki 0.198%.

**Artefakty:** dziennik (16 cykli), `RECIPE.md` + `scripts/santa_recipe_fast.sh`, `solve.py` (jedno wejście),
`TSPLIB_RESULTS.md`, `universal.jsonl`, podglądy :8903 (`/final.html`), `santa.mp4` (klatki rekordów od dnia 1),
kod: gpu_ls/gpu_lk/gpu_frag/gpu_xswap2/gpu_dp/orlong/uncross/excursion/macro/landmass/celltour/gpx/ox/alpha/ils/bench/bandit.

## 27.09.2026, 20:13 · po zamknięciu · cykl dziennika wyłączony

Wyzwanie zamknięte rano (wpis zamykający 08:50, wynik 1 526 256). Od tego czasu GPU nie liczy nic dla Santa
(0% obciążenia; 9.6 GB VRAM to stałe usługi: tłumacz Meet, embedder, Scribe). Cykliczny dziennik co 12 h zakończony —
to ostatni automatyczny wpis. Otwarte na kolejną sesję: repo z nowym README (obecny stan → branch legacy), tabela
wyników i czasów, karuzela PDF na LinkedIn z 16 cykli dziennika i `santa.mp4`.
