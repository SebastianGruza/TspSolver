# Adaptacyjna eskalacja operatorów + discovery kolejności

Dokumentacja koncepcji: **jak** algorytm sam dobiera kolejność zwiększania „mocy" (operatory LS,
K, krzyżowanie, merge) w trakcie biegu, oraz **jak** zbieramy dane, by tę kolejność odkryć
statystycznie i docelowo przewidywać z cech instancji.

Status: [zaimpl.] = działa w kodzie, [plan] = zaprojektowane, do zbudowania po zebraniu danych.

---

## Problem

Stała kolejność operatorów **nie jest optymalna dla wszystkich instancji**. Zaobserwowaliśmy:
- 3-opt bije wszystko na dużej rl5934, ale na mniejszych **Or-2/relokacje wygrywają pierwsze**
  (3-opt jest drogi per epoka).
- pr2392 (duże odległości, wysokie cv) startuje najlepiej od **greedy-edge crossover**, nie od LS.

Wniosek: kolejność powinna być **warunkowana instancją**, a nie zgadywana ręcznie.

---

## 1. Drabina eskalacji  [zaimpl.]

Baza LS = 2-opt + Or-1 (tanie, szybki exploit). Do tego **atomowe ruchy**, każdy = DOKŁADNIE
jedna mała zmiana, aplikowana gdy poprzednia się wyczerpie:
- `K += Δ` (szersze sąsiedztwo KNN),
- `+operator` (Or-2, Or-3, 3-opt, long-edge, **swap**, **segvar** — patrz §6),
- `ox` (krzyżowanie: klasyk → mix greedy-edge → pełny greedy-edge),
- `merge` (globalne mieszanie kolonii, jednorazowe).

Reprezentacja: `ls_mode=9` + **bitmaska `ops`** (bit per operator). Kontroler przełącza K/ops/ox/merge
per chunk (CPU steruje, per-epoka sync zostaje na GPU).

**Dlaczego atomowo:** każdy ruch daje własny „burst" poprawy, potem nasyca się w ~1 chunku.
Duże skoki (cały tryb naraz) marnują potencjał — jeden ruch wnosi wszystko i wyczerpuje się.
**Kolejność ruchów DOMINUJE nad granularnością** — high-value first (np. 3-opt na dużych)
bije K-bumpy-first.

## 2. Wyzwalacz: deceleracja  [zaimpl.]

Kiedy zaaplikować następny ruch? Gdy bieżąca konfiguracja **zwalnia**:

```
r      = świeży spadek best w oknie decel_win chunków
r_norm = r / (szczytowe tempo TEGO kroku)
CLIMB gdy r_norm < decel_rho   (świeże tempo < ułamek szczytu kroku)
```

**Dlaczego nie surowe tempo / nie różnorodność:** metryki różnorodności (uniq%, cv%) *lagują*
za tempem — bramkowanie na nich opóźniało climb ~2×. Deceleracja (recent/peak) łapie moment
poprawnie. `best_tabu`/force jako anty-deadlock: jeśli krok nie daje ŻADNEJ poprawy przez `d_max`
chunków, i tak eskaluj.

## 3. Early-stop  [zaimpl.]

Bieg kończy się gdy **3 kolejne chunki bez poprawy best** (`noimp >= 3`), z górnym capem `ge`.
Powód: małe instancje zbiegają szybko (kończą tanio), a te z potencjałem (np. pr1002) jadą
**do zbieżności**, nie ucięte sztywnym budżetem. Rozwiązuje napięcie „szybko dla małych" vs
„nie ucinać potencjału".

## 4. Bandyta discovery (greedy rollout)  [zaimpl.]

Zamiast stałej listy ruchów — na każdej eskalacji **próbuj każdego kandydata** i zostaw najlepszy:

```
snapshot stanu device (R, rlen, gbest, gbest_route, age, RNG, gbstall)
dla każdego kandydata {stay, +operatory, K+Δ, ox, merge}:
    restore(snapshot)                      # KAŻDY startuje z tego samego stanu + RNG (fair)
    zastosuj kandydata; uruchom trial_chunks epok
    zmierz burst = poprawa best,  dt = czas
    reward = burst / dt                    # JAKOŚĆ / CZAS, nie sam burst
    zaloguj (stan, kandydat, burst, dt, reward, faza, cechy) do SQLite
commit zwycięzcy (max reward)
```

Kluczowe zasady:
- **reward = burst / czas** — operator dający duży burst ale drogi (3-opt, greedy-edge) może
  przegrać z tańszym o mniejszym-ale-szybszym burście. Zgodne z celem anytime.
- **null-action „stay"** — kandydat „nie zmieniaj nic". Jeśli wygrywa, config zostaje: tani config
  bazowy wciąż produktywny → nie eskaluj przedwcześnie. (W praktyce „stay" wygrywa najczęściej.)
- **snapshot RNG krytyczny** — inaczej różnica burstów nie jest czysto efektem ruchu.
- Przegrane próby to nie strata — to **oznaczone dane** `(stan, akcja) → reward` (patrz §7-8).

Koszt: ~liczba_kandydatów × trial_chunks epok na eskalację. Discovery jest droższy, ale to
**jednorazowe zbieranie danych** — produkcja użyje stałej/przewidzianej kolejności (§8).

## 5. Tabu inkumbenta (anty-zakotwiczenie)  [zaimpl.]

Najlepszy osobnik w populacji, mając najniższą długość, prawie nigdy nie jest wypierany →
**kotwiczy populację**. Fix:
- `gbstall[gid]` = epoki odkąd best wyspy się nie poprawił (reset przy poprawie).
- Gdy `gbstall > best_tabu` (=5, 1 chunk), inkumbent (`rlen == gbest`) dostaje **rosnącą karę**
  `(len//250)·min(gbstall−5, 16)` w selekcji → wchodzi na „najgorszego", zostaje wyparty →
  populacja eksploruje dalej.
- **Prawdziwy best żyje osobno w `gbest_route`** z właściwą długością (immune) — karanie w
  populacji nie grozi utratą rozwiązania.

Komplementarne do dwóch istniejących kar tabu: wieku (`age >= tabu_age`) i duplikatu długości.

## 6. Operatory  [zaimpl.]

LS (KNN-guided, O(n·K)/sweep): 2-opt, Or-1/2/3 (relokacja 1/2/3 miast), 3-opt, long-edge
(naprawa ~15 najdłuższych krawędzi). Krzyżowanie: OX klasyk / OX-longedge / mix / **greedy-edge**
(edge-preserving, najkrótsza krawędź rodzicielska + KNN fallback).

**Zaadaptowane z oryginału Java/Aparapi:**
- **`swap`** — vertex swap CELOWANY w długie krawędzie: miasto z końca jednej z top-10 najdłuższych
  krawędzi zamieniane pozycją z bliskim sąsiadem (KNN). Zmienia 4 krawędzie, bez odwracania.
- **`segvar`** — relokacja DŁUŻSZYCH segmentów (len 5 i 10), pokrywa lukę medium/long ponad Or-2/3.

Adopcja w bandycie jest trywialna: każdy operator = **jeden bit + jeden kandydat**, a bandyta sam
odkrywa jego wartość i timing per instancja (segvar wygrywał step 0 na d2103, swap step 4 na pr1002).

## 7. Schemat bazy  [zaimpl.]  — co i po co

Discovery loguje do SQLite (`results/ladder_trials_*.db`) dwie tabele.

**`ladder_trials`** — jeden wiersz = jedna próba kandydata (per decyzja):

| grupa | kolumny | po co |
|---|---|---|
| stan | `state_ops, state_K, state_ox` | **kompletny stan Markowowski** (co jest włączone) |
| faza | `phase_uniq, phase_cv, phase_gbstall, phase_gbstall_max` | obserwowana „ukryta" faza szukania + tabu-pressure |
| czas | `cum_t` | czas skumulowany → cel **anytime-AUC** dla DP |
| cechy krawędzi | `feat_nn, feat_cv, feat_skew, feat_kurt, feat_far` | rozkład odległości (KNN) |
| cechy przestrzenne | `feat_clark, feat_gridcv, feat_aspect, feat_tight` | struktura punktów (skupienie/heterogeniczność/wydłużenie) |
| akcja+reward | `candidate, burst, dt, reward, best` | ruch i jego wynik (jakość/czas) |
| meta | `instance, n, seed, step` | identyfikacja / wielo-seed dla wariancji |

**`ladder_runs`** — jeden wiersz = jedna trajektoria (per bieg): `terminated_by` (early_stop/cap),
`final_step, final_best, total_time` + cechy. **Kluczowe dla credit assignment**: trajektoria
zamknięta early-stopem osiągnęła zbieżność; ta na capie mogła mieć jeszcze potencjał — DP musi
to rozróżniać.

Cechy przestrzenne (numpy, bez sklearn): **Clark-Evans R** (mean-NN / oczekiwane przy losowości:
<1 skupione, ≈1 losowe, >1 rozproszone), **grid-CV** (CV liczby punktów w siatce = heterogeniczność),
**aspect** (wydłużenie chmury), **tight** (frakcja NN < ½·mean = ciasne skupienia).

## 8. Model docelowy: MDP + DP + MLP  [plan]

Wybór następnego stanu to **sekwencyjny problem decyzyjny = MDP**, NIE łańcuch Markowa
(ten nie ma akcji/sterowania) ani HMM (u nas stan jest w pełni **obserwowany** — nic ukrytego;
fazę obserwujemy przez metryki różnorodności, §7).

**Struktura daje skrót:** stan jest **monotoniczny** (`ops` tylko rośnie, `K` tylko rośnie) →
graf stanów to **DAG** → MDP rozwiązuje się **DOKŁADNIE dynamicznym programowaniem**
(backward induction / najdłuższa ścieżka), bez przybliżeń RL. `V(stan) = max_akcja [reward + γ·V(następny)]`.
To bije greedy — DP widzi downstream (np. „Or-2 odblokowuje 3-opt").

**Warunkowanie na instancji:** regresor `reward ~ f(stan, faza, cechy, akcja)`:
- **Mały MLP** (2-3 warstwy) — tam gdzie danych DUŻO (tysiące wierszy per-decyzja). Nieliniowe
  interakcje `cecha×stan×akcja`. **To regresja nadzorowana, nie RL** (transitions deterministyczne),
  potem DP propaguje przez przewidziany reward.
- **NIE** uczony enkoder chmury punktów (Deep Sets/PointNet) na tym etapie — 36 instancji to za mało,
  przetrenuje się. Ręczne cechy przestrzenne (§7) to priory data-efficient. Enkoder to ścieżka na
  skalowanie (setki instancji).

Pipeline docelowy: **cechy (ręczne) → MLP reward(stan,akcja|cechy) → DP na DAG-u → optymalna
kolejność dla nowej instancji z jej cech, ZERO prób online**. Discovery (droższy) to jednorazowe
zebranie danych; produkcja jest darmowa.

---

## Narzędzia

- `disc_run.py <inst> [ge|0] [seed] [db]` — discovery na instancji (early-stop, log do bazy).
- `disc_agg.py` — agregat: statystycznie najlepsza kolejność (greedy path po medianie reward).
- `disc_peek.py [inst] [step] [db]` — podgląd DB, najświeższe na górze.
- `disc_web.py [port] [db] [log]` — serwer WWW na żywo (postęp + best vs Aparapi + wybory bandyty).
