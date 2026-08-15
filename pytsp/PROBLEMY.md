# Problemy, rozwiązania i efekty — adaptacyjna eskalacja + discovery

Dziennik problemów napotkanych przy budowie drabiny eskalacji i bandyty discovery, jak je
zaadresowaliśmy i **co to faktycznie dało** (z pomiarami). Uzupełnienie `DISCOVERY.md`
(który opisuje mechanizmy „jak są") — tu jest „czemu tak" i droga dojścia.

Motyw przewodni: **niemal każdy fix wyszedł z jednej podejrzanej liczby w podglądzie** (burst=0
tam gdzie nie powinno, wynik gorszy od Javy, za wczesny stop). Warto patrzeć na surowe dane —
kilka bugów było niewidocznych w „działa, daje wyniki".

---

## 1. Wyzwalacz eskalacji: różnorodność LAGuje za tempem

**Problem.** Pierwsza wersja bramkowała eskalację na kolapsie różnorodności (`uniq%`, `cv%`).
Intuicyjne — spadek różnorodności zapowiada zastój. Ale w praktyce climb rung0→rung1 odpalał
przy **ep65 zamiast ep30**.

**Diagnoza.** Metryki różnorodności są **opóźnione** względem tempa poprawy best. Zanim `uniq%`
spadnie poniżej progu, tempo best już dawno siadło. Bramka na różnorodności = climb ~2× za późno,
a rung1 zaraz po włączeniu dawał burst −8440 (którego nie łapaliśmy przez 30 epok).

**Fix.** Trigger na **deceleracji tempa best**: `r_norm = świeży_spadek / szczytowe_tempo_kroku`,
climb gdy `r_norm < decel_rho`. Różnorodność zeszła z roli bramki do roli **cechy** logowanej dla
modelu (obserwowana faza, nie warunek).

**Efekt.** Climb w właściwym momencie. Koncepcyjnie: **sygnał sterujący musi wyprzedzać, nie
opóźniać** — tempo best (lagging, ale bezpośrednie) pobiło różnorodność (leading, ale mylące bo
mierzy co innego).

## 2. Gruboziarnista drabina: duże skoki marnują potencjał

**Problem.** Każdy rung (duży skok configu) dawał **jeden produktywny chunk** (burst −8-10k),
potem od razu nasycał. Ruch wnosił wszystko naraz i się wyczerpywał.

**Fix.** **Atomowe ruchy** — dokładnie jedna mała zmiana na krok (K+Δ / +operator / ox / merge),
reprezentacja przez bitmaskę `ops` (`ls_mode=9`).

**Efekt + niespodzianka.** Sama granularność NIE wygrała — **kolejność ruchów dominuje**.
High-value-first (3-opt wcześnie) bije K-bumpy-first o **~12k jednostek w tym samym czasie**.
Wniosek: nie „ile małych zmian", tylko „w jakiej kolejności" — co uzasadniło całe discovery
(szukanie optymalnej kolejności per instancja).

## 3. Pomiar trialu: mismatch measure/act (1 epoka vs ~10)

**Problem.** Rollout mierzył burst na `trial_chunks=1` epoce, a zwycięzca po commitcie leci ~10
epok (chunk + dwell). Tempo **opada** (pierwsza epoka = szczyt), więc 1 epoka łapie PEAK i
**przeszacowuje operatory front-loaded** względem tych o równym, wolniej gasnącym tempie.

**Fix.** `trial_chunks=3` — okno reprezentatywne dla jednostki commitu.

**Efekt.** Ranking stabilniejszy: u2152 step0 zmienił zwycięzcę Or3→Or2 między trial=1 a trial=3
(trial=1 był mylący). Koncept: **mierz to, co dostaniesz** — okno pomiaru ma odpowiadać oknu akcji.

## 4. Reward musi uwzględniać CZAS

**Problem/decyzja.** Sam burst faworyzuje drogie operatory (3-opt, greedy-edge).

**Fix.** `reward = burst / dt` (jakość na sekundę).

**Efekt.** Time-aware zmienia decyzję w **22% kroków** (max-burst ≠ max-reward). Przykład pr2392:
Or-3 ma największy burst (4670) ale przegrywa z ox_mix (3275) bo **2× droższy**. To wprost cel
anytime — najlepszy wynik w każdym momencie, nie najgłębsza pojedyncza poprawa.

## 5. Null-action „stay"

**Problem.** Kandydaci to same zmiany — brak opcji „nie eskaluj". Bez tego bandyta forsował
operator nawet gdy tani config bazowy wciąż był produktywny.

**Fix.** Kandydat `stay` (bez zmian). Jak wygra → config zostaje.

**Efekt.** „stay" wygrywa najczęściej (blokuje przedwczesną eskalację), tani ruchy (stay+K+4)
dominują — potwierdza że time-aware systematycznie preferuje niski koszt. **Ale** to samo „stay"
później spowodowało problem (§9) — patrz tam.

## 6. Tabu inkumbenta: agresywna kara NISZCZY najlepszego

**Problem.** Najlepszy osobnik (najniższa długość) prawie nigdy nie jest wypierany → kotwiczy
populację. Moja pierwsza implementacja: **rosnąca kara addytywna** `(len//250)·min(gbstall−5,16)`
(do ×1.064) na inkumbenta.

**Diagnoza (z danych).** pr439: **wszystkie kandydaty burst=0** w step 0, `gbstall_max=24` (kara
na maksie). Wynik 107758 (0.505%) — **gorszy** niż bez tabu (107347, 0.121%). Mechanizm: kara
wypierała **jedynego** best-ever z populacji nawet bez zbieżności → populacja oddryfowywała →
operatory pracowały na gorszych osobnikach → nie potrafiły pobić best → **burst=0, zero sygnału
discovery** + gorsza jakość. `gbest_route` trzymał rekord, ale był biernym rekordem, nie w puli genów.

**Fix (próba 1 — reinjekcja):** wstrzykiwanie `gbest_route` z powrotem do populacji. Działało
(107317), ale złożone.

**Fix (próba 2 — Aparapi, wygrała):** prosta kara **×1.004** (jak w oryginale Java). **Samoregulacja**:
inkumbent staje się „najgorszym" tylko gdy cała populacja zbiegła do **<0.4% od best** (wtedy
potrząśnięcie pożądane — perturbacja z sąsiedztwa best → nowe besty). Przy różnorodności best
zostaje. Bez reinjekcji, bez agresji.

**Efekt.** pr439: 0.061% (×1.004) — bije wszystkie warianty. **Proste pobiło złożone.** Koncept:
kara diversyfikująca musi być **łagodna i samoregulująca** — silna kara na jedynego best usuwa
materiał genetyczny, na którym trzeba budować.

## 7. Migracja MARTWA w całej drabinie chunkowanej

**Problem (odkryty przez pytanie „jak merge w trialu?").** `do_mig = (ge+1) % migrate_every == 0`
używało **lokalnego `ge`** (0-4 w chunku 5-epokowym), a `migrate_every=10`. Więc `(ge+1)%10`
nigdy ≠ 0 w żadnym chunku → **żadna migracja ani merge NIGDY się nie odpalały** w chunkowanej
drabinie. Wyspy ewoluowały niezależnie, bez krzyżowania.

**Diagnoza.** merge ≡ stay w **100% kroków** (identyczny burst) — bo merge w trialu nic nie robił.

**Fix.** `do_mig = ((gge+1) % migrate_every == 0) or (force_merge == 1)` — globalne `gge` (nie
lokalne `ge`) + `force_merge` wymusza do_mig (kandydat merge faktycznie merge-uje).

**Efekt.** Cross-island diversity wróciło → **pr439 osiąga OPTIMUM (107217)** vs 107282 bez
migracji. merge ≠ stay. Koncept: w kodzie chunkowanym **każdy licznik oparty na epoce lokalnej
jest podejrzany** — musi być globalny, inaczej cichy no-op (kod „działał", dawał wyniki, a cały
mechanizm był martwy).

## 8. Merge back-to-back bezużyteczny

**Problem.** Merge miesza kolonie globalnie. Kolejny merge od razu nic nie wnosi (już zmieszane)
albo psuje świeżą różnorodność.

**Fix.** Cooldown: po wygranej merge `merge_cd=2` → następne 2 eskalacje merge wyfiltrowany z
kandydatów (3. znów pozwala).

**Efekt.** Nigdy merge 2× pod rząd, min. 2 eskalacje przerwy na re-dywergencję kolonii.

## 9. Early-stop PRZED próbą mocnych operatorów

**Problem (odkryty przez „czemu 442 gorzej niż Aparapi?").** pcb442 z tabu = 50961 (0.360%) —
gorszy niż Aparapi (0.315%) i bez-tabu (0.197%). Early-stop przy **step 3** (tylko stay/Or2/stay).

**Diagnoza.** Przy zbieżności rollout daje operatorom ~0 burst w 3-epokowym trialu → **„stay"
wygrywa** → ladder nie eskaluje do mocnych operatorów (3-opt, long-edge) → main-loop utyka na
gorszym lokalnym optimum → noimp≥3 → stop. Ladder **ubijał się zanim spróbował swojego głównego
narzędzia** (dodawania operatorów). To sprzężenie §5 (stay) z early-stopem: „stay" wygrywający
przy zerowych burstach = przedwczesny koniec.

**Fix.** (a) **„stay" nie może wygrać dopóki `ops != 252`** (wszystkie operatory dodane) — przy
zerowych burstach enumeracja `_cands` dodaje operatory w kolejności 3opt→Or2→Or3→LE→swap→segvar.
(b) **early-stop tylko w terminalu** (`ops==252`). Ladder musi przejść przez wszystkie operatory
zanim uzna zbieżność.

**Efekt.** pcb442: **0.360% → 0.028%** (step 3 → step 10, bije Aparapi 0.315% i v2 0.197%).
Koncept: **early-stop nie może wyprzedzać głównego mechanizmu poprawy** — skoro sensem drabiny
jest dodawanie operatorów, stop przed ich wyczerpaniem sabotuje własny algorytm.

## 10. Pułapka operacyjna: zwietrzały bytecode (.pyc)

**Problem.** Po `rsync` nowego kodu bieg używał STAREGO triggera mimo poprawnego źródła.

**Diagnoza.** `rsync -a` zachowuje mtime źródła. Jeśli mtime `.py` < mtime istniejącego `.pyc`,
Python bierze **zwietrzały `.pyc`**. Diagnostyka `inspect.getsource` pokazała nowy kod w module,
a runtime dawał stare zachowanie — mylące.

**Fix.** Po każdym deployu: `rm -rf __pycache__ && touch <plik>.py` przed kompilacją.

**Efekt.** Deterministyczny deploy. Lekcja ops: przy rsync + Python zawsze bustować cache.

---

## Podsumowanie efektów (pomiary)

| instancja | Aparapi | nasze (przed fixami) | nasze (po fixach) | opt |
|---|---|---|---|---|
| pr439 | 0.104% | 0.505% (agresywne tabu) | **0.000% (optimum)** | 107217 |
| pcb442 | 0.315% | 0.360% (za wczesny stop) | **0.028%** | 50778 |
| rl5934 | 2.168% | — | 2.131% (drabina, bije Javę) | 556045 |

## Meta-wnioski

1. **Patrz na surowe dane.** 4 realne bugi (tabu, migracja, early-stop, merge) były niewidoczne
   w „działa" — wyszły z jednej podejrzanej liczby (burst=0, wynik>Aparapi). Podgląd na żywo
   (serwer WWW) + logowanie każdej próby były kluczowe.
2. **Proste bije złożone.** ×1.004 (Aparapi) pobiło moją rosnącą karę + reinjekcję. Samoregulacja
   > skomplikowana logika.
3. **Sprzężenia są zdradliwe.** „stay" (dobry sam w sobie) + early-stop (dobry sam w sobie) =
   przedwczesny stop. Mechanizmy trzeba testować razem, nie osobno.
4. **W kodzie chunkowanym liczniki muszą być globalne.** Lokalny `ge` < `migrate_every` = cichy
   no-op całego mechanizmu.
5. **Sygnał sterujący ma wyprzedzać, nie opóźniać.** Deceleracja tempa > kolaps różnorodności.
