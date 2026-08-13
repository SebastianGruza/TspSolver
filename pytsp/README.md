# pytsp — port CUDA (Python) memetycznego solvera TSP

Port `TspSolver` z **Java + Aparapi (OpenCL)** na **Python + CUDA** (Numba,
cooperative groups). Cel: jakość rozwiązań **≥ oryginału, zwłaszcza dla dużych
grafów (n > 3000)**. Pełna tabela różnic vs Java i „dlaczego to się poprawiło" —
w [głównym README](../README.md).

## Architektura (jak zbudowane)

- **Persistent cooperative kernel** — cały bieg ewolucyjny w jednym launchu;
  bariera epoki to `grid.sync()` (cooperative groups). Dane rezydentne na GPU,
  CPU tylko odpala kernel i czyta najlepszą trasę. **Zero round-tripów GPU↔CPU**
  i zero pracy algorytmicznej na CPU (wymóg projektu).
- **Populacyjny memetyk** — `T` wysp × `pm` osobników. Na generację (per wyspa):
  OX crossover + local search dziecka → **selekcja elitarna** (dziecko wypiera
  najsłabszego) → **perturbacja double-bridge** najsłabszego + LS.
- **Local search** — 2-opt ⇄ Or-opt ⇄ 3-opt na listach `K` najbliższych sąsiadów
  (KNN) z tablicą pozycji i pruningiem: `O(n·K)` na sweep do lokalnego optimum.
- **Kolonie** (`C`) — migracja wewnątrz kolonii (race-free: bufor migranta +
  2 fazy `grid.sync`); opcjonalny **merge globalny** w 4 oknach wokół progów
  budżetu (0.25/0.5/0.75/0.9) → miesza kolonie, potem re-dywergencja.
- **Tabu** (opcja) — osobnik „za długo w czołówce" (starszy niż próg) dostaje
  **+0.4% do długości efektywnej** w selekcji (analog kary ×1.004 z oryginału),
  żeby dało się go wyprzeć = dywersyfikacja. `best-ever` per wyspa (`gbest`)
  śledzony osobno i **odporny na karę** — wynik nigdy nie gubi prawdziwego optimum.
- PRNG **XORShift 1:1 z oryginałem**; dystanse TSPLIB **EUC_2D / GEO / ATT**.

> Obecnie 100% Numba. Furtka: najgorętsze pętle można zejść do CUDA-C (CuPy
> RawKernel), ale na razie nie było potrzeby — jakość/czas dała już geometria
> (neighbor-list LS + dużo wysp), nie mikro-optymalizacja pętli.

## Środowisko (gpu-node)

Toolkit Debiana jest rozrzucony — Numba potrzebuje shimu, **ZAWSZE** eksportuj:

```bash
export CUDA_HOME=$HOME/cudashim     # symlinki: lib64/libcudadevrt.a + nvvm/libdevice
# venv: ~/TspSolver/.venv (numba, numpy, cuda-python, cupy-cuda12x)
```

Bez shimu cooperative groups nie linkuje (`libcudadevrt.a not found`).

## Uruchomienie

```bash
export CUDA_HOME=$HOME/cudashim
cd ~/TspSolver
# argumenty: <instancja.tsp> [T] [ge] [pm] [C] [merge 0/1] [tabu 0/1]
.venv/bin/python pytsp/memetic_ga.py instances/gr431.tsp 4096 200 4 4 1 0
```

- **T** — liczba wysp (więcej = wypełnia GPU; przy n<~1000 niemal darmowe,
  przy n≥3000 kosztuje czas, ale podnosi moc/util),
- **ge** — liczba generacji = budżet (więcej = lepiej aż do zbieżności),
- **pm** — osobniki/wyspę (crossover), **C** — kolonie,
- **merge / tabu** — przełączniki do A/B.

## Moduły

- **`memetic_ga.py`** — GŁÓWNY. Populacyjny GA: OX crossover, kolonie + merge,
  2/3-opt + Or-opt (neighbor-list), double-bridge, tabu, migracja na GPU.
- `tsp_io.py` — loader TSPLIB (EUC_2D/GEO/ATT) + `full.txd` (miasta PL);
  macierz `int32` (zaokrąglenie wg konwencji TSPLIB — konieczne dla optima); NN init.
- `prng_dev.py` — PRNG XORShift (device, współdzielony).
- `memetic.py` — wcześniejszy kooperatywny ILS (etap pośredni, pm=1).
- `gpu_core.py` — walidacja rdzenia (PRNG + `tour_len` + 2-opt).

## Wyniki (RTX 3090)

| instancja | oryginał (Java/Aparapi) | port (Python+CUDA) |
|:--|:--|:--|
| berlin52, kroA100 | optimum | **optimum (0.000%)** |
| gr431 (GEO, n=431) | 0.64% / 212 s | **0.000% / 85 s** |
| pr1002 (EUC, n=1002) | 1.19% / 817 s | **0.191% / 328 s** (0.173% z merge) |

Na testowanym zakresie port **bije oryginał jakościowo i czasowo**. `perm_ok=True`
na każdym biegu (operatory permutacyjnie poprawne — brak naprawy integralności).

## A/B (walidacja merge i tabu)

Obie opcjonalne dźwignie dywersyfikacji sprawdzone kontrolowanym A/B przy **tym samym
budżecie** (RTX 3090) — obie dają mały, spójny zysk na dużych instancjach, domyślnie OFF:

| mechanizm | instancja | off | on |
|:--|:--|:--|:--|
| **merge kolonii** (globalne mieszanie w 4 punktach budżetu 0.25/0.5/0.75/0.9) | pr1002 (n=1002) | 0.191 % | **0.173 %** |
| **tabu** (osobnik za długo w czołówce → +0.4 % kary do długości *efektywnej* w selekcji; `gbest` odporny na karę) | pcb3038 (n=3038, 300 epok) | 1.458 % | **1.319 %** |

Przy tym samym budżecie pcb3038 schodzi do **1.32 %** vs 2.5–3.1 % oryginału.
Włączasz przełącznikami `merge` / `tabu` (patrz *Uruchomienie*).

## Status / dalej

Zrobione: pełny pakiet operatorów + kolonie/merge + tabu, zwalidowane do optimum
na małych, bije oryginał na gr431/pr1002. W toku: **pcb3038 / 3k+** z pełnym
pakietem (A/B tabu, wysokie T fill GPU 200→290 W). Dalej opcjonalnie: Or-2/Or-3,
warp-per-island (nasycenie GPU dla dużych n), CUDA-C na hot loops jeśli profil zażąda.
