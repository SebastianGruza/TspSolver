# pytsp — port CUDA (Python) memetycznego solvera TSP

Port `TspSolver` z **Java + Aparapi (OpenCL)** na **Python + CUDA**. Cel:
jakość rozwiązań **≥ oryginału dla dużych grafów (n > 3000)**.

## Architektura (docelowa)

- **Persistent cooperative kernel** — cały bieg ewolucyjny w jednym launchu;
  bariera epoki to `grid.sync()` (cooperative groups). Dane rezydentne na GPU;
  CPU tylko odpala i czyta „best" (pinned memory). Zero round-tripów GPU↔CPU
  i zero pracy algorytmicznej na CPU (wymóg projektu).
- **Hybryda**: orkiestracja + prototyp w **Numbie**; najgorętsze pętle
  (2-opt/3-opt eval) docelowo w **CUDA-C przez CuPy RawKernel** (pod profil).
- Model jak w oryginale: per-wątek niezależna wyspa memetyczna (OX crossover,
  2/3-opt, relokacje, swap, Tabu, XORShift PRNG); migracja/selekcja/tabu
  przeniesione na GPU (tournament selection zamiast CPU-sortu).

## Środowisko (gpu-node)

Toolkit Debiana jest rozrzucony — Numba potrzebuje shimu, ZAWSZE:

```bash
export CUDA_HOME=$HOME/cudashim     # symlinki do libcudadevrt.a + nvvm/libdevice
~/TspSolver/.venv/bin/python ...    # venv: numba, numpy, cuda-python, cupy-cuda12x
```

Bez shimu cooperative groups nie linkuje (`libcudadevrt.a not found`).

## Moduły

- `tsp_io.py` — loader TSPLIB EUC_2D + `full.txd` (miasta PL); macierz int32
  (zaokrąglenie `nint` wg konwencji TSPLIB — konieczne dla trafiania w optima);
  NN init; `tour_len`. **[gotowe, zwalidowane]**

## Status

Faza 1 (fundament: framework, env, cooperative `grid.sync` zwalidowane) — done.
Faza 2a (warstwa danych) — done. Dalej: PRNG + operatory memetyczne →
cooperative kernel → walidacja do optimum (berlin52=7542, kroA100=21282) →
skalowanie n>3000 + CUDA-C hot loops.
