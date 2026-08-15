# Wyniki — memetic GA (CUDA/Numba), sesja 2026-08-14

Instancje: **rl5934** (opt 556045), **fnl4461** (opt 182566). T=8192, ge=100,
uniq/merge/tabu on, seed=1. Surowe logi w tym katalogu.

## 1. Jakość — główne konfiguracje

| konfiguracja | rl5934 gap | fnl4461 gap | uwaga |
|---|---|---|---|
| klasyk (2/3-opt cały czas) | 2.855% | 3.096% | baseline |
| adaptive-K (klasyk + smooth K) | 2.645% | — | |
| **hybryda (2/3-opt → relokacje@60 RAZEM, K+20%)** | **2.246%** | **2.554%** | **rekord** |
| relokacje standalone (bez 2/3-opt) | ~11% (stall) | — | falsyfikacja |
| *oryginał Java Aparapi* | *2.17%* | — | *~12060s* |

Przewaga hybrydy vs klasyk: **−0.61pp (rl5934), −0.54pp (fnl4461)** — spójna na obu rodzinach.
Główna dźwignia: relokacje@60 (skok −1.03pp rl5934 / −0.62pp fnl4461 w oknie po bramce).
K+20% dokłada ~−0.3pp. `lsab.log`, `hyb.log`, `gen.log`.

## 2. Krzyżowania (A/B na rekordowym presecie hybrydy, rl5934) — `ox.log`

| krzyżowanie | best | gap |
|---|---|---|
| OX klasyk | 568532 | 2.246% |
| OX long-edge (top10) | 568142 | 2.176% |
| OX mix50 (top10) | 568103 | 2.169% |
| greedy-edge (ox_mode=3) | — | 25/100=6.13%, 2.5× wolniej (ubity) |

Rozrzut OX = 429 jednostek (0.075%) = **szum na 1 seedzie**. Wniosek: przy silnym LS
**wybór krzyżowania jest wysycony** — nie jest dźwignią. greedy-edge sam od startu szkodzi
(kandydat wyłącznie na późną fazę). `ge.log`.

## 3. Metryki stall-detekcji — ultra-fast (ls_mode=3, K=8), chunk=5 — `met.log`, `metf.log`

Sygnał do wykrywania „kiedy dopalać wolne-głębokie mechanizmy":

| sygnał | jakość | knee rl5934 | knee fnl4461 |
|---|---|---|---|
| r_norm (tempo best) | SZUM, lagging, misfire | — | — |
| cv% (peak = koniec dywersyfikacji) | gładki, leading | ep30 | ep5 (brak fazy) |
| uniq% (kolaps różnorodności) | najgładszy, leading | plateau do ep30 | zjazd od ep5 (start 69) |
| best flatline | lagging | ~ep60 | ~ep35 |

**Wynik:** różnorodność (uniq%/cv%) wyprzedza zastój best o ~30 epok i **auto-skaluje próg
per instancja** — fnl4461 (startuje skonwergowany) przegina ~2× wcześniej niż rl5934.
Wyzwalacz eskalacji: `cv_peaked AND uniq% < f·plateau AND smoothed r_norm < τ`.
NIE triggerować na surowym r_norm.

## Parametry (solve_ga)
- `ls_mode`: 0=klasyk, 1=relokacje, 2=hybryda(2/3-opt→relokacje@opt_start), 3=ultra-fast(2-opt+Or-1)
- `opt_start`=0.6, `le_start`, `ox_mode` 0/1/2/3, `ox_topk`, K progi max(8,n//600)/max(12,n//300)/max(15,n//200)
- `metrics=True` → tabela best/rel-ep/r_norm/spread/cv/uniq/stagn co chunk

## 4. Drabina auto-eskalacji + bandyta discovery (15.08)

Kontroler eskalacji: baza 2-opt+Or-1, atomowe ruchy (K+2, +operator, ox, merge) odpalane
gdy DECELERACJA (świeży spadek < decel_rho * szczytowe tempo kroku). ls_mode=9 = bitmaska ops.

Kluczowe wnioski:
- Trigger = deceleracja (recent/peak), NIE surowe tempo — uniq% laguje, opóźniało climb ~2x.
- Kolejność ruchów DOMINUJE nad granularnością: high-value first (3-opt->Or2->Or3->long-edge
  ->greedy-edge) bije K-bumpy-first o ~12k jednostek w tym samym czasie.
- Auto-drabina (reorder) rl5934 final = 567893 = **2.131%** — bije hybrydę (2.246%),
  OX-mix50 (2.169%) I oryginal Javy (2.17%). Bez strojenia per instancja. Cena: wolno
  (dochodzi ~24700s; hybryde-level ~570k osiaga juz ~2660s).

Bandyta (discovery-mode): na eskalacji ROLLOUT — snapshot (R,rlen,gbest,gbr,age,RNG),
proba kazdego kandydata (trial_chunks) z tego samego startu, log (stan,ruch,burst,dt,
reward=burst/dt) do SQLite results/ladder_trials.db, commit zwyciezcy. Przetestowane kroA100
(perm_ok, 18 krotek). Offline agregacja (disc_agg.py) -> statystycznie najlepsza kolejnosc.
