# TSPLIB results — one entry point (`domains/santa/solve.py`)

Mode by size: **population** (GPU memetic `solve_ga`: 8192/4096/2048 individuals, OX, colonies, tabu) + LK/or-opt polish for n <= 5000; **fragments** (greedy -> LS/LK -> ILS with relaxed penalty 0 + repair) above. Budget and parameters scale with n. Lengths in the exact TSPLIB metric (nint = floor(x+0.5)). `previous solver` = best result of the 2025 CPU/GPU solver from the legacy README.

| instance | n | mode | length | optimum | gap | wall time | previous solver |
|---|---|---|---|---|---|---|---|
| berlin52 | 52 | fragments | 7542 | 7542 | 0.000% | 1 min | — |
| kroA100 | 100 | fragments | 21282 | 21282 | 0.000% | 2 min | — |
| fl417 | 417 | population | 11861 | 11861 | 0.000% | 2 min | — |
| gr431 | 431 | population | 171426 | 171414 | 0.007% | 4 min | 0.64% |
| pr439 | 439 | population | 107217 | 107217 | 0.000% | 3 min | 0.10% |
| pcb442 | 442 | population | 50792 | 50778 | 0.028% | 3 min | 0.32% |
| d493 | 493 | population | 35005 | 35002 | 0.009% | 4 min | 0.55% |
| att532 | 532 | population | 27705 | 27686 | 0.069% | 4 min | 0.98% |
| ali535 | 535 | population | 202380 | 202339 | 0.020% | 4 min | 1.04% |
| u574 | 574 | population | 36905 | 36905 | 0.000% | 6 min | 0.65% |
| rat575 | 575 | population | 6773 | 6773 | 0.000% | 5 min | 0.69% |
| p654 | 654 | population | 34643 | 34643 | 0.000% | 3 min | 0.44% |
| d657 | 657 | population | 48945 | 48912 | 0.067% | 6 min | 0.86% |
| gr666 | 666 | population | 295207 | 294358 | 0.288% | 7 min | 1.10% |
| u724 | 724 | population | 42009 | 41910 | 0.236% | 12 min | 1.06% |
| rat783 | 783 | population | 8807 | 8806 | 0.011% | 8 min | 1.43% |
| pr1002 | 1002 | population | 259558 | 259045 | 0.198% | 10 min | 1.19% |
| u1060 | 1060 | population | 224751 | 224094 | 0.293% | 7 min | 1.03% |
| vm1084 | 1084 | population | 239371 | 239297 | 0.031% | 14 min | 0.94% |
| pcb1173 | 1173 | population | 56982 | 56892 | 0.158% | 18 min | 2.05% |
| d1291 | 1291 | population | 50961 | 50801 | 0.315% | 19 min | 0.41% |
| rl1304 | 1304 | population | 253361 | 252948 | 0.163% | 11 min | 0.72% |
| rl1323 | 1323 | population | 270489 | 270199 | 0.107% | 11 min | 1.84% |
| nrw1379 | 1379 | population | 57172 | 56638 | 0.943% | 28 min | 1.73% |
| fl1400 | 1400 | population | 20131 | 20127 | 0.020% | 15 min | 0.26% |
| u1432 | 1432 | population | 154234 | 152970 | 0.826% | 10 min | 1.42% |
| fl1577 | 1577 | population | 22293 | 22249 | 0.198% | 24 min | 0.13% |
| d1655 | 1655 | population | 62298 | 62128 | 0.274% | 15 min | 0.63% |
| vm1748 | 1748 | population | 337735 | 336556 | 0.350% | 36 min | 1.24% |
| u1817 | 1817 | population | 57396 | 57201 | 0.341% | 31 min | 1.79% |
| rl1889 | 1889 | population | 317552 | 316536 | 0.321% | 46 min | 1.96% |
| d2103 | 2103 | population | 80823 | 80450 | 0.464% | 33 min | 0.66% |
| u2152 | 2152 | population | 64772 | 64253 | 0.808% | 34 min | 1.97% |
| u2319 | 2319 | population | 236318 | 234256 | 0.880% | 20 min | 1.55% |
| pr2392 | 2392 | population | 381977 | 378032 | 1.044% | 24 min | 1.97% |
| pcb3038 | 3038 | hybrid | 139578 | 137694 | 1.368% | 92 min | 2.50% |
| fl3795 | 3795 | population | 28913 | 28772 | 0.490% | 43 min | 0.98% |
| fnl4461 | 4461 | population | 186022 | 182566 | 1.893% | 162 min | 2.47% |
| rl5915 | 5915 | fragments | 575394 | 565530 | 1.744% | 40 min | 2.34% |
| rl5934 | 5934 | fragments | 565720 | 556045 | 1.740% | 41 min | 2.17% |
| gr9882 | 9882 | fragments (manual runs a-d) | 303930 | 300899 | 1.007% | 300 min | 3.57% |

Instances: 41 · median gap 0.198% · <= 0.1%: 16 · <= 1%: 35 · max 1.89%
