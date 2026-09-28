#!/bin/bash
# recipe from scratch (reproducibility measurement, Sebastian): after SWEEP_DONE -> greedy -> LS/LK -> ILS relaxed-penalty 0 for 4 h (or-opt, frag100, xswap2, orlong, uncross, excursion) -> post
export CUDA_HOME=$HOME/cudashim PYTHONPATH=$HOME/mf-santa
cd ~/mf-santa; RES=~/santa/results; PY=~/TspSolver/.venv/bin/python
until grep -q "SWEEP_DONE" $RES/sweep.log 2>/dev/null; do sleep 120; done
COMMON="--lk 3 --Lblk 30 --ils-margin 300 --lk-depth 5 --lk-br 8,5,3 --ils-w 400 --K 8 --lk-W 3000 --cand alpha --kick uniform --uncross 300 --orlong 3 --excursion 6 --lk-oropt"
echo "$(date) recipe_fast: step 1 greedy -> LS/LK" >> $RES/chain.log
$PY domains/santa/run.py --start greedy --time 1800 --pen 0.03 --relax-time 300 --full-time 300 --ils 0 --tag rf_r1 $COMMON > $RES/rf_r1.log 2>&1
R1=$(grep "^DONE" $RES/rf_r1.log | tail -1 | sed "s/.*-> //"); echo "$(date) recipe_fast: step 1 DONE $(grep ^DONE $RES/rf_r1.log | tail -1 | cut -c1-80)" >> $RES/chain.log
$PY domains/santa/run.py --start $R1 --time 600 --relax-time 300 --full-time 300 --pen 0.0 --frag 100 --xswap2 20 --ils 1000000 --ils-time 14400 --ils-rounds 2 --stall-iters 40 --stall-gain 100 --seed 91 --tag rf_r2 $COMMON > $RES/rf_r2.log 2>&1
R2=$(grep "^DONE" $RES/rf_r2.log | tail -1 | sed "s/.*-> //"); echo "$(date) recipe_fast: step 2 DONE $R2" >> $RES/chain.log
$PY domains/santa/orlong.py $R2 $RES/rf.tmp.csv --rounds 15 > $RES/rf_orlong.log 2>&1
$PY domains/santa/uncross.py $RES/rf.tmp.csv $RES/rf.tmp2.csv --all --rounds 40 --pen 0.0 > $RES/rf_uncross.log 2>&1
$PY domains/santa/run.py --start $RES/rf.tmp2.csv --time 600 --ils 0 --tag rf_post $COMMON > $RES/rf_post.log 2>&1; rm -f $RES/rf.tmp.csv $RES/rf.tmp2.csv
echo "$(date) recipe_fast DONE: $(grep ^DONE $RES/rf_post.log | tail -1 | cut -c1-90)" >> $RES/chain.log
