#!/bin/bash
# Recipe from scratch (docs/santa/RECIPE.md) - run once: bash scripts/santa_recipe.sh [TAGPREFIX]
# Requires: ~/santa/data/cities.csv, venv ~/TspSolver/.venv, ~/santa/gpx_merge.py
set -u
export CUDA_HOME=$HOME/cudashim PYTHONPATH=$HOME/mf-santa
cd ~/mf-santa
PY=~/TspSolver/.venv/bin/python; RES=~/santa/results; T=${1:-recipe}
COMMON="--lk 3 --Lblk 30 --ils-margin 300 --lk-depth 5 --lk-br 8,5,3 --ils-w 400 --K 8 --lk-W 3000 --cand alpha --kick uniform --uncross 300 --orlong 3"
STEP2="--time 600 --relax-time 300 --full-time 300 --pen 0.0 --frag 100 --xswap2 20 --ils 1000000 --ils-rounds 2"
log() { echo "$(date '+%F %T') $*" | tee -a $RES/${T}_recipe.log; }
best() { grep "^DONE" $RES/$1.log | tail -1 | sed "s/.*-> //"; }

log "step 1: Moore -> LS -> LS+LK (1 h)"
$PY domains/santa/run.py --start moore --time 3600 --pen 0.03 --relax-time 600 --full-time 600 --ils 0 --tag ${T}_r1 $COMMON > $RES/${T}_r1.log 2>&1
R1=$(best ${T}_r1); log "step 1 DONE $R1"

log "step 2: ILS relaxed penalty 0 + repair + frag100 + xswap2 (6 h)"
$PY domains/santa/run.py --start $R1 $STEP2 --ils-time 21600 --stall-iters 40 --stall-gain 150 --seed 1 --tag ${T}_r2 $COMMON > $RES/${T}_r2.log 2>&1
R2=$(best ${T}_r2); log "step 2 DONE $R2"

LINES=""
for sd in 21 22 23; do
  log "step 3: line seed $sd (3 h)"
  $PY domains/santa/run.py --start $R2 $STEP2 --ils-time 10800 --stall-iters 40 --stall-gain 150 --seed $sd --tag ${T}_r3_s$sd $COMMON > $RES/${T}_r3_s$sd.log 2>&1
  L=$(best ${T}_r3_s$sd); LINES="$LINES $L"; log "step 3 line $sd DONE $L"
done

log "step 4: GPX"
$PY ~/santa/gpx_merge.py $RES/${T}_gpx.csv $R2 $LINES > $RES/${T}_gpx.log 2>&1; log "step 4 $(grep GPX_CHILD $RES/${T}_gpx.log)"

log "step 5: follow-up from the GPX child (4 h)"
$PY domains/santa/run.py --start $RES/${T}_gpx.csv $STEP2 --ils-time 14400 --stall-iters 60 --stall-gain 100 --seed 7 --tag ${T}_r5 $COMMON > $RES/${T}_r5.log 2>&1
log "step 5 DONE $(grep ^DONE $RES/${T}_r5.log | tail -1)"
