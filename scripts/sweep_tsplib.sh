#!/bin/bash
# sweep over all TspSolver instances in ascending n; budget from n (solve.py); results in universal.jsonl + sweep.log
export CUDA_HOME=$HOME/cudashim PYTHONPATH=$HOME/mf-santa
cd ~/mf-santa; RES=~/santa/results; PY=~/TspSolver/.venv/bin/python
LIST=$(for f in ~/TspSolver/instances/*.tsp; do n=$(grep -m1 DIMENSION $f | tr -dc 0-9); echo "$n $(basename $f .tsp)"; done | sort -n | awk '{print $2}')
echo "$(date) sweep start: $(echo $LIST | wc -w) instancji" >> $RES/sweep.log
for i in $LIST; do
  [ -n "$SKIP_DONE" ] && grep -q "\"instance\": \"$i\"" $RES/universal.jsonl 2>/dev/null && continue
  r=$($PY domains/santa/solve.py $i --tag sw_$i 2>&1 | grep -E "^RESULT" | tail -1)
  echo "$(date +%H:%M) $i $(echo "$r" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()[7:]); print(f\"n={d['n']} len={d['length']} opt={d['optimum']} gap={d['gap_pct']:.3f}% wall={d['wall_s']:.0f}s\")" 2>/dev/null || echo FAIL)" >> $RES/sweep.log
done
echo "$(date) SWEEP_DONE" >> $RES/sweep.log
