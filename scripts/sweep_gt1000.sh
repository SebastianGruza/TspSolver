#!/bin/bash
# Regression sweep after the English-only refactor: all TspSolver instances with n > 1000, via solve.py (n-scaled budget).
# Output: ~/santa/results/sweep_gt1000.log (one line per instance) + universal.jsonl (tag rg_<instance>).
export CUDA_HOME=$HOME/cudashim PYTHONPATH=$HOME/mf-santa
cd ~/mf-santa; RES=~/santa/results; PY=~/TspSolver/.venv/bin/python
LIST=$(for f in ~/TspSolver/instances/*.tsp; do n=$(grep -m1 DIMENSION $f | tr -dc 0-9); [ "$n" -gt 1000 ] && echo "$n $(basename $f .tsp)"; done | sort -n | awk '{print $2}')
echo "$(date) sweep_gt1000 start: $(echo $LIST | wc -w) instances" >> $RES/sweep_gt1000.log
for i in $LIST; do
  r=$($PY domains/santa/solve.py $i --tag rg_$i 2>&1 | grep -E "^RESULT|Traceback" | tail -1)
  if echo "$r" | grep -q Traceback; then echo "$(date +%H:%M) $i FAIL: $r" >> $RES/sweep_gt1000.log; continue; fi
  echo "$(date +%H:%M) $i $(echo "$r" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()[7:]); print(f\"n={d['n']} mode={d.get('mode')} len={d['length']} opt={d['optimum']} gap={d['gap_pct']:.3f}% wall={d['wall_s']:.0f}s\")" 2>/dev/null || echo "PARSE_FAIL $r")" >> $RES/sweep_gt1000.log
done
echo "$(date) SWEEP_GT1000_DONE" >> $RES/sweep_gt1000.log
