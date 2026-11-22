#!/usr/bin/env bash
# T2-D485 (the manager's advance 3, 2026-11-14; zero grant, the local 5090 via gpu-submit; registered before any number): the five nearest
# CNN misses whose mechanism should grow with users, re-read at K = 9 (t2p_bench lc9: 9 public users train on session 1, 10 cyclic folds,
# every user NEW in exactly one) and at K = 5 (lc5, the within-protocol 'fewer users' twin), pooled members (no copies / TTA), `last`,
# seeds 400-404, paired by fold and seed with fresh controls on the same GPU (T2-D410's two base lines). Arms (tracks/t2_bci/sofia/arms/t2d485_k9.txt):
#   ctl_fbc / ctl_sh = D410's FBCNet rc75 (none) / ShallowNet vr (ea);  sw2 = D325 ShallowNet x2 width;  rc75m1 = D237 Bures-midpoint
#   re-colouring (FBCNet);  us5 = D365 user re-styling (both families);  fms5 = D297 MixStyle (FBCNet);  eegnex = D346 EEGNeX (a third family).
# Each arm runs with 4 parallel shards; the job checks every expected file. Read: scripts/local/t2_d485_read.py.
#   bash scripts/local/t2_d485_k9_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARMS=tracks/t2_bci/sofia/arms/t2d485_k9.txt
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts $ARMS scripts/local/t2_d485_k9_local.sh)" ] || { echo "commit the D485 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
d=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-d485-k9-5090
NS=4
job=$(mktemp "${TMPDIR:-/tmp}/t2_d485_k9_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=8G -p MemoryMax=10G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    echo "echo \"### \$(date +%T) d485: $line\" >> '$d/arm.log'"
    echo "$PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    for i in $(seq 0 $((NS - 1))); do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards $NS $line >> '$d/shard_$i.log' 2>&1 &"
    done
    echo "wait"
  done < "$ARMS"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done d485 \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
[ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d485" bash "$job"
