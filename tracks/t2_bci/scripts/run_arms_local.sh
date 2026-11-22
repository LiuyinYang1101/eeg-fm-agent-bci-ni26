#!/usr/bin/env bash
# Track 2: run a bench arms file (one t2p_bench.py argument line per arm, '#' comments allowed) on the LOCAL GPU, one arm
# after another (the local twin of sofia/t2p_multi.slurm). Submit through the queue:
#   gpu-submit bash tracks/t2_bci/scripts/run_arms_local.sh tracks/t2_bci/sofia/arms/<file>.txt <out dir>
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-~/miniconda3/envs/neural_interface_challenge/bin/python}"
EXPORT="${T2P_EXPORT:-$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1}"
ARMS="$1"; OUT="$2"; mkdir -p "$OUT/runs"
echo "### $(date -Is) git $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" status --porcelain -- src tracks/t2_bci | wc -l) dirty; arms $ARMS"
rc=0; a=0
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [ -z "${line// /}" ] && continue
  a=$((a + 1)); echo "### arm $a ($(date +%T)): $line"
  # shellcheck disable=SC2086
  "$PY" -u "$REPO/tracks/t2_bci/scripts/t2p_bench.py" --export "$EXPORT" --out "$OUT/runs" $line > "$OUT/arm${a}.log" 2>&1 || rc=$?
  grep -h "cellBA\|SKIPPED" "$OUT/arm${a}.log" | tail -n 4
  grep -h -E "Traceback|Error" "$OUT/arm${a}.log" | grep -v OutOfMemory | head -5
done < "$ARMS"
echo "### $(date -Is) $(ls "$OUT/runs"/*.npz 2>/dev/null | wc -l) result files, rc $rc"
exit $rc
