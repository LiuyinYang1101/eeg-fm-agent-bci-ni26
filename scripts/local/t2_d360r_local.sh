#!/usr/bin/env bash
# T2-D360r (T2-D323's EA ST-L on FRESH seeds 12-23) on the local 5090 (zero grant cost; the manager's 00:4x "move what fits to the
# 5090": 0 of 97 sofia jobs running). One protocol x a seed subset per gpu-submit job (≤ 3 h of local queue per track), each bench call
# in a MemoryMax=20G scope (the pattern of t2_d358_local.sh). The arms files' sofia --fm-ckpt paths are swapped for the local copies
# (same dir names -> the same result stems; t2p_fm.use_cpt re-checks the summary, the pinned base sha and the file sha).
# The 32 GB card OOMs at bs 64 (tsp 420, 2026-11-06 00:5x): --fm-micro ${MICRO:-4} = the exact gradient accumulation (TA568; one mixup
# draw per batch; ST has no BatchNorm), stems gain 'mb4'.
# Outputs: outputs/sofia_s1/ni26-t2p-d360r-stl-ea-s12-<kit|seen1>-5090/runs (both halves into one dir).
#   bash scripts/local/t2_d360r_local.sh <kit|seen1> <seeds, e.g. 12,13,14,15,16,17> [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
p=${1:?kit|seen1}; seeds=${2:?seed list}; dry=${3:-}
[[ $p == kit || $p == seen1 ]] || { echo "protocol kit|seen1"; exit 2; }
[ -z "$(git status --porcelain -- src/ tracks/t2_bci/scripts/ tracks/t1_image/scripts/t1_fm.py tracks/t2_bci/sofia/arms/t2d360r_*.txt)" ] || { echo "commit the D360r code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
d="$S/ni26-t2p-d360r-stl-ea-s12-$p-5090"
line="$(grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d360r_stl_ea_s12_$p.txt" | grep -v '^[[:space:]]*$' \
  | sed -E 's/^NSHARDS=[0-9]+ //; s#$S1_ROOT/runs/ni26/#'"$S"'/#; s/--seeds [0-9,]+/--seeds '"$seeds"'/') --fm-micro ${MICRO:-4}"
ck=$(grep -oE -- '--fm-ckpt [^ ]+' <<< "$line" | cut -d' ' -f2)
[ -f "$ck" ] && [ -f "$(dirname "$ck")/cpt_summary.json" ] || { echo "no local checkpoint + summary at $ck"; exit 2; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d360r_local.XXXX.sh")
tag="${seeds//,/_}"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "mkdir -p '$d/runs'"
  echo "echo \"### \$(date +%T) d360r $p seeds $seeds: $line\" >> '$d/arm.log'"
  echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected_$tag.txt' >/dev/null 2>&1 || rc=5"
  echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
  echo "n=\$(sort -u '$d/expected_$tag.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d ($seeds): \$(sort -u '$d/expected_$tag.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done \$(date +%T) rc=\$rc\" >> '$d/arm.log'; exit \$rc"
} > "$job"
if [ "$dry" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d360r-$p-$tag" bash "$job"
