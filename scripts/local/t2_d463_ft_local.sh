#!/usr/bin/env bash
# T2-D463 FULL-FT RE-READ on the local 5090 (zero grant; the manager's ask 2026-11-13, panel P10's open question "closed under an adapter
# inherited from the 100M MOR"): the RAW C2L300 file (c2l.pt = runs/fm/ni26fm_C2long_ck150k_s0/checkpoint-00150000.pt, sha256 cc3a98b4...)
# fully fine-tuned with M7's ST-L training recipe (lr 1e-4, LLRD 0.75, mixup 0.4, 100 epochs, batch 64 as 4 exact chunks) on the MOR line's
# own input path (wz, zwin calibration, rc 0.75), seeds 500-504 (the LoRA raw run's first five: a paired adapter-vs-full read), kit then seen1,
# in ONE gpu-submit job. Dirs: outputs/sofia_s1/ni26-t2p-d463-ft-{kit,seen1}-5090. Read: t2_d460_read.py --d463 --kinds=raw,ft --n=3
#   bash scripts/local/t2_d463_ft_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t1_image/scripts/t1_fm.py scripts/local/t2_d463_ft_local.sh scripts/local/t2_d460_read.py)" ] || { echo "commit the D463 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
CK=$O/sofia_s1/ni26-t2p-d463-ckpt-raw/c2l.pt
[ -f "$CK" ] && [ ! -L "$CK" ] && [ -f "$CK.contract.json" ] || { echo "need the fetched raw file at $CK (not the sofia symlink) + its contract"; exit 2; }
MICRO=${D463_MICRO:-4}
job=$O/sofia_s1/ni26-t2p-d463-ft-kit-5090/job.sh
mkdir -p "$O/sofia_s1/ni26-t2p-d463-ft-kit-5090/runs" "$O/sofia_s1/ni26-t2p-d463-ft-seen1-5090/runs"
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for pr in kit seen1; do
    d=$O/sofia_s1/ni26-t2p-d463-ft-$pr-5090
    line="--protocols $pr --seeds 500,501,502,503,504 --models fm:mor_p1v2:mean --aligns wz --fm-rc 0.75 --fm-ckpt $CK --fm-epochs 100 --fm-lr-enc 1e-4 --fm-llrd 0.75 --fm-mixup 0.4 --fm-micro $MICRO"
    echo "git rev-parse HEAD > '$d/git_head.txt'"
    echo "echo \"### \$(date +%T) d463 ft $pr: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "echo \"### done d463-ft-$pr \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; exit 0; fi
for pr in kit seen1; do [ -z "$(ls -A "$O/sofia_s1/ni26-t2p-d463-ft-$pr-5090/runs" 2>/dev/null)" ] || { echo "ft-$pr runs is not empty"; exit 2; }; done
gpu-submit -L "neural_interface_challenge:t2-d463-ft" bash "$job"
