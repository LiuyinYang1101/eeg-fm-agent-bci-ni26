#!/usr/bin/env bash
# T2-D638 W24' builder GOLDEN on the LOCAL 5090 (Track3 + the manager: the same GPU / env as D630's reference nets, gpu-submit job 114;
# queued behind D638's job 116): t2p_final's pooled-net path on ONE t2p_bench fold (--train-protocol ss8_P01, seed 880, the LAB recipes =
# D630's arm lines), predicting EXACTLY the fold's test rows in the bench's order; per family B (unweighted) and C (--user-balance); FBCNet C
# twice (its run-to-run drift: the spatial conv has groups = 13). Then scripts/local/t2_w24_golden_check.py vs D630's B / C pooled files:
# ShallowNet BITWISE; FBCNet / IFNet <= 1e-3 with 100 % argmax. -> outputs/t2_w24_golden/ (golden_<fam>_<arm>.npz, check.log)
#   mem-guard bash scripts/local/t2_w24_golden_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_w24_golden_check.py scripts/local/t2_w24_golden_local.sh)" ] || { echo "commit the W24' golden code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
G=$HOME/workspace/outputs/neural_interface_challenge/t2_w24_golden
[ ! -e "$G" ] || { echo "$G exists"; exit 3; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_w24_golden_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$G'; echo \"### \$(date +%T) w24 golden start (git \$(git rev-parse --short HEAD))\" >> '$G/check.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"   # = t2_d630_local.sh's job env
  for spec in "sh_B:shallow:ea:r1_pulx_vp2_e180_vr:" "sh_C:shallow:ea:r1_pulx_vp2_e180_vr:--user-balance" \
              "ifn_B:ifnet:ea:r1_pulx_vp2_e180_vr:" "ifn_C:ifnet:ea:r1_pulx_vp2_e180_vr:--user-balance" \
              "fbc_B:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75:" "fbc_C:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75:--user-balance" \
              "fbc_C_rep:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75:--user-balance"; do
    IFS=: read -r tag model al rec ub <<< "$spec"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_final.py --export '$EXP' --out '$G/pkg_$tag' --train all --train-protocol ss8_P01 --seeds 880 --epochs 180 --model $model --align $al --recipe $rec $ub --golden-out '$G/golden_$tag.npz' >> '$G/run_$tag.log' 2>&1 || { rc=5; echo '!! $tag failed' >> '$G/check.log'; }"
    echo "rm -rf '$G/pkg_$tag'"
  done
  echo "OMP_NUM_THREADS=1 $PY -u scripts/local/t2_w24_golden_check.py '$G' >> '$G/check.log' 2>&1 || rc=6"
  echo "echo \"### done w24 golden \$(date +%T) rc=\$rc\" >> '$G/check.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-w24-golden" bash "$job"
