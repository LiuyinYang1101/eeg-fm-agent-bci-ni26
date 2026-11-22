#!/usr/bin/env bash
# T2-D295 launch (backlog 65), two steps, both only after sofia returns (>= 2026-10-28 18:00):
#   bank : the external covariance bank on free zen5_vis CPU (zero GPU-h; RUN_NAME fixed: the rcx75 arms read
#          $S1_ROOT/runs/ni26/ni26-t2p-covbank/covbank.npz); prints the job id.
#   gpu [bank_job_id] : ONLY on the user's word on row 65: one `s1 sync ni26` (git archive HEAD: commit the code first), then three t2p_multi
#          jobs (FBCNet kit / seen1: rcx75 + rc75_cap10; ShallowNet kit + seen1: vr_cap10; ~0.5 GPU-h in all); with a bank job id the FBCNet
#          jobs wait for it (afterok), without one the bank must already exist.
# Read: `s1 fetch ni26 ni26-t2p-r65-<arm>-<id>` x 3, then `mem-guard bash scripts/local/t2_r65_read.sh`.
#   bash scripts/local/t2_r65_launch.sh bank
#   bash scripts/local/t2_r65_launch.sh gpu [bank_job_id]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_covbank.py tracks/t2_bci/sofia/t2p_covbank.slurm tracks/t2_bci/sofia/arms/t2r65_*.txt)" ] || { echo "commit the row 65 code first"; exit 2; }
case "${1:-}" in
  bank)
    s1 sync ni26
    S1_EXPORT=RUN_NAME=ni26-t2p-covbank S1_PURPOSE="T2 backlog 65 (a): external covariance bank (Stieger2021 / Lee2019 on the 41-ch map), free CPU, zero GPU-h; feeds T2-D295" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_covbank.slurm --partition=zen5_vis --time=02:00:00 --job-name=ni26-t2p-covbank -- stieger2021_v1 lee2019_mi_v1 shin2017b_v1 ;;
  gpu)
    dep=(); [ -n "${2:-}" ] && dep=(--dependency=afterok:$2)
    s1 sync ni26
    for arm in fbc_kit fbc_seen1 sh; do
      d=(); [[ $arm == fbc_* ]] && d=("${dep[@]}")
      S1_EXPORT=NSHARDS=3 S1_PURPOSE="T2-D295 nuisance arms (row 65, on the user's word): t2r65_$arm (rcx75 / cap10), feeds the iteration builds (PLAN 6b item 11)" \
        s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm "${d[@]}" --time=02:00:00 --job-name=ni26-t2p-r65-${arm//_/-} -- tracks/t2_bci/sofia/arms/t2r65_$arm.txt
    done ;;
  *) echo "usage: $0 bank | gpu [bank_job_id]"; exit 2 ;;
esac
