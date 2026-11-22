#!/usr/bin/env bash
# Backlog 65 (a): the external covariance bank (t2p_covbank.py; per-subject mean covariances on the PROTEUS montage), free CPU.
#   S1_EXPORT=RUN_NAME=ni26-t2p-covbank S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_covbank.slurm --partition=zen5_vis --time=02:00:00 \
#       --job-name=ni26-t2p-covbank -- stieger2021_v1 lee2019_mi_v1 shin2017b_v1
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")   # RUN_NAME=ni26-t2p-covbank -> the fixed path the rcx75 arms read
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
pools=""; for p in "$@"; do pools="$pools${pools:+,}$D/cpt_pools/$p"; done
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_covbank.py" --pools "$pools" --names-from "$D/export_proteus_v1" --out "$OUT/covbank.npz"
s1_finish $? "$OUT" 'covbank.npz' 1
