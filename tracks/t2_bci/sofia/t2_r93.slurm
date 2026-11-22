#!/usr/bin/env bash
# T2-D397: the oracle ceiling of a per-user ST-L copy for returning users (t2_r93_stl_copy.py), one H200.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_r93.slurm --time=02:00:00 --job-name=ni26-t2p-d397-stlcopy [-- <extra args, e.g. --epochs 1 --seeds 120>]
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2_r93_stl_copy.py" --export "$D/export_proteus_v1" --ckpt "$S1_ROOT/runs/ni26/ni26-t2p-cpt-d111-with/checkpoint-cpt.pth" \
  --out "$OUT/r93.json" --device cuda "$@"
s1_finish $? "$OUT" 'r93.json' 1
