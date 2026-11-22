#!/usr/bin/env bash
# T2-D396: supervised pretraining of one pooled CNN family on an external labelled corpus (t2p_cnn_p1.py) -> OUT/p1.pt for t2p_bench --cnn-init.
#   S1_EXPORT=RUN_NAME=<stable dir> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_cnn_p1.slurm --time=01:00:00 -- <model> <align> <recipe> <ext export name>
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
MODEL=${1:?model}; ALIGN=${2:?align}; REC=${3:?recipe}; EXT=${4:?ext export name}
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_cnn_p1.py" --export "$D/export_proteus_v1" --ext "$D/$EXT" --model "$MODEL" --align "$ALIGN" \
  --recipe "$REC" --out "$OUT/p1.pt" --device cuda
s1_finish $? "$OUT" 'p1.pt' 1
