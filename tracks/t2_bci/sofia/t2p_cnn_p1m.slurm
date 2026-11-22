#!/usr/bin/env bash
# T2-D398: multi-source supervised pretraining of one pooled CNN family on 7 labelled cpt_pools (t2p_cnn_p1.py --pools) -> OUT/p1.pt.
#   S1_EXPORT=RUN_NAME=<stable dir> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_cnn_p1m.slurm --time=01:30:00 -- <model> <align> [extra args]
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
P="$D/cpt_pools"
POOLS="$P/bnci2014_001_v1,$P/bnci2015_001_v1,$P/cho2017_v1,$P/lee2019_mi_v1,$P/schirrmeister2017_v1,$P/stieger2021_v1,$P/weibo2014_v1"   # not physionetmi (the in-house FM project's eval set) / shin2017b (licence unclear)
MODEL=${1:?model}; ALIGN=${2:?align}; shift 2
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_cnn_p1.py" --export "$D/export_proteus_v1" --pools "$POOLS" --model "$MODEL" --align "$ALIGN" \
  --out "$OUT/p1.pt" --device cuda "$@"
s1_finish $? "$OUT" 'p1.pt' 1
