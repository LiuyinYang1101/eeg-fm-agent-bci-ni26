#!/usr/bin/env bash
# T2-D124b: fingerprints of external multi-session corpora mapped onto the PROTEUS montage (t2p_fpcalib.py export), free CPU.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_fpcalib.slurm --partition=zen5_vis --time=02:00:00 \
#       --job-name=ni26-t2p-fpcalib -- stieger2021_v1 lee2019_mi_v1 shin2017b_v1
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
pools=""; for p in "$@"; do pools="$pools${pools:+,}$D/cpt_pools/$p"; done
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_fpcalib.py" export --pools "$pools" --names-from "$D/export_proteus_v1" --out "$OUT/fp_ext.npz"
s1_finish $? "$OUT" 'fp_ext.npz' 1
