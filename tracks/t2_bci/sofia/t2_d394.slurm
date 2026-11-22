#!/usr/bin/env bash
# T2-D394 step 2 on one H200: t2_d375_member.py --dres = the centred / raw probes + the G2-B1 density-ratio adj (12 seeds x 2 families x kit + seen1).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_d375m.slurm --time=01:30:00 --nice=300 --job-name=ni26-t2p-d394-dres
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
python -u "$S1_CODE/tracks/t2_bci/scripts/t2_d375_member.py" --export "$D/export_proteus_v1" --out "$OUT" --device cuda --dres
s1_finish $? "$OUT" 'd375_member.json' 1
