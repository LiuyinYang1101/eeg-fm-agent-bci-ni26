#!/usr/bin/env bash
# T2-D375 step 1 on one H200 (the local 5090 queue is held by another track's 1.6 h jobs): t2_d375_center.py = 12 seeds per family of the
# pooled FBCNet / ShallowNet-EA (the kit task), raw vs causally per-stream-centred linear heads on final_layer's input.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_d375.slurm --time=01:00:00 --nice=300 --job-name=ni26-t2p-d375-step1
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
python -u "$S1_CODE/tracks/t2_bci/scripts/t2_d375_center.py" --export "$D/export_proteus_v1" --out "$OUT" --device cuda
s1_finish $? "$OUT" 'd375_step1.json' 1
