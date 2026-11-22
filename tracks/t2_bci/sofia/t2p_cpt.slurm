#!/usr/bin/env bash
# Track 2: continued MAE pretraining of ST-EEGFormer on unlabelled public BCI EEG (t2p_cpt.py). Args after `--` go to it
# (--out is set here). Pools are read from data/ni26/t2_proteus/ (staged node-locally).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_cpt.slurm --time=00:30:00 --job-name=ni26-t2p-cpt-bench \
#       -- --fm st_base --extra export_dreyer2023_wide_v1 export_scherer2015_v1 --bench 20
#   T2-D109 (CPT v2): ... -- --fm st_large --extra export_dreyer2023_wide_v1 export_scherer2015_v1 cpt_pools/<name>_v1 ... \
#       --steps 15000 --batch 64 --accum 8 --sampling sqrt --token-scale --eval-every 1000
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGE="$S1_JOBTMP/cpt"; mkdir -p "$STAGE"
cp -r "$D/export_proteus_v1" "$STAGE/"
args=(); ext=0
for a in "$@"; do
  if [ "$a" = "--extra" ]; then ext=1; args+=("$a"); continue; fi
  # a sub-path (T2-D109: cpt_pools/<name>_v1) is staged under its basename
  if [ $ext = 1 ] && [[ $a != --* ]]; then cp -r "$D/$a" "$STAGE/" && args+=("$STAGE/$(basename "$a")"); continue; fi
  ext=0; args+=("$a")
done
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_cpt.py" --proteus "$STAGE/export_proteus_v1" --out "$OUT" "${args[@]}" &
PID=$!
trap 'kill -TERM "$PID"' USR1
rc=0; wait "$PID" || rc=$?
# a trapped USR1 returns `wait` at once: wait again while t2p_cpt saves checkpoint-cpt-partial.pth (exit 3; review T2 P3i)
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done
if grep -q -- "--bench" <<< "$*"; then s1_finish "$rc" "$OUT" 'bench.json' 1; fi
s1_finish "$rc" "$OUT" 'checkpoint-cpt.pth' 1
