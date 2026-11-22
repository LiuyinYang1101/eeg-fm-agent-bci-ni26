#!/usr/bin/env bash
# T2-D492: continued masked-patch pretraining of REVE-base / LaBraM-base (t2p_cpt_fm.py) on the pools of t2p_cpt.slurm. Args after `--` go to it
# (--out is set here). Pools are read from data/ni26/t2_proteus/ (staged node-locally).
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
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_cpt_fm.py" --proteus "$STAGE/export_proteus_v1" --out "$OUT" "${args[@]}" &
PID=$!
trap 'kill -TERM "$PID"' USR1
rc=0; wait "$PID" || rc=$?
# a trapped USR1: t2p_cpt_fm saves nothing partial (a killed run is resubmitted)
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done
if grep -q -- "--bench" <<< "$*"; then s1_finish "$rc" "$OUT" 'bench.json' 1; fi
s1_finish "$rc" "$OUT" 'cpt_summary.json' 1
