#!/usr/bin/env bash
# Track 2 PROTEUS bench on one H200: (0) the kit-contract export if it is missing (zen5_vis is often blocked by the
# per-user CPU cap; the export takes minutes), (1) the aligner self-test, (2) K shards of t2p_bench.py in parallel on
# the one GPU (small CNNs: one shard cannot fill an H200). Args after `--` go to t2p_bench.py.
#   S1_EXPORT=NSHARDS=6 S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_bench.slurm --time=02:00:00 \
#       --job-name=ni26-t2p-<what> -- --protocols loso,xsess,kit --models eegnet --aligns none,wz,sz,ea --seeds 0
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
EXPORT="${T2P_EXPORT:-$D/export_proteus_v1}"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BENCHOPT_DATA_HOME"
if [ ! -e "$EXPORT/meta.json" ]; then
  echo "### export missing -> running t2p_export.py ($(date +%T))"
  OMP_NUM_THREADS=24 python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_export.py" --out "$EXPORT" || s1_finish 3 "$OUT" 'none' 1
  mkdir -p "$OUT/export"; cp "$EXPORT"/{meta.json,audit.json,index.csv} "$OUT/export/" 2>/dev/null
  cp -r "$EXPORT/bids" "$OUT/export/" 2>/dev/null
fi
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_selftest.py" || s1_finish 3 "$OUT" 'none' 1
STAGE="${S1_JOBTMP:-/tmp}/t2p_export"; mkdir -p "$STAGE"; cp "$EXPORT"/{X.npy,index.csv,meta.json} "$STAGE/"
K="${NSHARDS:-6}"; pids=()
for i in $(seq 0 $((K-1))); do
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_bench.py" --export "$STAGE" --out "$OUT/runs" --shard "$i" --nshards "$K" "$@" \
    > "$OUT/shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' USR1 TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
grep -h "cellBA" "$OUT"/shard*_"$SLURM_JOB_ID".log | sort | tail -n 400
grep -h -E "Traceback|Error" "$OUT"/shard*_"$SLURM_JOB_ID".log | head -20
n=$(ls "$OUT/runs"/*.npz 2>/dev/null | wc -l); echo "### $n result files"
s1_finish "$rc" "$OUT" 'runs/*.npz' 1
