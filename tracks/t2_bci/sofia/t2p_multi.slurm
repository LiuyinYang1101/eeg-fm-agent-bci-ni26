#!/usr/bin/env bash
# Track 2 PROTEUS: several bench arms in ONE H200 job, run one after another, each with NSHARDS parallel shards.
# $1 = an arms file (repo path; one t2p_bench.py argument line per arm, '#' comments allowed). Same staging, self-test,
# per-job shard logs and OOM retry as t2p_bench.slurm. A line may start with `NSHARDS=<k> ` (review T2 P3j: role-specific
# LoRA / D132 retention arms need 1 per H200; the job default is $NSHARDS, else 3). Artefact gate = every result file the
# arms' arguments produce (t2p_bench --list-outputs), not 1.
#   S1_EXPORT=NSHARDS=3 S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=06:00:00 \
#       --job-name=ni26-t2p-<what> -- tracks/t2_bci/sofia/arms/<file>.txt
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
ARMS="$S1_CODE/$1"; [ -f "$ARMS" ] || { echo "!! no arms file $ARMS"; exit 2; }
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_selftest.py" || s1_finish 3 "$OUT" 'none' 1
STAGE="${S1_JOBTMP:-/tmp}/t2p_export"; mkdir -p "$STAGE"; cp "$EXPORT"/{X.npy,index.csv,meta.json} "$STAGE/"
K0="${NSHARDS:-3}"; rc=0; a=0
EXP="$OUT/expected_${SLURM_JOB_ID}.txt"; : > "$EXP"
trap 'kill $(jobs -p) 2>/dev/null' USR1 TERM
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [ -z "${line// /}" ] && continue
  K="$K0"
  if [[ $line =~ ^[[:space:]]*NSHARDS=([0-9]+)[[:space:]]+(.*)$ ]]; then K="${BASH_REMATCH[1]}"; line="${BASH_REMATCH[2]}"; fi
  a=$((a + 1)); echo "### arm $a ($(date +%T), $K shards): $line"
  # shellcheck disable=SC2086
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_bench.py" --export "$STAGE" --out "$OUT/runs" $line --list-outputs "$EXP" \
    > "$OUT/arm${a}_list_${SLURM_JOB_ID}.log" 2>&1 || { echo "!! arm $a: --list-outputs failed"; rc=5; }
  pids=()
  for i in $(seq 0 $((K-1))); do
    # shellcheck disable=SC2086
    python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_bench.py" --export "$STAGE" --out "$OUT/runs" --shard "$i" --nshards "$K" $line \
      > "$OUT/arm${a}_shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  grep -h "cellBA\|SKIPPED" "$OUT"/arm${a}_shard*_"$SLURM_JOB_ID".log | sort | tail -n 60
  grep -h -E "Traceback|Error" "$OUT"/arm${a}_shard*_"$SLURM_JOB_ID".log | grep -v OutOfMemory | head -5
done < "$ARMS"
n=$(ls "$OUT/runs"/*.npz 2>/dev/null | wc -l)
N=$(sort -u "$EXP" | wc -l)
miss=$(sort -u "$EXP" | while IFS= read -r f; do [ -f "$OUT/runs/$f" ] || echo "$f"; done)
echo "### $n result files; expected $N; missing $(printf '%s' "$miss" | grep -c . || true)"
[ -n "$miss" ] && { printf '%s\n' "$miss" | head -5; [ "$rc" = 0 ] && rc=3; }
s1_finish "$rc" "$OUT" 'runs/*.npz' "$N"
