#!/usr/bin/env bash
# T2 P1 student-rep ensemble: the P1 fine-tune seconds on the H200 (the D517 line's hardware) BEFORE any number; W21 as shipped, public streams.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_reptime.slurm --time=00:30:00 --job-name=ni26-t2p-reptime -- <W21 zip under runs/ni26> <streams>
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
OUT=$(s1_outdir)
Z="$S1_ROOT/runs/ni26/${1:?zip}"; N=${2:-6}
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
export TMPDIR="$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u "$S1_CODE/scripts/local/t2_rep_timing.py" --zip "$Z" --streams "$N" --export "$D/export_proteus_v1" 2>&1 | tee "$OUT/reptime.log"
rc=${PIPESTATUS[0]}
grep "^REPTIME" "$OUT/reptime.log" > "$OUT/reptime.json.txt"
s1_finish "$rc" "$OUT" "reptime.json.txt" 1
