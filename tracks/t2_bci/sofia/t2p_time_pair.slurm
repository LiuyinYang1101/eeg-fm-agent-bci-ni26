#!/usr/bin/env bash
# T2-D634 (astra's plan review via the manager, 2026-11-19): ONE more H200 timing of the gated W23 PREBUILD zip, with W21's zip timed in the
# SAME allocation back to back (A = W23, then B = W21, then A again), t2p_time_zip.py's 3-rep protocol each (= t2p_time_zip.slurm without
# the kit replay). Documents the run-to-run variability only; the registered gate stays (4)'s mean (no new gate unless a mean > 3,300 s).
#   S1_EXPORT="A_SHA=<sha>,B_SHA=<sha>" S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_time_pair.slurm --time=01:00:00 \
#       --job-name=ni26-t2p-timepair-<name> -- <zip A under runs/ni26> <zip B under runs/ni26> <name>
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
D="$S1_ROOT/data/ni26/t2_proteus"
ZA="$S1_ROOT/runs/ni26/${1:?zip A}"; ZB="$S1_ROOT/runs/ni26/${2:?zip B}"; NAME=${3:?name}
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
zsha_check() {   # = t2p_time_zip.slurm's
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
[ -n "${A_SHA:-}" ] && [ -n "${B_SHA:-}" ] || { echo "!! A_SHA / B_SHA unset"; s1_finish 2 "$OUT" none 1; }
zsha_check "$ZA" "$A_SHA" && zsha_check "$ZB" "$B_SHA" || s1_finish 2 "$OUT" none 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for step in A1:"$ZA" B:"$ZB" A2:"$ZA"; do
  tag=${step%%:*}; z=${step#*:}
  echo "### $(date +%T) timing $tag $(basename "$z")"
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_time_zip.py" --zip "$z" --export "$D/export_proteus_v1" --out "$OUT/${NAME}_$tag" --reps 3 || rc=5
done
s1_finish $rc "$OUT" "${NAME}_A2.json" 1
