#!/usr/bin/env bash
# T2-D352: re-time a packaged T2 ZIP --reps times on one H200 (t2p_time_zip.py = t2p_repack's own measurement), then the official kit
# warm-up replay once (t2p_kit_replay.sh: its wall time).
# T2-D381: S1_EXPORT=TIME_SET='cnn_tta_vp.copies=false' times a config variant (no kit replay then).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=00:45:00 --nice=300 --job-name=... -- <zip under runs/ni26> <name> [reps]
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
ZIP="$S1_ROOT/runs/ni26/${1:?zip under runs/ni26}"; NAME=${2:?name}; REPS=${3:-3}
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
SETS=()                                           # T2-D381: S1_EXPORT=TIME_SET='k=v;k2=v2' = a scratch timing of a config variant
IFS=';' read -ra _kv <<< "${TIME_SET:-}"; for kv in "${_kv[@]}"; do [ -n "$kv" ] && SETS+=(--set "$kv"); done
zsha_check() {   # T2-D569 F4 (Track3): print AND check the sha256 of a ZIP this job loads (vs its sibling .sha256 when present, vs an expected value when given)
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
[ -f "$ZIP" ] && zsha_check "$ZIP" "${ZIP_SHA_EXPECT:-}" || { echo "!! timing input"; s1_finish 2 "$OUT" none 1; }
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_time_zip.py" --zip "$ZIP" --export "$D/export_proteus_v1" --out "$OUT/$NAME" --reps "$REPS" "${SETS[@]}"; rc=$?
# the kit replay runs the ZIP as packaged, so only without TIME_SET (a variant's replay would time the unpatched ZIP)
[ $rc -eq 0 ] && [ ${#SETS[@]} -eq 0 ] && { bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$ZIP" "$OUT/$NAME"; rc=$?; }
s1_finish $rc "$OUT" "$NAME.json" 1
