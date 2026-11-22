#!/usr/bin/env bash
# T2-D580 in a W22 (night plan 2026-11-18; runs ONLY via launch_w22.sh, which refuses without the owner's typed W22 line): ONE repack on ONE GPU,
#   RS  t2p_repack --zip <base W21-template zip> --template t2p_submission_w21lp.py --side-replace <side.pt (smx4, seeds 51700-51709)>
#       (its own asserts: the same nets / keys / shapes / prior / scale / seeds, another recipe; every other entry byte-identical; the parts gate:
#       NEW's FBCNet and FM parts == REF's on every public window, only the side families move)
#   K1  the official kit (t2p_kit_replay.sh) on <name>.zip, refit ON: the same D553 / D560 / D569 line counts as t2p_w21lp_build.slurm's K1.
#   S1_EXPORT=BASE_SHA_EXPECT=<sha>|sibling S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w22_side.slurm --time=03:00:00 \
#       --job-name=ni26-t2p-w22side-<name> -- <base zip> <side.pt> <name>
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
BASE="$1"; SIDE="$2"; NAME="$3"
for z in "$BASE" "$SIDE"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
zsha_check() {   # T2-D569 F4 (Track3): print AND check the sha256 of an input (vs its sibling .sha256 when present, vs an expected value when given)
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
if [ "${BASE_SHA_EXPECT:-}" = sibling ]; then
  [ -f "$BASE.sha256" ] || { echo "!! BASE_SHA_EXPECT=sibling but no $BASE.sha256"; s1_finish 2 "$OUT" none 1; }
  BASE_SHA_EXPECT=$(cut -d' ' -f1 "$BASE.sha256")
fi
[ -n "${BASE_SHA_EXPECT:-}" ] || { echo "!! BASE_SHA_EXPECT unset"; s1_finish 2 "$OUT" none 1; }
zsha_check "$BASE" "$BASE_SHA_EXPECT" && zsha_check "$SIDE" "${SIDE_SHA_EXPECT:-}" || s1_finish 2 "$OUT" none 1
sha256sum "$BASE" "$SIDE" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w21lp.py"
echo "### $(date +%T) RS $NAME"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py" --zip "$BASE" --out "$OUT/$NAME" --export "$D/export_proteus_v1" --template "$T" --side-replace "$SIDE"
rc=$?
rm -rf "$OUT/${NAME}_ref" "$OUT/${NAME}_dref" "$OUT/${NAME}_bref" "$OUT/${NAME}_old" "$OUT/$NAME"
[ $rc = 0 ] || { echo "!! RS failed (rc $rc)"; s1_finish "$rc" "$OUT" "$NAME.repack.json" 1; }
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
pfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D560 FM proto at" | grep -cvE ": refits 1, .*kept_trained 0, errors 0 \(")
pok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D560 FM proto at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed|D560 FM proto failed|D569 cnn_lp failed")
lok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D569 cnn_lp at")
lerr=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D569 cnn_lp at" | grep -cvE ", errors 0 \(")
lsm=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -oE "D569 cnn_lp at [0-9]+: smoothed [0-9]+" | awk '{s += $NF} END {print s + 0}')
echo "### K1: $fok D553 FM-refit lines ($ffb with a fallback / error), $pok D560 proto lines ($pfb with a fallback / error), $lok D569 cnn_lp lines ($lsm smoothed, $lerr with errors), $fail failure lines"
[ "$ffb" = 0 ] && [ "$pfb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] && [ "$pok" -gt 0 ] && [ "$lok" -gt 0 ] && [ "$lsm" -gt 0 ] && [ "$lerr" = 0 ] || { echo "!! K1: D553 / D560 / D569 fallbacks / failures / none in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1
