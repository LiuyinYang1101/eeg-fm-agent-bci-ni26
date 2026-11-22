#!/usr/bin/env bash
# T2-D656 (registered 544853f8): a CONFIG-ONLY repack of a final T2 ZIP (= t2p_w23_build.slurm's step with the edits passed in, not a route JSON).
# Runs ONLY via tracks/t2_bci/sofia/launch_d656_lam.sh (PREBUILD: never uploaded, linked or renamed by hand; promotion needs the owner's typed line).
# One repack with W21lp's OWN template (t2p_submission_w21lp.py; --set-ref asserts it is the old package's byte for byte) and the --set edits in
# CFG_SETS (';'-separated dotted.key=JSON), then K1 = the official kit (t2p_kit_replay.sh) with W21's D553 / D560 / D569 line checks.
#   S1_EXPORT="BASE_SHA_EXPECT=<sha>,CFG_SETS=tta.lam=1.0,CFG_PREBUILD=1" S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_cfgonly_build.slurm \
#       --partition=zen5_vis --account=vsc --time=02:00:00 --job-name=ni26-t2p-cfg-<name> -- <base zip> <name>
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
BASE="$1"; NAME="$2"
[ -f "$BASE" ] || { echo "!! no input $BASE"; s1_finish 2 "$OUT" none 1; }
[ -n "${CFG_SETS:-}" ] || { echo "!! CFG_SETS unset"; s1_finish 2 "$OUT" none 1; }
if [ -n "${CFG_PREBUILD:-}" ] || [[ "$NAME" == *PREBUILD* ]]; then   # the flag and the name agree, and the job dir says it is not approved
  [ "${CFG_PREBUILD:-}" = 1 ] && [[ "$NAME" == *_PREBUILD ]] || { echo "!! CFG_PREBUILD='${CFG_PREBUILD:-}' and NAME '$NAME' disagree"; s1_finish 2 "$OUT" none 1; }
  printf '%s\n' "PREBUILD - NOT APPROVED. $NAME.zip = $(basename "$BASE") with the config-only edits '$CFG_SETS' (${CFG_WHAT:-T2-D656, the order hedge}), built $(date '+%F %T %Z')." \
    "Speculative; offered only under its registered rule and never uploaded without the owner's typed line." \
    "Never upload it, rename it or link it for upload by hand. BASE $BASE" > "$OUT/README_PREBUILD.txt"
  echo "### PREBUILD (not approved): $OUT/README_PREBUILD.txt written"
fi
zsha_check() {
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
[ -n "${BASE_SHA_EXPECT:-}" ] || { echo "!! BASE_SHA_EXPECT unset"; s1_finish 2 "$OUT" none 1; }
if [ "$BASE_SHA_EXPECT" = sibling ]; then   # T2-D658: a base built minutes earlier: its own build's .zip.sha256 is the expectation (required)
  [ -f "$BASE.sha256" ] || { echo "!! BASE_SHA_EXPECT=sibling but no $BASE.sha256"; s1_finish 2 "$OUT" none 1; }
  zsha_check "$BASE" "$(cut -d' ' -f1 "$BASE.sha256")" || s1_finish 2 "$OUT" none 1
else
  zsha_check "$BASE" "$BASE_SHA_EXPECT" || s1_finish 2 "$OUT" none 1
fi
sha256sum "$BASE" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"   # = t2p_w23_build.slurm
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w21lp.py"   # W21lp's own template (asserted byte-identical to the package's by --set-ref)
RPK="$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py"
SETARGS=(); IFS=';' read -ra _S <<< "$CFG_SETS"; for s in "${_S[@]}"; do SETARGS+=(--set "$s"); done
echo "### $(date +%T) repack $NAME (config-only: ${SETARGS[*]} --set-ref)"
python -u "$RPK" --zip "$BASE" --out "$OUT/$NAME" --export "$D/export_proteus_v1" --template "$T" "${SETARGS[@]}" --set-ref
rc=$?
rm -rf "$OUT/${NAME}_ref" "$OUT/${NAME}_dref" "$OUT/${NAME}_bref" "$OUT/${NAME}_old" "$OUT/$NAME"     # the check trees (files cap); ZIP + repack.json stay
[ $rc = 0 ] || { echo "!! repack failed (rc $rc)"; s1_finish "$rc" "$OUT" "$NAME.repack.json" 1; }
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
pfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D560 FM proto at" | grep -cvE ": refits 1, .*kept_trained 0, errors 0 \(")   # 'refits 0' = silently off: fails
pok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D560 FM proto at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed|D560 FM proto failed|D569 cnn_lp failed")
lok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D569 cnn_lp at")
lerr=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D569 cnn_lp at" | grep -cvE ", errors 0 \(")
lsm=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -oE "D569 cnn_lp at [0-9]+: smoothed [0-9]+" | awk '{s += $NF} END {print s + 0}')
echo "### K1: $fok D553 FM-refit lines ($ffb with a fallback / error), $pok D560 proto lines ($pfb with a fallback / error), $lok D569 cnn_lp lines ($lsm smoothed, $lerr with errors), $fail failure lines"
[ "$ffb" = 0 ] && [ "$pfb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] && [ "$pok" -gt 0 ] && [ "$lok" -gt 0 ] && [ "$lsm" -gt 0 ] && [ "$lerr" = 0 ] || { echo "!! K1: D553 / D560 / D569 fallbacks / failures / none in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1
