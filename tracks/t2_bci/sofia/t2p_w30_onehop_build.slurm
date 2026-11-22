#!/usr/bin/env bash
# T2-D652 (W30) ONE-HOP build (the manager's amendment 2026-11-21 17:31 + GO 17:4x): W30 = W25-ALL6 with its FM members swapped for the merged W30
# donor's, in ONE repack (= the tested chain's output: W25-ALL6 vs W21 differ in fm_members ONLY, and every hop after R1 is a template / config step;
# the FM-dependent refits run in-stream). Runs ONLY via tracks/t2_bci/sofia/launch_w30_onehop.sh (PREBUILD; promotion needs the owner's typed line).
# One t2p_repack --fm-from <merged W30 donor> on W25-ALL6 with W21lp's OWN template (the tested R1 form: --fm-from, no --set-ref), then K1 = the
# official kit with W21's D553 / D560 / D569 line checks.
#   S1_EXPORT="BASE_SHA_EXPECT=<W25-ALL6 sha>,W30_PREBUILD=1" S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w30_onehop_build.slurm \
#       --partition=zen5_vis --account=vsc --time=02:00:00 --job-name=ni26-t2p-w30oh-<name> -- <W25-ALL6 zip> <merged donor zip> <name>
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
BASE="$1"; DONOR="$2"; NAME="$3"
for f in "$BASE" "$DONOR"; do [ -f "$f" ] || { echo "!! no input $f"; s1_finish 2 "$OUT" none 1; }; done
if [ -n "${W30_PREBUILD:-}" ] || [[ "$NAME" == *PREBUILD* ]]; then   # the flag and the name agree, and the job dir says it is not approved
  [ "${W30_PREBUILD:-}" = 1 ] && [[ "$NAME" == *_PREBUILD ]] || { echo "!! W30_PREBUILD='${W30_PREBUILD:-}' and NAME '$NAME' disagree"; s1_finish 2 "$OUT" none 1; }
  printf '%s\n' "PREBUILD - NOT APPROVED. $NAME.zip = $(basename "$BASE") with its FM members from $(basename "$DONOR") (T2-D652 W30 one-hop), built $(date '+%F %T %Z')." \
    "The manager's amendment 2026-11-21 17:31 (the owner's extra T2 upload); offered only if every gate + Track3's review pass by 21:45 and the owner types the W30 line." \
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
zsha_check "$BASE" "$BASE_SHA_EXPECT" || s1_finish 2 "$OUT" none 1
sha256sum "$BASE" "$DONOR" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"   # = t2p_w23_build.slurm
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w21lp.py"   # W21lp's own template (asserted byte-identical to the package's by --set-ref)
RPK="$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py"
echo "### $(date +%T) repack $NAME (one hop: --fm-from $DONOR)"
python -u "$RPK" --zip "$BASE" --out "$OUT/$NAME" --export "$D/export_proteus_v1" --template "$T" --fm-from "$DONOR"
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
