#!/usr/bin/env bash
# T2-D638 W24' build (PREPARED; runs ONLY via launch_w24_d638.sh after rule (b) passes; PREBUILD until the owner's typed W24' line): the base
# (W21, or the W23 PREBUILD when rule (b) picks the session-1 router; sha-pinned) with its six CNN files replaced by the user-balanced donor
# (t2p_final --train all --user-balance, W21's CNN specs; sha-pinned): (1) t2p_w24_swap.py (asserts the donor's CNN specs == the base's; only
# the six files replaced; every other file byte-identical) -> (2) t2p_repack with W21's own template, NO edits = the inert repack (clean
# equality NEW vs the swapped zip, clean_max_dp == 0 asserted; robustness cases) -> (3) K1 = the official kit with W21's D553 / D560 / D569
# line checks.
#   S1_EXPORT="BASE_SHA_EXPECT=<sha>,DONOR_SHA_EXPECT=<sha>[,W24_PREBUILD=1]" S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w24_build.slurm \
#       --partition=zen5_vis --account=vsc --time=02:00:00 --job-name=ni26-t2p-w24-<name> -- <base zip> <donor zip> <name>
#SBATCH --partition=zen5_vis
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
for z in "$BASE" "$DONOR"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
if [ -n "${W24_PREBUILD:-}" ] || [[ "$NAME" == *PREBUILD* ]]; then
  [ "${W24_PREBUILD:-}" = 1 ] && [[ "$NAME" == *_PREBUILD ]] || { echo "!! W24_PREBUILD='${W24_PREBUILD:-}' and NAME '$NAME' disagree"; s1_finish 2 "$OUT" none 1; }
  printf '%s\n' "PREBUILD - NOT APPROVED. $NAME.zip = the base with its six CNN files replaced by the user-balanced donor (T2-D638 W24'), built $(date '+%F %T %Z') after rule (b) passed (the manager's GO)." \
    "It needs the owner's typed W24' line in T2's shell. Never upload it, rename it or link it for upload by hand." \
    "The package config.json is the BASE's byte for byte, so it does NOT say user_balance: the donor (and its user_balance config) is recorded in ${NAME}_swap.zip.swap.json / the inputs sha256 (documentation only; Track3)." \
    "If the donor was built on the LOCAL 5090 (DONOR_LOCAL), its CNN nets were trained with the local torch (2.11) while config.json records the base's 2.9.1: see ${NAME}_swap.zip.swap.json versions_base / versions_donor; the inert repack, K1 and K2 here run under the worker's 2.9.1 (the load-compatibility proof)." \
    "BASE $BASE" "DONOR $DONOR" > "$OUT/README_PREBUILD.txt"
  echo "### PREBUILD (not approved): $OUT/README_PREBUILD.txt written"
fi
zsha_check() {
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
[ -n "${BASE_SHA_EXPECT:-}" ] && [ -n "${DONOR_SHA_EXPECT:-}" ] || { echo "!! BASE_SHA_EXPECT / DONOR_SHA_EXPECT unset"; s1_finish 2 "$OUT" none 1; }
[ "$DONOR_SHA_EXPECT" = record ] && DONOR_SHA_EXPECT=""   # the donor job of THIS chain (afterok): its sha is recorded here and pinned from here on
zsha_check "$BASE" "$BASE_SHA_EXPECT" && zsha_check "$DONOR" "$DONOR_SHA_EXPECT" || s1_finish 2 "$OUT" none 1
sha256sum "$BASE" "$DONOR" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
echo "### $(date +%T) swap"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w24_swap.py" --base "$BASE" --donor "$DONOR" --out "$OUT/${NAME}_swap.zip" || { echo "!! swap failed"; s1_finish 3 "$OUT" none 1; }
echo "### $(date +%T) inert repack (W21's template, no edits)"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py" --zip "$OUT/${NAME}_swap.zip" --out "$OUT/$NAME" --export "$D/export_proteus_v1" \
  --template "$S1_CODE/tracks/t2_bci/submission/t2p_submission_w21lp.py"
rc=$?
rm -rf "$OUT/${NAME}_ref" "$OUT/${NAME}_old" "$OUT/$NAME"
[ $rc = 0 ] || { echo "!! repack failed (rc $rc)"; s1_finish "$rc" "$OUT" "$NAME.repack.json" 1; }
python -c "import json, sys; r = json.load(open(sys.argv[1])); print('### inert repack:', r['clean_windows'], 'windows, differ', r['clean_differ'], 'max dp', r['clean_max_dp']); sys.exit(0 if r['clean_differ'] == 0 and r['clean_max_dp'] == 0.0 else 1)" \
  "$OUT/$NAME.repack.json" || { echo "!! the inert repack is not the swapped zip bitwise"; s1_finish 4 "$OUT" "$NAME.repack.json" 1; }
rm -f "$OUT/${NAME}_swap.zip"
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
