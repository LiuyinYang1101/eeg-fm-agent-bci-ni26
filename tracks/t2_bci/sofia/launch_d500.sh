#!/usr/bin/env bash
# T2-D500 (returning-side config variants of W14-lite; config only, R3): repack the GATED W14-lite (5504d134...) with the approved W14 template unchanged and
# the per-user copies' r edited (--set ... --set-ref: REF = NEW; binding gates = OLD vs NEW probability change, the independent decode
# recombination, robustness, kit replay; ms / window on the H200 is a separate timing job). Two dose points:
#   m2   = D462's doubling: user_blend.r 0.3333 -> 0.6667, user_blend2.r / user_blend3.r 0.5 -> 1.0
#   m15  = 1.5x (seat L1's least-harm dose): 0.5 / 0.75 / 0.75
#   copy = the copy-only end: all three r = 1.0 (a routed returning user gets its copies only)
#   mem-guard bash tracks/t2_bci/sofia/launch_d500.sh dry|GO m2|m15|copy
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; dose=${2:?m2|m15|copy}; zip=ni26-t2p-repack-t2p_ms_M7_w14lite33-1617836/t2p_ms_M7_w14lite33.zip; name=t2p_ms_M7_w14lite33_r$dose
case "$dose" in m2) R1=0.6667; R23=1.0;; m15) R1=0.5; R23=0.75;; copy) R1=1.0; R23=1.0;; *) exit 2;; esac
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d500.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D500 returning-side variant: W14-lite with the copies' r = $dose ($R1 / $R23 / $R23; config only, R3)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=zen5_vis --time=04:00:00 --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" --template tracks/t2_bci/submission/t2p_submission_w14.py --set "user_blend.r=$R1" --set "user_blend2.r=$R23" --set "user_blend3.r=$R23" --set-ref \
  2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
