#!/usr/bin/env bash
# T2-D512 final stack, step A (part P2): repack a GATED W14-shape zip with its FM members REPLACED by a tilt donor's (t2p_final --fm-donor:
# 3 plain + 3 CAR ST-L, tilt 0.15, all users; launch_d512_tiltdonor.sh), the W15 template WITHOUT cfg instream (= W14's arithmetic bitwise, so
# OLD vs REF stays inert; step B then sets instream with --set-ref, which needs the template unchanged), and the tilt mix's fm start vector
# (tracks/t2_bci/submission/configs/t2_d512_fm_start_vector_tilt.json). t2p_repack.slurm: OLD vs REF inert, parts (CNN = OLD's, FM = the donor's),
# per-family decode, robustness, kit replay. Step B (W15) starts from this zip (launch_d513_w15.sh).
# REFUSES TO RUN unless W15_APPROVED carries the user's own line typed in T2's shell (the final-stack line covers the parts D512 keeps).
#   W15_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_d512_fmswap.sh dry|GO <base zip under runs/ni26> <donor zip under runs/ni26> <new name>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; zip=${2:?base zip}; donor=${3:?donor zip}; name=${4:?new name}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W15_APPROVED:-}" == *"template change approved"* ]] || { echo "the final stack is PREPARED, NOT APPROVED: set W15_APPROVED to the user's own line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d512_fmswap.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D512 final stack step A ($zip, FM from $donor -> $name; approved: ${W15_APPROVED:0:60}): feeds T2's final packages" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=${T2_PART:-zen5_vis} --time=${T2_TIME:-04:00:00} --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" --template tracks/t2_bci/submission/t2p_submission_w15.py --fm-from "$R/$donor" \
  --set "tta.per_family.fm=@tracks/t2_bci/submission/configs/t2_d512_fm_start_vector_tilt.json#fm" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
