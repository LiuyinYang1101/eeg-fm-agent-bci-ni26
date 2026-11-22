#!/usr/bin/env bash
# T2-D513 (template W15 = W14 + in-stream adaptation; PREPARED 2026-11-15): build a W15 final-stack package by repacking a GATED W14-lite-shape zip
# with the W15 template + cfg instream (tracks/t2_bci/submission/configs/t2_d513_instream.json), --set-ref (a change MEANT to move the outputs:
# NEW serves as REF, OLD vs NEW must differ). The base is a gated W14-shape zip, e.g. the tilt swap's (launch_d512_fmswap.sh). t2p_repack.slurm checks parts / decode / robustness / the kit replay under the worker guard; then the H200 timing job (afterok).
# REFUSES TO RUN unless W15_APPROVED carries the user's own line typed in T2's shell (date + the words).
#   W15_APPROVED="2026-11-15 hh:mm, the user in T2's shell: 'T2: template change approved: ...'" \
#     [INERT=1] mem-guard bash tracks/t2_bci/sofia/launch_d513_w15.sh dry|GO <base zip under runs/ni26> <new name>
# (INERT=1: step A, the template change alone; without it: step B, instream on a W15-template base)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; zip=${2:?base zip under runs/ni26}; name=${3:?new name}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W15_APPROVED:-}" == *"template change approved"* ]] || { echo "W15 is PREPARED, NOT APPROVED: set W15_APPROVED to the user's own line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_d513_w15.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "${INERT:-}" == 1 ]; then    # step A without a member swap: the W15 template alone (no instream) = W14 bitwise; t2p_repack's clean equality proves it
  ex=(--template tracks/t2_bci/submission/t2p_submission_w15.py)
else                               # step B: the base must already carry the W15 template (--set-ref needs it byte for byte)
  ex=(--template tracks/t2_bci/submission/t2p_submission_w15.py --set-ref --set "instream=@tracks/t2_bci/submission/configs/t2_d513_instream.json#instream")
fi
# t2p_repack: --set-ref takes --set edits only (no --fm-from / --fm-keep), so a member swap (the tilt donor) is its OWN earlier repack
# (tracks/t2_bci/sofia/launch_d512_fmswap.sh) and this one starts from its zip
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
out=$(env "${dry[@]}" S1_PURPOSE="T2-D513 W15 final-stack build ($zip -> $name; approved: ${W15_APPROVED:0:60}): feeds T2's final packages" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=${T2_PART:-zen5_vis} --time=${T2_TIME:-04:00:00} --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" "${ex[@]}" 2>&1)
echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
