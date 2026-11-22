#!/usr/bin/env bash
# T2-D482 (template W14 = W13 + a per-member input reference; PREPARED 2026-11-14, NOT APPROVED): build a W14 package by repacking a GATED
# W13 zip with the W14 template, APPENDING three CAR ST-L members from the T2-D475e donor (seeds 600-602, trained on all users; their car flag
# and CAR alignment scale kept; equal-weight mean over the 5 + 3 members = 0.35 / 8 each), and the fm family's start vector for that mix
# (tracks/t2_bci/submission/configs/t2_d482_fm_start_vector.json, T2-D459's rule on the 5 + 3 mix). t2p_repack.slurm on zen5_vis checks:
# OLD vs REF (W14, old config) on every public window (the template is inert without a ref); the CNN parts = REF's; the merged FM part =
# (5 x REF's + 3 x cref's) / 8 with cref = a package holding only the added members (<= 1e-6); the output = the per-family reference decode
# (<= 1e-6); robustness cases; the kit replay under the worker guard; ms / window.
# REFUSES TO RUN unless W14_APPROVED carries the user's own line typed in T2's shell (date + the words), e.g.
#   W14_APPROVED="2026-11-1x hh:mm, the user in T2's shell: 'T2: template change approved: ...'" \
#     mem-guard bash tracks/t2_bci/sofia/launch_d482_w14.sh dry|GO <W13 zip under runs/ni26> <new name>
#   e.g. ... GO ni26-t2p-repack-t2p_ms_M7_w13-1615509/t2p_ms_M7_w13.zip t2p_ms_M7_w14
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; zip=${2:?W13 zip under runs/ni26}; name=${3:?new name}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W14_APPROVED:-}" == *"template change approved"* ]] || { echo "W14 is PREPARED, NOT APPROVED: set W14_APPROVED to the user's own line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d482_w14.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
DONOR=$R/ni26-t2p-d475-car5donor-1615573/t2p_d475_car5donor.zip
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D482 W14 build ($zip + 3 CAR donor members -> $name; approved: ${W14_APPROVED:0:60}): feeds T2's upload B" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=zen5_vis --time=04:00:00 --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" --template tracks/t2_bci/submission/t2p_submission_w14.py --fm-add-from "$DONOR" --fm-add-seeds 600,601,602 \
  --set "tta.per_family.fm=@tracks/t2_bci/submission/configs/t2_d482_fm_start_vector.json#fm" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
