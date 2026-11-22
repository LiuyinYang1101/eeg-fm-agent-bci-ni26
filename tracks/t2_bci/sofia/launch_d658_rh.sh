#!/usr/bin/env bash
# T2-D658 (registered 0330189f): RH = W30's one-hop ZIP + {fm_weight 0.55, cnn2.weight / cnn3.weight x 0.45 / 0.65} (Track3's pins: THREE keys; the
# main CNN family is the template's remainder; the check asserts the implied shares). A SECOND, parallel path: it must not touch or delay plain W30,
# so every job runs on the H200 at --nice 500 with --exclude acc012, behind W30's own H200 timing. PREBUILD ONLY (the owner's line is separate).
#   (1) t2p_cfgonly_build.slurm afterok W30's one-hop build: BASE_SHA_EXPECT=sibling (the W30 build's own .zip.sha256), the three --set, --set-ref, K1
#   (2) t2p_pkgcheck.slurm d656 (D656_EXPECT_SET = the three keys, D658_SHARE_SCALE 0.45/0.65) vs the W30 ZIP: + the PAIRED ms / window = the timing gate
#   (3) K2. No separate H200 timing job: the registered paired fallback (a weight change cannot alter the compute path).
#   mem-guard bash tracks/t2_bci/sofia/launch_d658_rh.sh dry|PREBUILD
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILD}; [[ "$mode" == dry || "$mode" == PREBUILD ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_cfgonly_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/launch_d658_rh.sh)" ] || { echo "commit first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
J1W=1649496; [ -f $O/t2_d652/onehop_ids.env ] && J1W=$(source $O/t2_d652/onehop_ids.env; echo $J1)   # W30's one-hop build (a relaunch rewrites it)
W30_REL=ni26-t2p-w30oh-t2p_w30oh_PREBUILD-$J1W/t2p_w30oh_PREBUILD.zip
R=$S1_ROOT/runs/ni26
SETS="fm_weight=0.55;cnn2.weight=0.1631868131868132;cnn3.weight=0.06923076923076923"   # 0.23571428571428574 x 0.45/0.65; 0.1 x 0.45/0.65 (repr)
N=t2p_rh_PREBUILD
P="T2-D658 RH PREBUILD = W30 one-hop + fm_weight 0.55 (CNN x 0.45/0.65; config only): the owner's risky alternative, NOT approved; never delays plain W30"
H=(--partition=zen4_h200 --nice=500 --exclude=acc012)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == PREBUILD ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee -a "${LAUNCH_LOG:-/dev/null}" | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
echo "### BASE $W30_REL (afterok $J1W) -> $N ($SETS)"
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=sibling,CFG_SETS=$SETS,CFG_PREBUILD=1,CFG_WHAT=T2-D658_RH" S1_PURPOSE="$P: (1) the config-only repack + K1" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_cfgonly_build.slurm "${H[@]}" --time=01:30:00 --dependency="afterok:$J1W" --job-name="ni26-t2p-cfg-$N" -- "$R/$W30_REL" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-cfg-$N-$J1/$N.zip"
J2=$(sub S1_EXPORT="D656_EXPECT_SET=$SETS,D658_SHARE_SCALE=0.45/0.65" S1_PURPOSE="$P: (2) the config-only check vs W30 (exact keys + implied shares) + the paired timing" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${H[@]}" --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- d656 "$W30_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${H[@]}" --time=01:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-rhkit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
echo "D658 RH PREBUILD (NOT APPROVED): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; the zip runs/ni26/$Z_REL"
