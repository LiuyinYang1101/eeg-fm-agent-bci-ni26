#!/usr/bin/env bash
# T2-D656 (registered 544853f8): the ORDER HEDGE PREBUILD = W25-ALL6 with the decode lambda tta.lam = <LAM> (config only). SPECULATIVE (the manager's
# 2026-11-21 16:2x plan on the owner's 'any risky attempt'): started at the registration, not waiting for the read; discarded unless the D656 read
# picks <LAM>. NO GO MODE HERE: promotion / an upload link needs the owner's typed line (a separate step; nothing is linked or uploaded by this).
#   (1) t2p_cfgonly_build.slurm on zen5_vis: the W25-ALL6 sha checked on sofia; one repack --set tta.lam=<LAM> --set-ref + K1
#   (2) t2p_pkgcheck.slurm d656 on zen5_vis afterok (1): config diff == {tta.lam}, files equal, finite + rerun-equal on 14 streams, PAIRED ms / window
#   (3) K2 (t2p_w19_kitfull.slurm) on zen5_vis afterok (2): the kit's predict_streams == (2)'s NEW predictions
#   (4) the H200 timing afterok (1) (3 reps; the D517 line <= 3,300 s). Fallback (registered): no H200 start by 18:45 -> (2)'s paired timing
#       within +-2 % -> W25-ALL6's 3,242.6 s.
#   mem-guard bash tracks/t2_bci/sofia/launch_d656_lam.sh dry|PREBUILD <LAM>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILD}; LAM=${2:?lambda}; [[ "$mode" == dry || "$mode" == PREBUILD ]] || { echo "dry|PREBUILD only (no GO: the owner's line is a separate step)"; exit 2; }
[[ "$LAM" == 1.0 || "$LAM" == 0.98 ]] || { echo "LAM must be 1.0 or 0.98 (the registered candidates)"; exit 2; }
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_cfgonly_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_d656_lam.sh)" ] || { echo "commit first"; exit 2; }
W25_SHA=2b03eb037d3ac4962ec5d54e146364c1a5fc6bacb57fcf976b55614b9714ff1a
W25_REL=ni26-t2p-w21lp-t2p_fin2_w25all6_PREBUILD-1644829/t2p_fin2_w25all6_PREBUILD.zip   # the bytes of the promoted W25-ALL6 (2b03eb03); the local t2p_fin2_w25all6.zip is a symlink to it
LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/$W25_REL
h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of W25-ALL6: $h"; [ "$h" = "$W25_SHA" ] || { echo "!! the local W25-ALL6 copy's sha256 != $W25_SHA"; exit 4; }
R=$S1_ROOT/runs/ni26
BASE=$R/$W25_REL
L3=${LAM/./}                                            # 1.0 -> 10, 0.98 -> 098
N=t2p_w25_lam${L3}_PREBUILD
P="T2-D656 the order hedge PREBUILD = W25-ALL6 + tta.lam ${LAM} (config only; speculative, the manager's plan on the owner's 'any risky attempt'): not approved, never uploaded without the owner's typed line"
V=(--partition=zen5_vis --account=vsc)
VH=(--partition=zen4_h200)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == PREBUILD ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
echo "### BASE $W25_REL (sha $W25_SHA) -> $N (tta.lam ${LAM})"
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$W25_SHA,CFG_SETS=tta.lam=$LAM,CFG_PREBUILD=1" S1_PURPOSE="$P: (1) the config-only repack (--set-ref) + K1 (zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_cfgonly_build.slurm "${V[@]}" --time=02:00:00 --job-name="ni26-t2p-cfg-$N" -- "$BASE" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-cfg-$N-$J1/$N.zip"
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  "${VH[@]}" --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$W25_SHA,D656_EXPECT_LAM=$LAM" S1_PURPOSE="$P: (2) the config-only package check vs W25-ALL6 + the paired timing (zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=02:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- d656 "$W25_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=02:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-d656kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
echo "D656 lambda ${LAM} PREBUILD (NOT APPROVED): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL"
