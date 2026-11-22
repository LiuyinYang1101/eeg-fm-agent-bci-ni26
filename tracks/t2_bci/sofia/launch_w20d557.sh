#!/usr/bin/env bash
# T2-D557 -> W20+D557 (W20's template UNCHANGED, donors only: the manager 2026-11-18; the owner's W20 line covers "D557's augmented donors only
# if both D557 reads pass"). REFUSES without W20_APPROVED (the owner's W20 line) and D557_READS_PASSED=1 (D557-2 + the W20 interaction read,
# recorded in DECISIONS first). Chain: (1) the W19-style build on the H200 (t2p_w19_build.slurm: R1 --fm-from the merged augmented donor on
# W18 b2, R2 S10, R3 side refit, R4 fm_side_refit, K1) -> (2) the W20 repacks on zen5_vis (t2p_w20_build.slurm on (1)'s zip: R5a, R5b fm_proto,
# K1) -> (3) the post-build check vs (1) (t2p_w20_pkgcheck, zen5_vis) -> (4) K2 (zen5_vis); (5) the H200 timing afterok (2).
#   W20_APPROVED="..." D557_READS_PASSED=1 mem-guard bash tracks/t2_bci/sofia/launch_w20d557.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W20_APPROVED:-}" == *"template change approved"* && "${W20_APPROVED:-}" == *"W20"* && "${W20_APPROVED:-}" == *"D557"* ]] || { echo "set W20_APPROVED to the owner's W20 line (it names D557's donors)"; exit 3; }
[ "${D557_READS_PASSED:-}" = 1 ] || { echo "D557-2 and the W20 interaction read must both have passed (record them first)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w20d557.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
W18=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip; DONOR=$R/ni26-t2p-d557-merge/t2p_w20d557_donor.zip; SIDE=$R/t2_d549_side10/side.pt
N1=t2p_fin2_w19d557; N2=t2p_fin2_w20d557
P="T2-D557 W20+D557 (W20's template unchanged, the augmented donors; the owner's W20 line) for gating"
V=(--partition=zen5_vis --account=vsc)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
J1=$(sub S1_PURPOSE="$P: (1) the W19-style build with the merged augmented donor (R1-R4 + K1, H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm \
  --time=05:00:00 --job-name="ni26-t2p-w19-$N1" -- "$W18" "$DONOR" "$SIDE" "$N1")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z1="$R/ni26-t2p-w19-$N1-$J1/$N1.zip"; Z1_REL="ni26-t2p-w19-$N1-$J1/$N1.zip"
J2=$(sub S1_PURPOSE="$P: (2) the W20 repacks R5a / R5b + K1 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm "${V[@]}" --time=05:00:00 \
  --dependency="afterok:$J1" --job-name="ni26-t2p-w20-$N2" -- "$Z1" "$N2")
Z2_REL="ni26-t2p-w20-$N2-$J2/$N2.zip"
J3=$(sub S1_PURPOSE="$P: (3) the post-build check vs (1) (gates I / H / HF / HP / B / S; zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" \
  --time=08:00:00 --dependency="afterok:$J2" --job-name="ni26-t2p-pkgcheck-$N2" -- w20 "$Z1_REL" "$Z2_REL" "$N2")
J4=$(sub S1_PURPOSE="$P: (4) condition K2 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=03:00:00 \
  --dependency="afterok:$J3" --job-name="ni26-t2p-w20kit-$N2" -- "$Z2_REL" "ni26-t2p-pkgcheck-$N2-$J3/$N2.pkgcheck.pred.npz" "$N2")
J5=$(sub S1_PURPOSE="$P: (5) the H200 timing (3 reps), the D517 line <= 3,150 s for the -lite rule" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-time-$N2" -- "$Z2_REL" "$N2" 3)
echo "W20+D557: (1) $J1 -> (2) $J2 -> (3) check $J3 -> (4) K2 $J4; (5) timing $J5; the zip runs/ni26/$Z2_REL"
