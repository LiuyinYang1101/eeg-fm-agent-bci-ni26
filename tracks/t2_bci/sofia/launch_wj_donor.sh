#!/usr/bin/env bash
# THE STACK J's DONORS (G + V = GRL + vp; the manager + Track3 2026-11-20 ≈ 17:0x, before any number) = the W29 donor lines (W25-ALL6's six,
# user-balanced, + --fm-grl <the engagement pick>) + vp 2 in the seed knobs of the five headed seeds 621-625 and their continuations (D582's
# convention; no vp on the MI 620 = J's s0 = G's). SPECULATIVE once BOTH W29's and W30's member sign gates pass; never fetched; cancelled / discarded
# unread if either part fails inclusion or the stack read fails. The lambda is refused unless it equals lambda_pick.json's.
#   MI   fm_0 seed 620 = launch_d529_midonor.sh d2's line (the MI CPT root, mixup 0.4, tilt 0.15, head-less);
#   A    fm_1 621 + the 622 base, B fm_3 623 + the 624 base (CAR), C fm_5 625 (CAR) = launch_d557_donor.sh's COMMON (the d111-all CPT, the D557
#        augmentation, --fm-side dcmean) with the same seed knobs;
#   cont fm_2 622 from A's fm_1.pt and fm_4 624 from B's fm_1.pt (25 epochs, lr x 0.1), afterok A / B = D557's continuation lines.
# Names t2p_wj_donor_*; S1_PURPOSE the owner's line; every job < 100 GPU-h (1 GPU, <= 3 h).
#   mem-guard bash tracks/t2_bci/sofia/launch_wj_donor.sh dry|GO <completed smoke job id> <lambda max>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:-}; LAM=${3:?the lambda max}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
PYX="$HOME/miniconda3/envs/neural_interface_challenge/bin/python"
pk=$("$PYX" -c "import json; print(json.load(open('$HOME/workspace/outputs/neural_interface_challenge/t2_d649/lambda_pick.json'))['lambda_max'])" 2>/dev/null || echo none)
[ "$mode" == dry ] || "$PYX" -c "import sys; sys.exit(0 if abs(float('$pk') - float('$LAM')) < 1e-12 else 1)" 2>/dev/null || { echo "REFUSED: lambda $LAM != the engagement pick $pk"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_wj_donor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sm=(); [ -n "$SMOKE" ] && sm=(S1_SMOKE="$SMOKE")
P="owner 2026-11-20 16:31:32 (one stack): THE STACK J donors (GRL $LAM + vp 2) SPECULATIVE at both member gates; discarded unless the stack J ships"
sub() { env "${dry[@]}" "${sm[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
AUG=(--fm-aug 0.2:30:0.1 --fm-mixup 0.8)
COMMON=(--model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-weight 0.35
        --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-donor --fm-side dcmean "${AUG[@]}" --user-balance --fm-grl "$LAM")
JM=$(sub S1_PURPOSE="$P; donor MI (620)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name=ni26-t2p-wj-donor-mi -- \
  t2p_wj_donor_mi --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 \
  --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 620 --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-preft-d529-mi-root/checkpoint-cpt.pth" \
  --fm-seed-knobs '{"620":{"tilt":0.15}}' --fm-donor --user-balance --fm-grl "$LAM")
[ "$mode" == dry ] && { echo "dry: the MI donor validated; the rest share t2p_final.slurm"; exit 0; }
JA=$(sub S1_PURPOSE="$P; donor A (621, 622 base)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=03:00:00 --job-name=ni26-t2p-wj-donor-a -- \
  t2p_wj_donor_a "${COMMON[@]}" --fm-epochs 100 --fm-seeds 621,622 --fm-seed-knobs '{"621":{"tilt":0.15,"vp":2},"622":{"tilt":0.15,"vp":2}}')
JB=$(sub S1_PURPOSE="$P; donor B (623, 624 base)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=03:00:00 --job-name=ni26-t2p-wj-donor-b -- \
  t2p_wj_donor_b "${COMMON[@]}" --fm-epochs 100 --fm-seeds 623,624 --fm-seed-knobs '{"623":{"car":1,"tilt":0.15,"vp":2},"624":{"car":1,"tilt":0.15,"vp":2}}')
JC=$(sub S1_PURPOSE="$P; donor C (625)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=02:00:00 --job-name=ni26-t2p-wj-donor-c -- \
  t2p_wj_donor_c "${COMMON[@]}" --fm-epochs 100 --fm-seeds 625 --fm-seed-knobs '{"625":{"car":1,"tilt":0.15,"vp":2}}')
J1=$(sub S1_PURPOSE="$P; continuation of 622 (from A)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --dependency="afterok:$JA" \
  --job-name=ni26-t2p-wj-cont-622 -- t2p_wj_cont_622 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 622 --fm-seed-knobs '{"622":{"tilt":0.15,"vp":2}}' \
  --fm-init-zip "$R/ni26-t2p-wj-donor-a-$JA/t2p_wj_donor_a.zip" --fm-init-map 622:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
J2=$(sub S1_PURPOSE="$P; continuation of 624 (from B)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --dependency="afterok:$JB" \
  --job-name=ni26-t2p-wj-cont-624 -- t2p_wj_cont_624 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 624 --fm-seed-knobs '{"624":{"car":1,"tilt":0.15,"vp":2}}' \
  --fm-init-zip "$R/ni26-t2p-wj-donor-b-$JB/t2p_wj_donor_b.zip" --fm-init-map 624:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
echo "WJ donors: MI $JM, A $JA, B $JB, C $JC; continuations 622 $J1 (after A), 624 $J2 (after B)"
