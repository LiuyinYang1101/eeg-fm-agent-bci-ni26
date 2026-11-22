#!/usr/bin/env bash
# T2-D642 (W25) build PREPARATION, SPECULATIVE (the manager / the owner's "all resources", 2026-11-19; discarded unless a member passes the
# screen AND the full-chain set read): W21's six FM members re-trained on all users exactly as their donors were, PLUS --user-balance
# (t2p_final's wub -> fit_eval_fm data['wtr']):
#   MI   fm_0 seed 620 = launch_d529_midonor.sh d2's line (the MI CPT root, mixup 0.4, tilt 0.15, head-less);
#   A    fm_1 621 + the 622 base, B fm_3 623 + the 624 base (CAR), C fm_5 625 (CAR) = launch_d557_donor.sh's COMMON (the d111-all CPT, the D557
#        augmentation, --fm-side dcmean) with the same seed knobs;
#   cont fm_2 622 from A's fm_1.pt and fm_4 624 from B's fm_1.pt (25 epochs, lr x 0.1), afterok A / B = D557's continuation lines.
# Names t2p_d642_donor_*; S1_PURPOSE the owner's line; every job < 100 GPU-h (1 GPU, <= 3 h).
#   mem-guard bash tracks/t2_bci/sofia/launch_d642_donor.sh dry|GO <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:-}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d642_donor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sm=(); [ -n "$SMOKE" ] && sm=(S1_SMOKE="$SMOKE")
P="owner 2026-11-19: target first place, all resources; T2-D642 user-balanced FM (W25), member screen + speculative donors"
sub() { env "${dry[@]}" "${sm[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
AUG=(--fm-aug 0.2:30:0.1 --fm-mixup 0.8)
COMMON=(--model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-weight 0.35
        --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-donor --fm-side dcmean "${AUG[@]}" --user-balance)
JM=$(sub S1_PURPOSE="$P; donor MI (620)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name=ni26-t2p-d642-donor-mi -- \
  t2p_d642_donor_mi --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 \
  --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 620 --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-preft-d529-mi-root/checkpoint-cpt.pth" \
  --fm-seed-knobs '{"620":{"tilt":0.15}}' --fm-donor --user-balance)
[ "$mode" == dry ] && { echo "dry: the MI donor validated; the rest share t2p_final.slurm"; exit 0; }
JA=$(sub S1_PURPOSE="$P; donor A (621, 622 base)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=03:00:00 --job-name=ni26-t2p-d642-donor-a -- \
  t2p_d642_donor_a "${COMMON[@]}" --fm-epochs 100 --fm-seeds 621,622 --fm-seed-knobs '{"621":{"tilt":0.15},"622":{"tilt":0.15}}')
JB=$(sub S1_PURPOSE="$P; donor B (623, 624 base)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=03:00:00 --job-name=ni26-t2p-d642-donor-b -- \
  t2p_d642_donor_b "${COMMON[@]}" --fm-epochs 100 --fm-seeds 623,624 --fm-seed-knobs '{"623":{"car":1,"tilt":0.15},"624":{"car":1,"tilt":0.15}}')
JC=$(sub S1_PURPOSE="$P; donor C (625)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=02:00:00 --job-name=ni26-t2p-d642-donor-c -- \
  t2p_d642_donor_c "${COMMON[@]}" --fm-epochs 100 --fm-seeds 625 --fm-seed-knobs '{"625":{"car":1,"tilt":0.15}}')
J1=$(sub S1_PURPOSE="$P; continuation of 622 (from A)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --dependency="afterok:$JA" \
  --job-name=ni26-t2p-d642-cont-622 -- t2p_d642_cont_622 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 622 --fm-seed-knobs '{"622":{"tilt":0.15}}' \
  --fm-init-zip "$R/ni26-t2p-d642-donor-a-$JA/t2p_d642_donor_a.zip" --fm-init-map 622:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
J2=$(sub S1_PURPOSE="$P; continuation of 624 (from B)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --dependency="afterok:$JB" \
  --job-name=ni26-t2p-d642-cont-624 -- t2p_d642_cont_624 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 624 --fm-seed-knobs '{"624":{"car":1,"tilt":0.15}}' \
  --fm-init-zip "$R/ni26-t2p-d642-donor-b-$JB/t2p_d642_donor_b.zip" --fm-init-map 624:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
echo "D642 donors: MI $JM, A $JA, B $JB, C $JC; continuations 622 $J1 (after A), 624 $J2 (after B)"
