#!/usr/bin/env bash
# T2-D582 SPECULATIVE DONORS (the manager after the Fable review, 2026-11-18 item e: started NOW at nice 50, before any read; inside GO,
# < 16 GPU-h per job; no build). = launch_d557_donor.sh (the W20+D557 donors: five headed members 621 / 622 plain, 623 / 624 / 625 CAR, d111-all
# CPT, sz, gain, 100 epochs, llrd 0.75, rc 0.75, tilt 0.15, --fm-side dcmean, --fm-aug 0.2:30:0.1, --fm-mixup 0.8; D546b continuations of 622 / 624)
# + vp 2 per seed (the t2p_final seed knob = t2p_bench --fm-vp 2: the same r1 bank). No merge, no build here: W22 needs a NEW owner line.
#   [NICE=50 is the default] mem-guard bash tracks/t2_bci/sofia/launch_d582_donor.sh dry|GO smoke|all [<smoke job id>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?smoke|all}; dep=()
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == smoke || "$part" == all ]] || exit 2
[ -n "${3:-}" ] && dep=(--dependency="afterok:$3")
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d582_donor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
AUG=(--fm-aug 0.2:30:0.1 --fm-mixup 0.8)   # T2-D582: D557's dose (+ vp 2 in each seed's knobs below)
COMMON=(--model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-weight 0.35
        --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-donor --fm-side dcmean "${AUG[@]}")
P="T2-D582 SPECULATIVE all-user donors (the manager, item e: now at nice 50, before the reads; no build: W22 waits for a NEW owner line): the five headed FM members re-trained on all users with D557's dose + vp 2"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
if [ "$part" == smoke ]; then
  sub S1_PURPOSE="$P; SMOKE (621 + 623, 1 epoch)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=00:45:00 --job-name=ni26-t2p-d582-donor-smoke -- \
    t2p_d582_donor_smoke "${COMMON[@]}" --fm-epochs 1 --fm-seeds 621,623 --fm-seed-knobs '{"621":{"tilt":0.15,"vp":2},"623":{"car":1,"tilt":0.15,"vp":2}}' >/dev/null
  exit 0
fi
JA=$(sub S1_PURPOSE="$P; job A (621, 622)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=03:00:00 "${dep[@]}" --job-name=ni26-t2p-d582-donor-a -- \
  t2p_d582_donor_a "${COMMON[@]}" --fm-epochs 100 --fm-seeds 621,622 --fm-seed-knobs '{"621":{"tilt":0.15,"vp":2},"622":{"tilt":0.15,"vp":2}}')
JB=$(sub S1_PURPOSE="$P; job B (623, 624)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=03:00:00 "${dep[@]}" --job-name=ni26-t2p-d582-donor-b -- \
  t2p_d582_donor_b "${COMMON[@]}" --fm-epochs 100 --fm-seeds 623,624 --fm-seed-knobs '{"623":{"car":1,"tilt":0.15,"vp":2},"624":{"car":1,"tilt":0.15,"vp":2}}')
JC=$(sub S1_PURPOSE="$P; job C (625)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=02:00:00 "${dep[@]}" --job-name=ni26-t2p-d582-donor-c -- \
  t2p_d582_donor_c "${COMMON[@]}" --fm-epochs 100 --fm-seeds 625 --fm-seed-knobs '{"625":{"car":1,"tilt":0.15,"vp":2}}')
[ "$mode" == dry ] && { echo "dry: the three donor jobs validated; the continuations need their ids"; exit 0; }
J1=$(sub S1_PURPOSE="$P; D546b continuation of 622 (from A)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=01:30:00 --dependency="afterok:$JA" \
  --job-name=ni26-t2p-d582-cont-622 -- t2p_d582_cont_622 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 622 --fm-seed-knobs '{"622":{"tilt":0.15,"vp":2}}' \
  --fm-init-zip "$R/ni26-t2p-d582-donor-a-$JA/t2p_d582_donor_a.zip" --fm-init-map 622:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
J2=$(sub S1_PURPOSE="$P; D546b continuation of 624 (from B)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=01:30:00 --dependency="afterok:$JB" \
  --job-name=ni26-t2p-d582-cont-624 -- t2p_d582_cont_624 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 624 --fm-seed-knobs '{"624":{"car":1,"tilt":0.15,"vp":2}}' \
  --fm-init-zip "$R/ni26-t2p-d582-donor-b-$JB/t2p_d582_donor_b.zip" --fm-init-map 624:fm_1.pt --fm-cont-epochs 25 --fm-cont-lr 0.1)
echo "D582 donors: A $JA, B $JB, C $JC; continuations 622 $J1 (after A), 624 $J2 (after B)"
