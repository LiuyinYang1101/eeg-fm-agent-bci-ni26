#!/usr/bin/env bash
# T2-D475e build preparation (the manager's asks, 2026-11-14; weights / timing only, no template change, NOT an adoption):
#   time  : t2p_fm_stack_time.slurm on M7-W13 (gated): ms / window with +0 / +2 / +3 / +5 ST-L copies (stand-ins for CAR members), H200
#   donor : t2p_final.slurm --fm-donor: five CAR ST-L (seeds 600-604; M7's FM recipe: st_large, sz, gain, mixup 0.4, llrd 0.75, rc 0.75,
#           100 epochs, the d111-all CPT) trained on ALL users + a 1-epoch FBCNet placeholder; a donor of fm files for W14 IF the user
#           approves it; the package itself is never a candidate (DONOR_NOT_A_CANDIDATE marker; W13 feeds car members plain input)
#   bash tracks/t2_bci/sofia/launch_d475_prep.sh time|time2|donor dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
what=${1:?time|time2|timew14|donor|donor3c}; mode=${2:?dry|GO}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia src/ni26)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh   # the file cap: a new snapshot only when T2 code changed
case "$what" in
  time)
    env "${dry[@]}" S1_PURPOSE="T2-D475e runtime MEASUREMENT (the manager's ask): M7-W13 with +0/+2/+3/+5 ST-L copies (CAR stand-ins), ms / window and implied sealed s; feeds the user's runtime-bar call for W14" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_fm_stack_time.slurm --time=01:30:00 --job-name=ni26-t2p-stacktime-M7w13 -- \
      "$R/ni26-t2p-repack-t2p_ms_M7_w13-1615509/t2p_ms_M7_w13.zip" t2p_M7w13_stack --dups 0,2,3,5 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200 ;;
  time2)   # T2-D475g: CNN seeds per family x extra ST-L (k:n = k copies, first n CNN seeds per family; 0 = all)
    env "${dry[@]}" S1_PURPOSE="T2-D475g runtime MEASUREMENT (the manager's ask): M7-W13 with 15 vs 10 CNN seeds per family x +0/+3/+5 ST-L; feeds the 5+3 vs 10-per-family+5+5 choice for W14" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_fm_stack_time.slurm --time=02:00:00 --job-name=ni26-t2p-stacktime2-M7w13 -- \
      "$R/ni26-t2p-repack-t2p_ms_M7_w13-1615509/t2p_ms_M7_w13.zip" t2p_M7w13_stack2 --variants 0:0,3:0,5:0,0:10,3:10,5:10 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200 ;;
  donor)
    KN='{"600":{"car":1},"601":{"car":1},"602":{"car":1},"603":{"car":1},"604":{"car":1}}'
    env "${dry[@]}" S1_PURPOSE="T2-D475e build PREPARATION (the manager's ask; not an adoption): five CAR ST-L (seeds 600-604, M7's FM recipe + car) on all users, a donor of fm files for W14 if the user approves; never a candidate" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=06:00:00 --job-name=ni26-t2p-d475-car5donor -- \
      t2p_d475_car5donor --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
      --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 600,601,602,603,604 \
      --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor \
      2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200 ;;
  timew14)   # T2-D482's runtime gate (the user's bar 52 ms, H200 basis): the gated M7-W14 package itself, as built, no copies
    env "${dry[@]}" S1_PURPOSE="T2-D482 runtime GATE: the built M7-W14 package timed on the H200 (the basis of the user's 52 ms bar; the zen5_vis repack timing is not that basis); feeds upload B" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_fm_stack_time.slurm --time=01:00:00 --job-name=ni26-t2p-stacktime-M7w14 -- \
      "$R/ni26-t2p-repack-t2p_ms_M7_w14-1615940/t2p_ms_M7_w14.zip" t2p_M7w14_time --dups 0 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200 ;;
  donor3c)   # T2-D484's pass branch ONLY (registered rule): D475e's donor recipe, three seeds, from the CAR-view CPT on all users
    KN='{"600":{"car":1},"601":{"car":1},"602":{"car":1}}'
    env "${dry[@]}" S1_PURPOSE="T2-D484 pass branch (build PREPARATION, not an adoption): three CAR ST-L (seeds 600-602) from the CAR-view CPT on all users, a donor of fm files for W14 if the user approves; never a candidate" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=05:00:00 --job-name=ni26-t2p-d484-car3donor -- \
      t2p_d484_car3donor --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
      --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 600,601,602 \
      --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d484-car-all/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor \
      2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200 ;;
  *) exit 2 ;;
esac
