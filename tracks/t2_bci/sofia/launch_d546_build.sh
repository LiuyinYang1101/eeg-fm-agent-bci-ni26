#!/usr/bin/env bash
# T2-D546 (b) build PREPARATION (not an adoption, never a candidate by itself; the manager's OK ≈ 22:5x, with conditions): W18 b2's own
# fm_2 (seed 622, plain) and fm_4 (seed 624, CAR) CONTINUED on all users exactly as the nominated arm (b): from the packaged state
# (fp16 -> fp32, exact; side heads included), 25 epochs, every lr x 0.1, no warm-up, cosine to 0, data seed s + 50,000, the same
# recipe as D531's SD donor d2 (sz, gain, mixup 0.4, llrd 0.75, rc 0.75, tilt 0.15, d111-all CPT, --fm-side dcmean). t2p_final refuses if
# the member's recipe, its alignment scale or its side prior differ from this build's. Used ONLY if the 819-821 stack keeps D546b AND the
# owner types the W19 line; started now because the FM fine-tune cut-off is Nov 18 12:00. ONE job, ≈ 1 GPU-h, 2:00 wall.
# T2-D549 (the manager ≈ 01:3x, Nov 17): N = 1 = the b1-draw fallback's own continuation (W18 b1's fm_2 = 612 plain, fm_6 = 614 CAR: slots 2 / 4),
# the same recipe (D531's SD donor d1: seeds 610-615), the same asserts. N = 2 (default) = the b2 job 1626636 exactly.
#   mem-guard bash tracks/t2_bci/sofia/launch_d546_build.sh dry|GO [1|2]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; n=${2:-2}; [[ "$mode" == dry || "$mode" == GO ]] && [[ "$n" == 1 || "$n" == 2 ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d546_build.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "$n" == 2 ]; then
  B2=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip; SEEDS=622,624; MAP=622:fm_2.pt,624:fm_4.pt; NAME=t2p_d546b_cont_b2
  KN='{"622":{"tilt":0.15},"624":{"car":1,"tilt":0.15}}'
else
  B2=$R/ni26-t2p-w18-t2p_fin1_w18-1622522/t2p_fin1_w18.zip; SEEDS=612,614; MAP=612:fm_2.pt,614:fm_6.pt; NAME=t2p_d546b_cont_b1
  KN='{"612":{"tilt":0.15},"614":{"car":1,"tilt":0.15}}'
fi
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D546b build PREPARATION (not an adoption): W18 b$n's slot 2 / 4 members (seeds $SEEDS) continued 25 epochs at lr x 0.1 on all users (= the nominated arm b), used only if the 819 stack keeps D546b and the owner's W19 line" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=02:00:00 --job-name="ni26-t2p-d546b-cont-b$n" -- \
  "$NAME" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds "$SEEDS" \
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor --fm-side dcmean \
  --fm-init-zip "$B2" --fm-init-map "$MAP" --fm-cont-epochs 25 --fm-cont-lr 0.1 \
  2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
