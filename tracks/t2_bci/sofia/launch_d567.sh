#!/usr/bin/env bash
# T2-D567 (registered 2026-11-18 before any job / number; inside GO; the manager's last-try proposal): the HEAD-LESS MI member (slot 0) with
# D557's augmentation at its PROTEUS fine-tune stage (--fm-aug 0.2:30:0.1 = gain 0.2 + time shift +-30 samples (+-0.25 s at 120 Hz) + channel
# dropout 0.1, --fm-mixup 0.8; the MOABB MI stage = the MI root checkpoint, unchanged).
#   READ members (t2p_multi, the side + feature dumps, 2 lc9 folds per job, exactly D555's / D558's MI lines + the augmentation):
#     seed 832 (B = draws 828-830, vs D555's emi@832)  and  seed 842 (C = draws 831-833, vs D558's hmi@842): 10 jobs, 0.9 GPU-h each plain.
#   DONOR (t2p_final, all users, speculative): D529's MI donor d2 s620 (= W20's slot 0, job 1620735) + the augmentation: 1 job.
# No new smoke: the MI lines (D555 / D558) and the augmentation (D557 / D558's t2p_multi lines, D557's t2p_final donors) both ran; only the
# combination is new, and the augmentation is applied in t2p_fm's training loop independently of the side head (t2p_fm.py:1117-1160).
#   mem-guard bash tracks/t2_bci/sofia/launch_d567.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d567.sh tracks/t2_bci/sofia/arms/t2d567_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
R=$S1_ROOT/runs/ni26
P="T2-D567 the MI member (slot 0) with D557's augmentation at its PROTEUS fine-tune (the manager's last try on W20+D557); FM before the Nov 19 04:00 cut-off"
for f in tracks/t2_bci/sofia/arms/t2d567_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d567_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm \
    --time=02:00:00 --job-name="ni26-t2p-d567-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
s=620; KN="{\"$s\":{\"tilt\":0.15}}"
env "${dry[@]}" S1_PURPOSE="$P; the all-user DONOR (speculative; D529's MI donor d2 s620 + the augmentation), merged into slot 0 only if both D567 reads pass and the owner types the W21 line" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name="ni26-t2p-d567-midonor-s$s" -- \
  "t2p_d567_midonor_s$s" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-aug 0.2:30:0.1 --fm-mixup 0.8 --fm-seeds "$s" \
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-preft-d529-mi-root/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor \
  2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
