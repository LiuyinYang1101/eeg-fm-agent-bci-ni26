#!/usr/bin/env bash
# T2-D586 SPECULATIVE DONOR (the manager after the Fable review, 2026-11-18 item e: started before the reads, at NICE=50; inside GO, < 16 GPU-h).
# ONE member: the plain non-continuation SD donor 621 of launch_d557_donor.sh's job A (d111-all CPT, sz, gain,
# 100 epochs, llrd 0.75, rc 0.75, tilt 0.15, --fm-side dcmean, --fm-aug 0.2:30:0.1, --fm-mixup 0.8) + --fm-phase 0.25:0.7854. No merge, no build:
# W22 needs a NEW owner line (the merge would replace W21's member 621 only).
#   [NICE=50 is the default] mem-guard bash tracks/t2_bci/sofia/launch_d586_donor.sh dry|GO all
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?all}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == all ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d586_donor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
AUG=(--fm-aug 0.2:30:0.1 --fm-mixup 0.8 --fm-phase 0.25:0.7854)   # T2-D586: D557's dose + the shared-phase surrogate
COMMON=(--model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all --fm st_large --fm-align sz --fm-calib gain --fm-weight 0.35
        --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-donor --fm-side dcmean "${AUG[@]}")
P="T2-D586 SPECULATIVE all-user donor 621 (the manager, item e: before the reads; no build: W22 waits for a NEW owner line)"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="$P" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --nice="${NICE:-50}" --time=02:30:00 --job-name=ni26-t2p-d586-donor-621 -- \
  t2p_d586_donor_621 "${COMMON[@]}" --fm-epochs 100 --fm-seeds 621 --fm-seed-knobs '{"621":{"tilt":0.15}}' 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
