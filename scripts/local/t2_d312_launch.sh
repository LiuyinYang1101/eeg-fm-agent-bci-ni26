#!/usr/bin/env bash
# T2-D312 launch (multi-seed ST-L slot on fresh seeds; pre-registered in docs/DECISIONS_T2.md before any number): one `s1 sync ni26`
# (git archive HEAD: commit the arms first), then 6 one-GPU jobs = the shipped ST-L recipe (T2-D247r's rcll arm, unchanged) at seeds
# 24-35 / 36-47 / 48-59, kit + seen1. ≈ 1.6 GPU-h each, ≈ 9.6 GPU-h total, inside the old T2 GO 280.
# Read: `s1 fetch ni26 ni26-t2p-d312-stl-<p>-<blk>-<id>` x 6, then
#   mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d312_read.py
#   bash scripts/local/t2_d312_launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/sofia/arms/t2d312_*.txt scripts/local/t2_d312_read.py)" ] || { echo "commit the D312 files first"; exit 2; }
s1 sync ni26
for p in kit seen1; do
  for b in a b c; do
    S1_EXPORT=NSHARDS=1 S1_PURPOSE="T2-D312 multi-seed ST-L slot, fresh ST-L seeds (arms t2d312_stl_${p}_$b, the shipped recipe); feeds whether k=2/3 ST-L ZIPs enter the Nov 6+ iterations" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 --job-name=ni26-t2p-d312-stl-$p-$b \
      -- tracks/t2_bci/sofia/arms/t2d312_stl_${p}_$b.txt 2>&1 | grep -E "SUBMITTED|refus|error|ERROR|exit"
  done
done
