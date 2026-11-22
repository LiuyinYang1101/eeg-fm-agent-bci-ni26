#!/usr/bin/env bash
# T2-D293 launch (the user approved backlog row 59 on 2026-10-28; launch only after sofia returns, >= 2026-10-28 18:00): one `s1 sync ni26`
# (git archive HEAD: commit the D293 code first), then three t2p_multi jobs (FBCNet kit / seen1, ShallowNet kit + seen1), each ~15-25 min on
# one H200 (T2-D220's shape: 0.60 GPU-h in all). Read: `s1 fetch ni26 ni26-t2p-d293-<arm>-<id>` x 3, then scripts/local/t2_d293_read.sh.
#   bash scripts/local/t2_d293_launch.sh            (prints the job ids)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/sofia/arms/t2d293_*.txt)" ] || { echo "commit the D293 code first"; exit 2; }
s1 sync ni26
for arm in fbc_kit fbc_seen1 sh; do
  S1_EXPORT=NSHARDS=3 S1_PURPOSE="T2-D293 DANN pooled CNNs (user-approved row 59): arms t2d293_$arm (lambda 0.1 / 0.3), feeds the iteration builds (PLAN 6b item 11)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:00:00 --job-name=ni26-t2p-d293-${arm//_/-} -- tracks/t2_bci/sofia/arms/t2d293_$arm.txt
done
