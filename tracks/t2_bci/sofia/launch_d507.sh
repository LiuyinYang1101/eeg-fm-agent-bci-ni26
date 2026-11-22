#!/usr/bin/env bash
# T2-D507 (A1: Scherer2015 in the like-for-like form on the K = 9 ten-stranger panel). One t2p_multi job per arm (6 x 1 H200).
#   bash tracks/t2_bci/sofia/launch_d507.sh dry|GO [arms...]      (default: c0 c s h1 h25 h25b)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; shift || true
arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(c0 c s h1 h25 h25b)
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d507.sh tracks/t2_bci/sofia/arms/t2d507_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
for a in "${arms[@]}"; do
  env "${dry[@]}" S1_EXPORT="NSHARDS=3" S1_PURPOSE="T2-D507 arm $a: does Scherer2015 (CAR, own head / low weight / domain BN) widen the ShallowNet trunk's coverage of strangers (ten-stranger K=9 panel); feeds T2's Nov 20/21 builds" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 --job-name="ni26-t2p-d507-$a" -- tracks/t2_bci/sofia/arms/t2d507_$a.txt 2>&1 \
    | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-170
done
