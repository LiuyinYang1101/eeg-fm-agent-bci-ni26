#!/usr/bin/env bash
# T2-D453 (the user's idea, 2026-11-11: pick useful users from other datasets by performance). One t2p_multi job per stage.
#   bash tracks/t2_bci/sofia/launch_d453.sh dry|GO 1
#   bash tracks/t2_bci/sofia/launch_d453.sh dry|GO 2 <PLUS set, e.g. 2+5+7> <TOP3 set>     (writes arms/t2d453_s2.txt; commit it first)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; st=${2:?1|2}
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d453.sh tracks/t2_bci/sofia/arms/t2d453_*.txt scripts/local/t2_d453_read.py)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
env "${dry[@]}" S1_EXPORT="NSHARDS=3" S1_PURPOSE="T2-D453 stage $st: which Scherer2015 users help a PROTEUS ShallowNet (per-user selection of external training data); feeds T2's remaining sealed uploads" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=04:00:00 --job-name="ni26-t2p-d453-s$st" -- tracks/t2_bci/sofia/arms/t2d453_s$st.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-170
