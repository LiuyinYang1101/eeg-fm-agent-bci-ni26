#!/usr/bin/env bash
# T2-D568's B stage, PRE-LAUNCHED at nice 100 (a labelled deviation in D568's row: the manager's 11:5x call; B's verdict counts only after a C
# pass; on a C FAIL every B job is s1-cancelled and the B dirs deleted unread). 32 jobs: REF = D557's 16 B lines unchanged re-run with the feature
# dumps (t2d568_bref_*), ARM = the same at the stronger dose (t2d568_barm_*). The code path = D557's jobs; no new smoke.
#   mem-guard bash tracks/t2_bci/sofia/launch_d568b.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d568b.sh tracks/t2_bci/sofia/arms/t2d568_b*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d568_bref_*.txt tracks/t2_bci/sofia/arms/t2d568_barm_*.txt | wc -l)" -eq 32 ] || { echo "need 32 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
for f in tracks/t2_bci/sofia/arms/t2d568_bref_*.txt tracks/t2_bci/sofia/arms/t2d568_barm_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d568_}
  P="T2-D568 B stage pre-launched at nice 100 ($([[ $a == bref_* ]] && echo "REF = D557's dose re-run with feature dumps" || echo "ARM = the stronger dose")); read only after a C pass, cancelled + deleted unread on a C fail"
  w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    --nice=100 --job-name="ni26-t2p-d568-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
