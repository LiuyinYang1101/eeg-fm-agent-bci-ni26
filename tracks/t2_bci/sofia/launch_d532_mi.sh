#!/usr/bin/env bash
# T2-D532 (the W18 fresh-draw stack read; PREPARATION for its D529 part): the MI-expert ST-L member at the FRESH seed 826 (swapped for D531's CTL
# s826 in the fresh read), stage 2 from D529's stage-1 checkpoint (ni26-t2p-preft-d529-mi-root), fold halves, afterok D529's smoke.
#   mem-guard bash tracks/t2_bci/sofia/launch_d532_mi.sh dry|GO <D529 smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; dep=${2:?the D529 smoke job id}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d532_mi.sh tracks/t2_bci/sofia/arms/t2d532_mi_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D532 W18 fresh-draw read, the D529 part: the MI-expert ST-L member at fresh seed 826 (swap for D531's CTL s826), K = 9; feeds T2 upload B's W18 stack read"
for f in tracks/t2_bci/sofia/arms/t2d532_mi_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d532_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 \
    --dependency="afterok:$dep" --job-name="ni26-t2p-d532-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
