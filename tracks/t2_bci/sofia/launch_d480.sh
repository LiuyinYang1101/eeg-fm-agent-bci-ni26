#!/usr/bin/env bash
# T2-D480 (panel P11; the manager's go inside T2's GO; registered before any number): M7's ST-L member line on surface-Laplacian input
# (--lap = ni26.stream.lap_windows, k 4 nearest neighbours) = the view without far-field (reference, ocular) sources, kit + seen1, seeds
# 472-485 (T2-D455 / D475's), so the read pairs it with the plain and CAR nets on the same seeds. Eight H200 jobs (four per protocol,
# 4 / 4 / 3 / 3 seeds, one shard each: ST-L fine-tunes OOM at 3 per H200), --nice behind the D475e donor copy and Track3's jobs.
#   mem-guard bash tracks/t2_bci/sofia/launch_d480.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d480.sh tracks/t2_bci/sofia/arms/t2d480_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D480 the no-ocular third view (surface Laplacian on the ST-L members' input, paired with T2-D455 plain and T2-D475 CAR nets; panel P11): feeds W14's member mix"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; }
for pr in kit seen1; do
  for j in 0 1 2 3; do
    sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; $pr part $j" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 --nice=200 \
      --job-name="ni26-t2p-d480-stllap-$pr-$j" -- "tracks/t2_bci/sofia/arms/t2d480_stllap_${pr}_$j.txt"
  done
done
