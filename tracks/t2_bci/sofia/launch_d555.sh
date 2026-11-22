#!/usr/bin/env bash
# T2-D555 (the FM MAIN heads' in-stream refit; registered 2026-11-17 before any number; the manager's OK; inside GO and
# the s1 caps): the five HEADED FM members of W19c's slot re-run with NI26_SIDE_DUMP=1 so the 822-824 read can re-fit their heads in-stream:
#   SD 827 (plain) / 829 / 831 (CAR) = D531's arms, fold halves (6 jobs, walls 2:30 = D531's measured 1:15 + margin);
#   D546's confirm bases 835 (plain) / 836 (CAR) + the arm-b continuation cb = D546's arms with arm b only, fold pairs (10 jobs, walls 2:00).
#   mem-guard bash tracks/t2_bci/sofia/launch_d555.sh dry|GO smoke|all [<smoke job id>: the members wait on it, afterok]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?smoke|all}; dep=()
[ -n "${3:-}" ] && dep=(--dependency="afterok:$3")
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == smoke || "$part" == all ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d555.sh tracks/t2_bci/sofia/arms/t2d555*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D555 FM MAIN-head in-stream refit: the six W19 FM members re-run with the head + feature dumps, K = 9; feeds the D555 read on fresh draws 825-827 (an option for W19, must end before the Nov 18 12:00 FM cut-off)"
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; SMOKE (lc9_0, 2 + 1 epochs)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm \
    --time=00:30:00 --job-name="ni26-t2p-d555-smoke" -- tracks/t2_bci/sofia/arms/t2d555smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
for f in tracks/t2_bci/sofia/arms/t2d555_stl*.txt tracks/t2_bci/sofia/arms/t2d555_{plain_s835,car_s836,mi_s832}_p{0..4}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d555_}
  w=02:00:00; [[ "$a" == stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    "${dep[@]}" --job-name="ni26-t2p-d555-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
