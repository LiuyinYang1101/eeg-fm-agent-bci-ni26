#!/usr/bin/env bash
# T2-D557 (the combined FM augmentation part; registered 2026-11-17 before any number; the manager's ask; inside GO and
# the s1 caps): the five HEADED FM members of W19c's slot re-run with NI26_SIDE_DUMP=1 so the 822-824 read can re-fit their heads in-stream:
#   SD 827 (plain) / 829 / 831 (CAR) = D531's arms, fold halves (6 jobs, walls 2:30 = D531's measured 1:15 + margin);
#   D546's confirm bases 835 (plain) / 836 (CAR) + the arm-b continuation cb = D546's arms with arm b only, fold pairs (10 jobs, walls 2:00).
#   mem-guard bash tracks/t2_bci/sofia/launch_d557.sh dry|GO smoke|all [<smoke job id>: the members wait on it, afterok]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?smoke|all}; dep=()
[ -n "${3:-}" ] && dep=(--dependency="afterok:$3")
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == smoke || "$part" == all ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d557.sh tracks/t2_bci/sofia/arms/t2d557*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D557 combined FM augmentation (gain + shift + channel dropout + mixup 0.8) on the five headed W19 FM members, K = 9, side dumps; feeds the D557 stack read on fresh draws 828-830 (a W20 part candidate)"
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1" S1_PURPOSE="$P; SMOKE (lc9_0, 2 + 1 epochs)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm \
    --time=00:30:00 --job-name="ni26-t2p-d557-smoke" -- tracks/t2_bci/sofia/arms/t2d557smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
for f in tracks/t2_bci/sofia/arms/t2d557_stl*.txt tracks/t2_bci/sofia/arms/t2d557_{plain_s835,car_s836}_p{0..4}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d557_}
  w=02:00:00; [[ "$a" == stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    "${dep[@]}" --job-name="ni26-t2p-d557-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
