#!/usr/bin/env bash
# T2-D546 (the owner's warm-start continuation of converged FM members; registered 2026-11-16 20:11:20; the manager's OK ≈ 20:4x; inside GO and the
# s1 caps, every job <= 2 GPU-h): SD s828 (plain) / s830 (CAR) for the nomination on 816-818, s835 / s836 for the 819-821 confirm; one job per fold
# pair = the base (100 epochs) + arms a / b from its state (25 epochs at lr x 0.1). CPU smoke of the code path: scripts/local/t2_d546_smoke.py PASS.
#   mem-guard bash tracks/t2_bci/sofia/launch_d546.sh dry|GO nom|confirm|all
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?nom|confirm|all}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == nom || "$part" == confirm || "$part" == all ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d546.sh tracks/t2_bci/sofia/arms/t2d546_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D546 the owner's warm-start continuation (base + re-init head/top + low-LR arms) of SD FM members, K = 9; feeds the T2 W19 stack (owner direction 2026-11-16)"
arms=()
[[ "$part" == nom || "$part" == all ]] && arms+=(tracks/t2_bci/sofia/arms/t2d546_{plain_s828,car_s830}_p{0..4}.txt)
[[ "$part" == confirm || "$part" == all ]] && arms+=(tracks/t2_bci/sofia/arms/t2d546_{plain_s835,car_s836}_p{0..4}.txt)
for f in "${arms[@]}"; do
  a=$(basename "$f" .txt); a=${a#t2d546_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:00:00 \
    --job-name="ni26-t2p-d546-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
