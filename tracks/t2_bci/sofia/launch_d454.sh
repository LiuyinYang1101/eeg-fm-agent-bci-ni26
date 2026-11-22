#!/usr/bin/env bash
# T2-D454 (the user's idea, 2026-11-11): one first stage PER TASK on outside datasets (MI / mental arithmetic / the Scherer command set),
# each its own model, then PROTEUS trains the head (LP) or the whole member (FT), and the three experts vote.
#   bash tracks/t2_bci/sofia/launch_d454.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_preft.py tracks/t2_bci/sofia/t2p_preft.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d454.sh tracks/t2_bci/sofia/arms/t2d454_*.txt scripts/local/t2_d454_read.py)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
P="T2-D454 (task experts from outside datasets, then PROTEUS; the user's idea): feeds T2's remaining sealed uploads"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
declare -A POOLS=([mi]="dreyer bnci2014_001 bnci2015_001 cho2017 lee2019_mi physionetmi schirrmeister2017 stieger2021 weibo2014" [math]="shin2017b" [word]="scherer")
declare -A STEPS=([mi]=4000 [math]=1500 [word]=1500)
declare -A J
for e in mi math word; do
  J[$e]=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-preft-d454-$e" S1_PURPOSE="$P; stage 1 $e" s1 submit ni26 tracks/t2_bci/sofia/t2p_preft.slurm --time=03:00:00 \
    --job-name="ni26-t2p-preft-d454-$e" -- ${POOLS[$e]} -- --steps "${STEPS[$e]}" --holdout person --decision T2-D454)
done
for e in ctl mi math word; do
  dep=(); [ "$e" != ctl ] && [ -n "${J[$e]:-}" ] && dep=(--dependency="afterok:${J[$e]}")
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; stage 2 $e" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=04:00:00 "${dep[@]}" \
    --job-name="ni26-t2p-d454-$e" -- tracks/t2_bci/sofia/arms/t2d454_$e.txt >/dev/null
done
