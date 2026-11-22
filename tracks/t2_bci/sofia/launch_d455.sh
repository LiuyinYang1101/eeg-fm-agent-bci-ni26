#!/usr/bin/env bash
# T2-D455: confirmation of T2-D454's post-hoc signal (the MI expert as ST-L members inside M7), fresh seeds, kit + seen1, M7's member recipe.
#   mem-guard bash tracks/t2_bci/sofia/launch_d455.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_preft.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d455.sh tracks/t2_bci/sofia/arms/t2d455_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
P="T2-D455 (MI-expert ST-L members inside M7, confirmation of D454's post-hoc read): feeds T2's third sealed upload"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
J=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-preft-d455-mi-seen1,PREFT_INIT=ni26-t2p-cpt-d111-with" S1_PURPOSE="$P; MI first stage for seen1" s1 submit ni26 tracks/t2_bci/sofia/t2p_preft.slurm --time=03:00:00 \
  --job-name="ni26-t2p-preft-d455-mi-seen1" -- dreyer bnci2014_001 bnci2015_001 cho2017 lee2019_mi physionetmi schirrmeister2017 stieger2021 weibo2014 -- --steps 4000 --holdout person --for-protocol seen1 --decision T2-D455)
for a in ctl_kit ctl_seen1 mi_kit mi_seen1; do
  dep=(); [ "$a" == mi_seen1 ] && [ -n "${J:-}" ] && dep=(--dependency="afterok:$J")
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=07:00:00 "${dep[@]}" \
    --job-name="ni26-t2p-d455-${a/_/-}" -- tracks/t2_bci/sofia/arms/t2d455_$a.txt >/dev/null
done
