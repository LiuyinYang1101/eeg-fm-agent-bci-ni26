#!/usr/bin/env bash
# T2-D475 arm A: M7's ST-L member line on common-average-referenced input (--car), kit + seen1, seeds 472-485 = T2-D455's control seeds.
# Two H200 jobs (one per protocol), one shard each (ST-L fine-tunes OOM at 3 per H200). Arm B (FBCNet) runs on the local 5090:
# scripts/local/t2_d475_fbc_local.sh (its control, T2-D419's e180, was made there too).
#   mem-guard bash tracks/t2_bci/sofia/launch_d475.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d475.sh tracks/t2_bci/sofia/arms/t2d475_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
P="T2-D475 arm A (common average reference on the ST-L members' input, paired with T2-D455's controls): feeds a CAR template step for T2's last upload"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
for pr in kit seen1; do
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=07:00:00 \
    --job-name="ni26-t2p-d475-stlcar-$pr" -- "tracks/t2_bci/sofia/arms/t2d475_stlcar_$pr.txt"
done
