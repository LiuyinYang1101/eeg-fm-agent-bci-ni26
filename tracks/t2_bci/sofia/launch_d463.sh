#!/usr/bin/env bash
# T2-D463: the C2L300 in-house checkpoint as a T2 member. Run ONLY after the in-house FM project's hand-off message for that file (raw or ann).
#   mem-guard bash tracks/t2_bci/sofia/launch_d463.sh dry|GO raw|ann <path under S1_ROOT, e.g. runs/fm/<dir>/checkpoint-XXXXXXXX.pt>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; v=${2:?raw|ann}; path=${3:?checkpoint path under S1_ROOT}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$v" == raw || "$v" == ann ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/fm tracks/t2_bci/sofia/t2p_mor_ckpts.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d463.sh tracks/t2_bci/sofia/arms/t2d463_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
P="T2-D463 (C2L300 $v as a T2 FM member): feeds T2's third sealed upload"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
J=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-d463-ckpt-$v,CKPT_PATH=$path" S1_PURPOSE="$P; link + contract" s1 submit ni26 tracks/t2_bci/sofia/t2p_mor_ckpts.slurm \
  --partition=zen5_vis --time=00:30:00 --job-name="ni26-t2p-d463-ckpt-$v")
for pr in kit seen1; do
  dep=(); [ -n "${J:-}" ] && dep=(--dependency="afterok:$J")
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=04:00:00 "${dep[@]}" \
    --job-name="ni26-t2p-d463-$v-$pr" -- "tracks/t2_bci/sofia/arms/t2d463_${v}_${pr}.txt" >/dev/null
done
