#!/usr/bin/env bash
# T2-D348 (the user's trick 1: CPT length): per protocol (kit = d89b's pool, seen1 = d111-with's) and --steps 1000 / 2000 / 10000,
# one CPT v1-recipe run (own cosine) and, afterok, the shipped ST-L fine-tune (T2-D247's lines) from it on seeds 0-11.
#   bash tracks/t2_bci/sofia/launch_d348_cptlen.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_cpt.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/sofia/t2p_cpt.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d348_cptlen.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null
mkdir -p tracks/t2_bci/sofia/arms
sub() {   # jobname dep time slurm export args...
  local jn="$1" dep="$2" tm="$3" sl="$4" ex="$5"; shift 5
  local depflag=(); [ -n "$dep" ] && [ "$dep" != 0 ] && depflag=(--dependency="afterok:$dep")
  [ -n "${EXCLUDE:-}" ] && depflag+=(--exclude="$EXCLUDE")          # bad nodes (SOFIA_OPS §6)
  local out; out=$(env "${dry[@]}" S1_EXPORT="$ex" S1_PURPOSE="T2-D348 (the user's trick 1, CPT length; GO 430): $jn" \
    s1 submit ni26 "$sl" --time="$tm" "${depflag[@]}" --job-name="$jn" -- "$@" 2>&1)
  echo "$out" | grep -E "SUBMITTED|DRY RUN|refus|error|ERROR" | cut -c1-160 >&2
  local j; j=$(sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"); echo "${j:-0}"
}
for p in kit seen1; do
  tag=$([ $p == kit ] && echo kit || echo se1); pool=(); [ $p == seen1 ] && pool=(--proteus-pool seen1)
  for n in 1000 2000 10000; do
    k=$(printf 's%02dk' $((n / 1000))); run="ni26-t2p-cpt-d348$tag-$k"; tm=$([ $n == 10000 ] && echo 06:00:00 || echo 02:00:00)
    if [[ " ${DONE_CPT:-} " == *" $run "* ]]; then j=0; echo "$run: CPT done, fine-tune only" >&2; else   # a resubmission after a node fault
    j=$(sub "$run" "" "$tm" tracks/t2_bci/sofia/t2p_cpt.slurm "RUN_NAME=$run" --fm st_large --extra export_dreyer2023_wide_v1 export_scherer2015_v1 \
          --steps $n --batch 64 --accum 8 --lr 6.58e-5 --wd 1e-5 --mask-ratio 0.75 --eval-every 500 "${pool[@]}"); fi   # the pinned ST-L is weights-only: v1's values (cpt_summary of d89b / d111-with)
    a=tracks/t2_bci/sofia/arms/t2d348_${tag}_$k.txt
    { echo "# T2-D348: T2-D247's shipped ST-L fine-tune ($p) from the $n-step CPT $run, seeds 0-11 (paired with stl_rcll = the 5k members)"
      echo "NSHARDS=1 --protocols $p --seeds 0,1,2,3,4,5,6,7,8,9,10,11 --models fm:st_large:mean --aligns sz --fm-calib gain --fm-ckpt $R/$run/checkpoint-cpt.pth --fm-mixup 0.4 --fm-rc 0.75 --fm-llrd 0.75"; } > "$a"
    echo "$p $n: cpt job $j, arms $a" >&2
    [ "$mode" == GO ] && sub "ni26-t2p-d348-ft-$tag-$k" "$j" 03:00:00 tracks/t2_bci/sofia/t2p_multi.slurm "NSHARDS=1" "$a" >/dev/null
  done
done
