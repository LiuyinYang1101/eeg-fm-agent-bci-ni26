#!/usr/bin/env bash
# T2-FM-D6: T2 fine-tunes from the ST-L + P1 checkpoints (two fm runs: t2fm-p1-kit from CPT v1 kit d89b, t2fm-p1-seen1 from CPT v1 seen1
# d111-with; each protocol reads only its own run, as M6's members read their own CPT v1). ni26-tag jobs (data/ni26 + the model file).
#   recipe plain = stl_rcll's line (sz, gain, mixup 0.4, rc 0.75, LLRD 0.75), seeds 0-11 (paired with stl_rcll: the dose curve)
#   recipe vp12  = M6's vp12 member's line (the same + --fm-vp 2), seeds 12-23 (paired with vp12: the runtime-neutral swap / the add)
#   bash tracks/t2_bci/sofia/launch_p1_ft.sh arms|dry|GO <recipe plain|vp12> <steps comma> [kit afterok job] [seen1 afterok job]
#   XOPTS: extra sbatch options (excludes, --nice=300). Commit + sync the arms before GO.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?arms|dry|GO}; rec=${2:?plain|vp12}; steps=${3:?steps}; dk=${4:-}; ds=${5:-}
R=$S1_ROOT/runs/fm
case "$rec" in
  plain) seeds=0,1,2,3,4,5,6,7,8,9,10,11; extra="" ;;
  vp12)  seeds=12,13,14,15,16,17,18,19,20,21,22,23; extra=" --fm-vp 2" ;;
  *) echo "recipe plain|vp12" >&2; exit 2 ;;
esac
IFS=',' read -r -a S <<< "$steps"
for st in "${S[@]}"; do
  st5=$(printf '%05d' "$st")
  for p in kit seen1; do
    tag=$([ $p = kit ] && echo p1k || echo p1s); dep=$([ $p = kit ] && echo "$dk" || echo "$ds")
    a=tracks/t2_bci/sofia/arms/t2fm_p1_${rec}_s${st5}_$p.txt
    { echo "# T2-FM-D6: ST-L + P1 (t2fm-p1-$p, $tag) checkpoint step $st -> the $rec recipe ($p), seeds $seeds"
      echo "NSHARDS=1 --protocols $p --seeds $seeds --models fm:st_large:mean --aligns sz --fm-calib gain --fm-ckpt $R/t2fm-p1-$p/ck_${tag}_s$st5/checkpoint-cpt.pth --fm-mixup 0.4 --fm-rc 0.75 --fm-llrd 0.75$extra"; } > "$a"
    [ "$mode" == arms ] && continue
    dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
    dopt=(); [ -n "$dep" ] && dopt=(--dependency=afterok:$dep)
    env "${dry[@]}" S1_EXPORT=NSHARDS=1 S1_PURPOSE="T2-FM-D6 ST-L + P1 step $st fine-tune ($rec, $p) (feeds the pick: whether a P1-post-trained ST-L replaces / joins M6's ST-L members; the user's 'further pretrain ST-EEGFormer' directive, FM block)" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 "${dopt[@]}" ${XOPTS:-} --job-name=ni26-t2p-p1ft-$rec-s$st5-$p -- "$a" 2>&1 \
      | grep -E 'SUBMITTED|DRY RUN|refus|error' | sed -E 's/ part=.*//' | cut -c1-90
  done
done
