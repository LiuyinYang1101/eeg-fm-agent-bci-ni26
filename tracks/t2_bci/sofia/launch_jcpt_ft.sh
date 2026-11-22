#!/usr/bin/env bash
# T2-D350 / D354: pre-chained T2 fine-tunes from each J-CPT checkpoint (afterok on the CPT job). The pool of every J-CPT arm is leak-free for
# both protocols (PROTEUS P01 P02 P03 P06 P07 + P10 s1 only), so kit AND seen1 fine-tune from the same checkpoint. The shipped ST-L line
# (T2-D247: gain + sz, mixup 0.4, rc 0.75, LLRD 0.75), seeds 0-11 (paired with stl_rcll).
#   bash tracks/t2_bci/sofia/launch_jcpt_ft.sh arms|dry|GO <cpt job id> <run name> <save tag>     ('arms' writes the arms files only; commit them, then GO)
set -euo pipefail
# STEPS: the checkpoint steps to launch (default '02000 05000 10000 20000'); a <cpt job> of '-' submits without a dependency.
# XOPTS: extra sbatch options for every submit (e.g. XOPTS=--exclude=acc006, the broken node of 2026-11-05 22:04 UTC)
cd "$(dirname "$0")/../../.."
mode=${1:?arms|dry|GO}; job=${2:?cpt job}; run=${3:?run name}; tag=${4:?save tag}
R=$S1_ROOT/runs/ni26
dep="--dependency=afterok:$job"; [ "$job" == - ] && dep=""     # job '-' = the checkpoints exist already: no dependency
for st in ${STEPS:-02000 05000 10000 20000}; do
  for p in kit seen1; do
    a=tracks/t2_bci/sofia/arms/t2jcpt_${tag}_s${st}_$p.txt
    { echo "# J-CPT ($run, $tag) checkpoint step $st -> the shipped ST-L fine-tune ($p), seeds 0-11"
      echo "NSHARDS=1 --protocols $p --seeds 0,1,2,3,4,5,6,7,8,9,10,11 --models fm:st_large:mean --aligns sz --fm-calib gain --fm-ckpt $R/$run/ck_${tag}_s$st/checkpoint-cpt.pth --fm-mixup 0.4 --fm-rc 0.75 --fm-llrd 0.75"; } > "$a"
    [ "$mode" == arms ] && continue
    dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
    env "${dry[@]}" S1_EXPORT=NSHARDS=1 S1_PURPOSE="J-CPT $tag step $st fine-tune ($p) (feeds the pick: which CPT checkpoint the shipped ST-L members use; GO 730)" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 $dep ${XOPTS:-} --job-name=ni26-t2p-jft-$tag-s$st-$p -- "$a" 2>&1 \
      | grep -E 'SUBMITTED|DRY RUN|refus|error' | sed -E 's/ part=.*//' | cut -c1-90
  done
done
