#!/usr/bin/env bash
# T2-FM-D2 (T2-D356): T2 fine-tunes from checkpoints of the ST-EEGFormer-base from-scratch run (tag fm, runs/fm/<run>), as ni26-tag jobs
# (they read data/ni26 + the model file only). The shipped ST-L line on st_base: gain + sz, mixup 0.4, rc 0.75, LLRD 0.75, seeds 0-11,
# kit + seen1 (the fm pool's PROTEUS rows are P01 P02 P03 P06 P07 + P10 s1 only, asserted by t2p_store_pool; t2p_bench's leak guard
# re-checks the summary's proteus_streams).
#   bash tracks/t2_bci/sofia/launch_stbase_ft.sh arms|dry|GO <run name> <save tag> <steps (comma)> [afterok job]
#   ('arms' writes the arms files only; commit + push + `s1 sync ni26`, then GO)
set -euo pipefail
# XOPTS: extra sbatch options for every submit (e.g. XOPTS=--exclude=acc006, the broken node of 2026-11-05 22:04 UTC)
cd "$(dirname "$0")/../../.."
mode=${1:?arms|dry|GO}; run=${2:?run name}; tag=${3:?save tag}; steps=${4:?steps}; dep=${5:-}
R=$S1_ROOT/runs/fm
IFS=',' read -r -a S <<< "$steps"
for st in "${S[@]}"; do
  st5=$(printf '%05d' "$st")
  for p in kit seen1; do
    a=tracks/t2_bci/sofia/arms/t2fm_${tag}_s${st5}_$p.txt
    { echo "# T2-FM-D2: ST-base from scratch ($run, $tag) checkpoint step $st -> the shipped ST-L line on st_base ($p), seeds 0-11"
      echo "NSHARDS=1 --protocols $p --seeds 0,1,2,3,4,5,6,7,8,9,10,11 --models fm:st_base:mean --aligns sz --fm-calib gain --fm-ckpt $R/$run/ck_${tag}_s$st5/checkpoint-cpt.pth --fm-mixup 0.4 --fm-rc 0.75 --fm-llrd 0.75"; } > "$a"
    [ "$mode" == arms ] && continue
    dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
    dopt=(); [ -n "$dep" ] && dopt=(--dependency=afterok:$dep)
    env "${dry[@]}" S1_EXPORT=NSHARDS=1 S1_PURPOSE="T2-FM-D2 ST-base-from-scratch step $st fine-tune ($p) (feeds the pick of an FM member trained from scratch; T2-D356, GO 730)" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 "${dopt[@]}" ${XOPTS:-} --job-name=ni26-t2p-fmft-$tag-s$st5-$p -- "$a" 2>&1 \
      | grep -E 'SUBMITTED|DRY RUN|refus|error' | sed -E 's/ part=.*//' | cut -c1-90
  done
done
