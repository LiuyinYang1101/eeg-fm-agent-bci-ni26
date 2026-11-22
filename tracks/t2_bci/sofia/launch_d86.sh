#!/usr/bin/env bash
# D86 (T2 PROTEUS FM harnessing round 1) launcher: one H200 job per (FM, arm); baseline = the FM's D82 run. Run locally
# after `s1 sync ni26`:   mem-guard bash tracks/t2_bci/sofia/launch_d86.sh mor_p1v2 st_base
set -euo pipefail
SMOKE=1451751
EXTRA=$S1_ROOT/data/ni26/t2_proteus/export_scherer2015_v1
COMMON="--protocols loso,xsess,kit --aligns wz --seeds 0,1,2"
for fm in "$@"; do
  arms=("H1|--fm-layer 6|6" "H2|--fm-epochs 12|4" "H3|--fm-ema 0.995|8" "H4|--fm-lora 8 --fm-lr-enc 1e-3|6"
        "H6|--extra $EXTRA|12" "H7|--fm-mixup 0.4|8")
  rd=mean
  [[ $fm == st_* ]] && arms+=("H8|--fm-drop-path 0.1|8")
  for spec in "${arms[@]}" "H5|ATTN|8"; do
    IFS='|' read -r arm flags h <<< "$spec"
    m="fm:$fm:mean"
    if [ "$flags" = ATTN ]; then m="fm:$fm:attn"; flags=""; fi
    S1_SMOKE=$SMOKE S1_EXPORT=NSHARDS=3 \
    S1_PURPOSE="ni26 T2 PROTEUS D86 FM harnessing $arm on $fm ($m $flags) x {loso,xsess,kit} x seeds 0-2; feeds D86 (arm vs the FM's D82 baseline)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_bench.slurm --time=$(printf '%02d' "$h"):00:00 --job-name="ni26-t2p-d86-${fm//_/}-$arm" \
      -- $COMMON --models "$m" $flags 2>&1 | grep -E "SUBMITTED|refused|exit|!!"
  done
done
