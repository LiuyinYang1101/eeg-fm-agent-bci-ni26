#!/usr/bin/env bash
# D82 (T2 PROTEUS FM screen) launcher: one H200 job per FM, 3 shards per GPU, protocols loso,xsess,kit, seeds 0-2,
# recipe fmE30 (t2p_bench defaults), aligns = D81's carry rule ($1, e.g. "none,ea"). Run locally after `s1 sync ni26`.
#   bash tracks/t2_bci/sofia/launch_d82.sh none,ea
set -euo pipefail
ALIGNS="$1"
SMOKE=1451751   # ni26-t2p-fmsmoke (COMPLETED): st_base + mor_p1v2 at 4 s on the kit split
declare -A WALL=([st_base]=6 [rand_st_base]=6 [st_large]=14 [mor_p1v2]=8 [mor_p1off]=8 [mor100]=8 [reve]=4 [labram]=4)   # one align
for fm in st_base st_large mor_p1v2 mor_p1off mor100 reve labram rand_st_base; do
  h=${WALL[$fm]}
  ns=3; [ "$fm" = st_large ] && ns=2   # ST-L: ~40 GB activations per process at batch 64 x 1,313 tokens
  S1_SMOKE=$SMOKE S1_EXPORT=NSHARDS=$ns \
  S1_PURPOSE="ni26 T2 PROTEUS D82 FM screen: $fm (readout mean, fmE30) x align {$ALIGNS} x {loso,xsess,kit} x seeds 0-2; feeds D82 (a) FM vs EEGNet, (b) mor_p1v2 vs mor_p1off, (c) pretrained vs random" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_bench.slurm --time=$(printf '%02d' "$h"):00:00 --job-name="ni26-t2p-d82-${fm//_/}" \
    -- --protocols loso,xsess,kit --models "fm:$fm:mean" --aligns "$ALIGNS" --seeds 0,1,2 2>&1 | grep -E "SUBMITTED|refused|exit|!!"
done
