#!/usr/bin/env bash
# T2-D417 (sweep (d) of the extended plan; the AUD re-doses; GO 1,200 typed 2026-11-09, standing approval R3): 12 ST-L member fine-tunes on fresh
# seeds 108-119 = 6 recipes x kit / seen1 (arms/t2d417_<dose>_<proto>.txt): c0 = M7's LP-FT member recipe (the PAIRED control), a / b / c = AUD 1
# (lp10 / lp15 / lp20 + lh3 + vp2), l1 / l2 = AUD 4 (LoRA-in 16 at lr 5e-4 / 2e-3, LLRD 1.0). One t2p_multi job per arm (1 GPU, <= 3 h).
#   bash tracks/t2_bci/sofia/launch_d417.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d417.sh tracks/t2_bci/sofia/arms/t2d417_*.txt scripts/local/t2_d417_read.py)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null
for d in c0 a b c l1 l2; do
  for p in kit seen1; do
    a=tracks/t2_bci/sofia/arms/t2d417_${d}_$p.txt; jn=ni26-t2p-d417-$d-$p
    out=$(env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="T2-D417 (AUD re-doses, GO 1,200): ST-L member $d $p seeds 108-119; feeds the upload-5 (Nov 19) ST-L member choice in M7's shape" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 --job-name="$jn" -- "$a" 2>&1)
    echo "$jn: $(echo "$out" | grep -E "SUBMITTED|DRY RUN|refus|error|ERROR" | cut -c1-160 | tr '\n' ' ')"
  done
done
