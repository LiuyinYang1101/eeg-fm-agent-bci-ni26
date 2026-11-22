#!/usr/bin/env bash
# T2-D529 (registered 2026-11-16 before any number; inside GO and the s1 caps): ONE MI-expert ST-L member in W17's FM slot on K = 9.
# Stage 1 = t2p_preft on D454's nine MI corpora from the PUBLIC ST-L root (PREFT_INIT=root; no PROTEUS window), then a 1-epoch
# stage-2 smoke (afterok), then D512's plain tilt line from that checkpoint at seeds 820 / 821 on the ten lc9 folds (fold halves, afterok the smoke).
#   mem-guard bash tracks/t2_bci/sofia/launch_d529.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/t2p_preft.slurm tracks/t2_bci/sofia/launch_d529.sh tracks/t2_bci/sofia/arms/t2d529*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D529 one MI-expert ST-L member (MI first stage from the public root) swapped into W17's FM slot, K = 9 strangers; feeds T2 upload B's W18 stack read (weights only)"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
MI="dreyer bnci2014_001 bnci2015_001 cho2017 lee2019_mi physionetmi schirrmeister2017 stieger2021 weibo2014"
J1=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-preft-d529-mi-root,PREFT_INIT=root" S1_PURPOSE="$P; stage 1 (MI corpora, public root)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_preft.slurm --time=01:15:00 --job-name=ni26-t2p-preft-d529-mi-root \
  -- $MI -- --steps 4000 --holdout person --for-protocol none --decision T2-D529)
[ "$mode" == dry ] && { echo "dry: stage 1 validated; the chained jobs need its id"; exit 0; }
[ -n "$J1" ] || { echo "stage 1 not submitted"; exit 2; }
JS=$(sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; stage-2 smoke" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:30:00 \
  --dependency="afterok:$J1" --job-name=ni26-t2p-d529-smoke -- tracks/t2_bci/sofia/arms/t2d529smoke.txt)
[ -n "$JS" ] || { echo "smoke not submitted"; exit 2; }
for f in tracks/t2_bci/sofia/arms/t2d529_mi_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d529_}
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 \
    --dependency="afterok:$JS" --job-name="ni26-t2p-d529-$a" -- "$f" >/dev/null
done
echo "stage 1 $J1, smoke $JS"
