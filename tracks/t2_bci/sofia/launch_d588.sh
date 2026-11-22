#!/usr/bin/env bash
# T2-D588 (consistency regularisation in the FM fine-tune: 0.5 (CE1 + CE2) + 1.0 x symmetric KL between two independently augmented views,
# on D557's dose, the five headed FM lines; registered 2026-11-18 before any job; inside GO):
#   smoke: ONE 1 + 1-epoch job on lc9_0 (t2d588smoke.txt: the two-view path on the GPU, the KL term in its log, the continuation included)
#   c:     the 16 C jobs (D558's aug arms at seeds 837-841 + --fm-cons 1.0), nice 0
#   bd:    B ARM (16: D557's B lines + --fm-cons 1.0; the B REF = D582's bref jobs, no new REF) + D (16: D578's set-D lines + the flag),
#          PRE-LAUNCHED at nice 100 under the row's rule (no B / D output opened before a C pass; on a C fail every B / D job cancelled and
#          the dirs deleted unread)
# Walltimes: D558's aug jobs fit in ~8,000 s (5-fold SD lines) / ~4,050 s (2-fold base + cb lines) on the H200 (their logs); two forward /
# backward passes per step ~1.85x the training time -> 05:00:00 / 03:00:00 (~1.9x the measured job time, with margin).
#   mem-guard bash tracks/t2_bci/sofia/launch_d588.sh smoke|c|bd dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
part=${1:?smoke|c|bd}; mode=${2:?dry|GO}
[[ "$part" == smoke || "$part" == c || "$part" == bd ]] && [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d588.sh tracks/t2_bci/sofia/arms/t2d588*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d588_*.txt | wc -l)" -eq 48 ] || { echo "need 48 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="T2-D588 SMOKE: 1 + 1 epochs on lc9_0 with --fm-cons 1.0 (the two-view path on the GPU, the KL term logged) before the C stage" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:40:00 --job-name=ni26-t2p-d588-smoke -- tracks/t2_bci/sofia/arms/t2d588smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
parts=(c); [ "$part" == bd ] && parts=(barm d)
for pt in "${parts[@]}"; do
  nice=100; [ $pt = c ] && nice=10   # T2-D588 (A), the manager: C at nice 10 (D583 / D586's B / D reads stay ahead); bd at nice 100
  P="T2-D588 consistency regularisation (sym KL 1.0, two augmented views) on D557's dose ($pt; $([ $pt = c ] && echo "the C stage, read first at the stack bar" || echo "pre-launched at nice 100: read only after a C pass, cancelled + deleted unread on a C fail"))"
  for f in tracks/t2_bci/sofia/arms/t2d588_${pt}_*.txt; do
    a=$(basename "$f" .txt); a=${a#t2d588_}
    w=03:30:00; [[ "$a" == *stl* ]] && w=06:30:00   # T2-D588 (A) smoke 1638898: 35 s / epoch at 32 per view -> SD 5 x 100 x 35 s = 4.9 h, base + cb 2 x 125 x 35 s = 2.4 h (+ margin)
    env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
      --nice=$nice --job-name="ni26-t2p-d588-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
    [ "$mode" == dry ] && break
  done
  [ "$mode" == dry ] && break
done
