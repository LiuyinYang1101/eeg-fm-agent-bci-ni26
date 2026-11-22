#!/usr/bin/env bash
# T2-D544 (word-α, the high-risk bet; registered 2026-11-16 20:03:46; the manager's OK ≈ 20:0x; hard calendar: every member by Nov 18 12:00 or the item
# is dropped): stage 1 = t2p_preft on Scherer2015 from the PUBLIC root (D454's word stage: 1,500 steps, person hold-out; PREFT_INIT=root) ->
# stage 1b = t2p_interp.slurm (0.5 root + 0.5 expert, zen5_vis, afterok) -> the word-α SD members at s827 (nomination) and s833 (confirm), afterok
# the interpolation; the confirm BASE (SD s833 from the root) has no dependency. Fold pairs, 1:15 walls.
#   mem-guard bash tracks/t2_bci/sofia/launch_d544.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/t2p_preft.slurm tracks/t2_bci/sofia/t2p_interp.slurm tracks/t2_bci/sofia/launch_d544.sh tracks/t2_bci/sofia/arms/t2d544_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D544 word-α FM member (Scherer2015 word stage from the public root, 0.5-interpolated with the root, then the SD fine-tune), K = 9; feeds the T2 W19 stack (owner direction 2026-11-16)"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
# the confirm BASE first (no dependency)
for f in tracks/t2_bci/sofia/arms/t2d544_sd_s833_p{0..4}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d544_}
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; confirm base $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=01:15:00 --job-name="ni26-t2p-d544-$a" -- "$f" >/dev/null
  [ "$mode" == dry ] && break
done
J1=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-preft-d544-word-root,PREFT_INIT=root" S1_PURPOSE="$P; stage 1 (Scherer2015 word stage, public root)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_preft.slurm --time=01:00:00 --job-name=ni26-t2p-preft-d544-word-root \
  -- scherer -- --steps 1500 --holdout person --for-protocol none --decision T2-D544)
[ "$mode" == dry ] && { echo "dry: stage 1 validated; the chained jobs need its id"; exit 0; }
[ -n "$J1" ] || { echo "stage 1 not submitted"; exit 2; }
J2=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-d544-wordalpha,EXPERT=ni26-t2p-preft-d544-word-root" S1_PURPOSE="$P; stage 1b (0.5 root + 0.5 word expert, CPU)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_interp.slurm --partition=zen5_vis --account=vsc --time=00:30:00 --dependency="afterok:$J1" --job-name=ni26-t2p-d544-interp)
[ -n "$J2" ] || { echo "stage 1b not submitted"; exit 2; }
for f in tracks/t2_bci/sofia/arms/t2d544_wa_s8{27,33}_p{0..4}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d544_}
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=01:15:00 \
    --dependency="afterok:$J2" --job-name="ni26-t2p-d544-$a" -- "$f" >/dev/null
done
echo "stage 1 $J1 -> interp $J2 -> 10 word-α members"
