#!/usr/bin/env bash
# T2-D498 (the manager's GO 2026-11-14 21:0x, inside GO and the s1 caps; registered before any number): the FM members of the ten-stranger
# full-stack read on the K = 9 panel (lc9, 10 folds): 3 plain (D455's line) + 3 CAR (D475's line) ST-L, 100 epochs, from the PUBLIC ST-L
# checkpoint (every shipped CPT pool holds each fold's held-out user). A 1-epoch smoke (lc9, CAR path, empty val split) first; then one job
# per seed (10 folds each, NSHARDS 1: the 2-shard smoke 1617890 OOM'd on every fold, D455's line runs one per H200), afterok on the smoke.
#   mem-guard bash tracks/t2_bci/sofia/launch_d498.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d498.sh tracks/t2_bci/sofia/arms/t2d498_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D498 ten-stranger read of the full stack: where the 0.84 local -> 0.819 sealed loss sits and which family carries new users; feeds upload C's composition"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
SM=$(sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; smoke (1 epoch, lc9, CAR path)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=01:30:00 \
  --job-name="ni26-t2p-d498-smoke" -- tracks/t2_bci/sofia/arms/t2d498_smoke.txt)
dep=(); [ -n "${SM:-}" ] && dep=(--dependency="afterok:$SM")
for a in stl_s760 stl_s761 stl_s762 stlcar_s763 stlcar_s764 stlcar_s765; do
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a (10 folds)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=08:00:00 "${dep[@]}" \
    --job-name="ni26-t2p-d498-$a" -- "tracks/t2_bci/sofia/arms/t2d498_$a.txt" >/dev/null
done
