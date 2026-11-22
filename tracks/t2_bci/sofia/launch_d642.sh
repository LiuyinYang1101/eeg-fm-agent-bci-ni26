#!/usr/bin/env bash
# T2-D642 (W25 = the chosen base + user-balanced FM members; the owner 2026-11-19 "aim high and target the first place's 0.859", all
# resources, relayed by the manager): the member-level READ in ship composition (ss8 + rt = 12 folds) for ALL SIX FM slots (W21's fm_0..5 =
# D578's lab lines, fresh paired seeds 900..905), three arms each: B (unweighted), C (--user-balance), T (B at seed + 50: the seed-noise floor; the manager).
# One t2p_multi8 job per (slot, arm): 8 H200s (shard i on GPU i), 12 fold-fits = 2 waves, 4 h walltime (32 GPU-h reserved each, < 100;
# 18 jobs = 576 GPU-h in flight, <= 1,000). FM side + feature dumps on (the full-chain read needs them). A smoke first (the gate needs it).
#   mem-guard bash tracks/t2_bci/sofia/launch_d642.sh dry|GO smoke
#   mem-guard bash tracks/t2_bci/sofia/launch_d642.sh dry|GO all <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; stage=${2:?smoke|all}; [[ "$mode" =~ ^(dry|GO)$ && "$stage" =~ ^(smoke|all)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi8.slurm tracks/t2_bci/sofia/launch_d642.sh tracks/t2_bci/sofia/arms/t2d642_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d642_s*_[BCT].txt | wc -l)" -eq 18 ] || { echo "need 18 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-19: target first place, all resources; T2-D642 user-balanced FM (W25), member screen + speculative donors"
EXPV="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1"
if [ "$stage" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="$EXPV" S1_PURPOSE="$P; SMOKE (slot fm_1, ss8_P01, 1 epoch, B + C)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi8.slurm \
    --gpus-per-node=1 --cpus-per-task=24 --time=00:45:00 --job-name=ni26-t2p-d642-smoke -- tracks/t2_bci/sofia/arms/t2d642_smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror|start at" | cut -c1-200
  exit 0
fi
SMOKE=${3:?the completed smoke job id}
for f in tracks/t2_bci/sofia/arms/t2d642_s*_[BCT].txt; do
  a=$(basename "$f" .txt); a=${a#t2d642_}
  env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="$EXPV" S1_PURPOSE="$P; read $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi8.slurm \
    --gpus-per-node=8 --cpus-per-task=192 --time=04:00:00 --job-name="ni26-t2p-d642-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror|start at" | cut -c1-200
  [ "$mode" == dry ] && break
done
