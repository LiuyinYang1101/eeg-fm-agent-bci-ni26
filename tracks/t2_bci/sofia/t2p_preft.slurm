#!/usr/bin/env bash
# T2-D103: supervised multi-dataset pre-fine-tune of ST-L (t2p_preft.py) from the CPT checkpoint, on labelled public BCI
# corpora. Args = pool names: MOABB stores under data/ni26/t2_proteus/cpt_pools/<name>_v1 (label column `event`), plus the
# kit-format externals scherer (trig_code: MOABB 1.7.2's Scherer names are wrong, its codes are right) and dreyer
# (trig_description). Every named pool must exist (pre-registered list). Output = RUN_NAME dir, read by --fm-ckpt.
#   S1_EXPORT=RUN_NAME=ni26-t2p-preft-d103 S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_preft.slurm \
#       --time=04:00:00 --job-name=ni26-t2p-preft-d103 -- scherer dreyer shin2017b stieger2021 ... [-- --steps 4000]
#   MOR (T2-D106): S1_EXPORT=RUN_NAME=ni26-t2p-preft-d106mor,PREFT_FM=mor_p1v2 ... -- <pools> -- --align wz --calib zwin --decision T2-D106
#   T2-D108: ... RUN_NAME=ni26-t2p-preft-d108a,PREFT_FM=mor_p1v2 ... -- <pools> -- --align wz --calib zwin --steps 20000 --lr-enc 1e-4 \
#       --holdout person --save-at 8000,12000,16000,20000 [--mae-grad-ratio 0.5 for arm B] --decision T2-D108
#   T2-D123: ... RUN_NAME=ni26-t2p-cpt-mor-d123,PREFT_FM=mor_p1v2 ... -- proteus dreyer scherer -- --align wz --calib zwin \
#       --mae-only --exclude-subjects P04,P09 --steps 5000 --eval-every 1000 --decision T2-D123
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
POOLS=(); EXTRA=(); seen_sep=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ $seen_sep = 1 ]; then EXTRA+=("$a"); continue; fi
  case "$a" in
    scherer) p="$D/export_scherer2015_v1:trig_code" ;;
    proteus) p="$D/export_proteus_v1:trig_code" ;;      # T2-D123 (MAE-only: labels unused; --exclude-subjects P04,P09 for the kit)
    sessions) p="$D/cpt_pools/proteus_sessions_v1:task" ;;   # T2-D129 continuous-session crops (MAE-only; 'task' = a non-empty label stand-in)
    dreyer)  p="$D/export_dreyer2023_wide_v1:trig_description" ;;
    *)       p="$D/cpt_pools/${a}_v1:event" ;;
  esac
  [ -s "${p%:*}/meta.json" ] || { echo "!! pool $a missing (${p%:*})"; s1_finish 2 "$OUT" 'none' 1; }
  POOLS+=(--pool "$p")
done
echo "### pools: ${POOLS[*]}"
# PREFT_FM (S1_EXPORT): st_large (default) starts from the D89b CPT checkpoint; any other FM (T2-D106: mor_p1v2) from
# its pinned root checkpoint (t2p_preft --init-ckpt empty); its input contract comes in the extra args (--align / --calib)
FM="${PREFT_FM:-st_large}"
INIT=()
[ "$FM" = st_large ] && INIT=(--init-ckpt "$S1_ROOT/runs/ni26/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth")
# T2-D455: PREFT_INIT (S1_EXPORT) = another runs/ni26 CPT dir to start from (seen1: ni26-t2p-cpt-d111-with, with --for-protocol seen1)
[ -n "${PREFT_INIT:-}" ] && [ "${PREFT_INIT}" != root ] && INIT=(--init-ckpt "$S1_ROOT/runs/ni26/$PREFT_INIT/checkpoint-cpt.pth")
# T2-D529: PREFT_INIT=root = start from the PINNED PUBLIC root checkpoint of --fm (t2p_preft's empty --init-ckpt; K = 9 members never see a CPT)
[ "${PREFT_INIT:-}" = root ] && INIT=()
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_preft.py" --fm "$FM" "${INIT[@]}" \
    --names-from "$D/export_proteus_v1" "${POOLS[@]}" --out "$OUT" "${EXTRA[@]}"
rc=$?
s1_finish "$rc" "$OUT" 'checkpoint-cpt.pth' 1
