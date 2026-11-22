#!/usr/bin/env bash
# T2-D318 on one GPU of zen5_vis (free; the user's standing choice for quick tests): H-a (t2_d318_heeg.py: HEEGNet + its EEGNet
# control) and H-b (t2p_bench.py: FBCNet bboth, linear head vs the Lorentz-MLR head, rc75 recipe, pooled), kit + seen1, seeds 0-11.
# Six processes share the GPU: H-a heeg/kit, heeg/seen1, eegnetctl/both; H-b 3 bench shards. 96 result files.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_d318.slurm --time=06:00:00 --job-name=ni26-t2p-d318
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
EXPORT="${T2P_EXPORT:-$D/export_proteus_v1}"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -e "$EXPORT/meta.json" ] || { echo "no export at $EXPORT"; s1_finish 3 "$OUT" 'none' 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
STAGE="${S1_JOBTMP:-/tmp}/t2p_export"; mkdir -p "$STAGE"; cp "$EXPORT"/{X.npy,index.csv,meta.json} "$STAGE/"
S=0,1,2,3,4,5,6,7,8,9,10,11
HA="$S1_CODE/tracks/t2_bci/scripts/t2_d318_heeg.py"
mkdir -p "$OUT/ha/runs" "$OUT/hb/runs"
pids=()
python -u "$HA" --export "$STAGE" --out "$OUT/ha/runs" --models heeg --protocols kit --seeds "$S" > "$OUT/ha_heeg_kit_${SLURM_JOB_ID}.log" 2>&1 & pids+=($!)
python -u "$HA" --export "$STAGE" --out "$OUT/ha/runs" --models heeg --protocols seen1 --seeds "$S" > "$OUT/ha_heeg_seen1_${SLURM_JOB_ID}.log" 2>&1 & pids+=($!)
python -u "$HA" --export "$STAGE" --out "$OUT/ha/runs" --models eegnetctl --protocols kit,seen1 --seeds "$S" > "$OUT/ha_eg_${SLURM_JOB_ID}.log" 2>&1 & pids+=($!)
for i in 0 1 2; do
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_bench.py" --export "$STAGE" --out "$OUT/hb/runs" --shard "$i" --nshards 3 \
    --protocols kit,seen1 --models fbcnet_bboth,fbcnet_bboth_lmlr --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds "$S" \
    > "$OUT/hb_shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' USR1 TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
grep -h "\[d318\].*test" "$OUT"/ha_*_"$SLURM_JOB_ID".log | tail -n 60
grep -h -E "Traceback|Error" "$OUT"/h?_*_"$SLURM_JOB_ID".log | head -20
echo "### $(ls "$OUT"/ha/runs/*.npz 2>/dev/null | wc -l) H-a + $(ls "$OUT"/hb/runs/*.npz 2>/dev/null | wc -l) H-b result files (expected 48 + 48)"
s1_finish "$rc" "$OUT" 'h?/runs/*.npz' 96
