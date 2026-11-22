#!/usr/bin/env bash
# T2-D350 J-CPT arm A: ST-EEGFormer MAE continued pretraining on the cross-track ni26 pools (t2p_cpt.py --jcpt-spec). $1 = the spec in the
# repo (pool paths 'rel' under $S1_ROOT/data/ni26); every pool is staged node-locally and the staged spec gets 'path'. Args after $1 go to
# t2p_cpt.py (--out is set here; PROTEUS = data/ni26/t2_proteus/export_proteus_v1 with --proteus-pool jcpt).
#   S1_PURPOSE="..." S1_EXPORT=RUN_NAME=ni26-t2p-jcpt-A s1 submit ni26 tracks/t2_bci/sofia/t2p_jcpt.slurm --time=10:00:00 --job-name=... \
#       -- tracks/t2_bci/sofia/jcpt/jcpt_A_spec.json --fm st_large --steps 20000 --batch 64 --accum 8 --lr 2e-5 --schedule const \
#          --wd 1e-5 --mask-ratio 0.75 --save-at 2000,5000,10000,20000 --save-tag jA --eval-every 2500 --token-scale
# (the pinned ST-L is weights-only: --lr / --wd / --mask-ratio are required; v1 = 6.58e-5 / 1e-5 / 0.75)
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGE="$S1_JOBTMP/jcpt"; mkdir -p "$STAGE"
SPEC="$S1_CODE/$1"; shift
cp -r "$D/t2_proteus/export_proteus_v1" "$STAGE/"
python - "$SPEC" "$D" "$STAGE" <<'PY' || exit 2
import json, shutil, sys, time
from pathlib import Path
spec, D, S = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
names = [Path(p["rel"]).name for p in spec["pools"]]
assert len(set(names)) == len(names), names
for p in spec["pools"]:
    t0 = time.time(); src = D / p["rel"]; dst = S / Path(p["rel"]).name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("emb*.npy", "bids", "boundary*"))
    p["path"] = str(dst)
    print(f"[jcpt stage] {p['rel']} -> {dst} ({sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) / 2**30:.1f} GB, {time.time() - t0:.0f} s)", flush=True)
(S / "spec.json").write_text(json.dumps(spec, indent=1))
PY
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_cpt.py" --proteus "$STAGE/export_proteus_v1" --proteus-pool jcpt --jcpt-spec "$STAGE/spec.json" \
  --out "$OUT" "$@" &
PID=$!
trap 'kill -TERM "$PID"' USR1
rc=0; wait "$PID" || rc=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done
cp "$STAGE/spec.json" "$OUT/jcpt_spec_staged.json" 2>/dev/null
if grep -q -- "--bench" <<< "$*"; then s1_finish "$rc" "$OUT" 'bench.json' 1; fi
s1_finish "$rc" "$OUT" 'checkpoint-cpt.pth' 1
