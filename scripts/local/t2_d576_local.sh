#!/usr/bin/env bash
# T2-D576 (D561 stage 2 as INSIGHT ONLY; registered 2026-11-18 before any job): a FEEDER on the local 5090 (gpu-submit), ONE job at a time, each
# submitted only when the shared queue holds no running / queued job (so a priority job submitted meanwhile runs between steps):
#   step ctl  (Track3 F1): FBCNet's stack line WITHOUT syn re-run on lc9_0, seed 831 (the C chain's command) -> outputs/t2_d576/cnn_fbc_ctl/runs;
#             every array of each output == the chain's t2_d558/cnn_fbc/runs file of the same name (else the re-run noise is stated beside the read)
#   step g<k> (k 0..9): the fold-clean generator of lc9_<k> (t2p_meanflow.py on that fold's TRAINING windows: D413b's wfm30k config), then
#             eu100 + calib 3 + bank 8 x n_train (tag final3) -> outputs/t2_d576/bank_lc9_<k>_final3.npz
#   step arm: FBCNet's stack line + syn 0.2 (recipe r1_pulxx_vp2_e180_vr_rc75_mf20, --syn-bank outputs/t2_d576/bank_{protocol}_final3.npz) at
#             831-833 (copies + tv8c, 3 shards), then t2p_fpcalib blend -> outputs/t2_d576/route_fbc_syn
#   root:     outputs/t2_d576_syn_831 = every entry of t2_d571_fbw20d557_831 symlinked except route_fbc = the syn routed files under the
#             stack's names (rc75_mf20tv8c -> rc75tv8c)
#   mem-guard bash scripts/local/t2_d576_local.sh            (runs the whole feeder; resumable: a step whose outputs exist is skipped)
set -uo pipefail
cd "$(dirname "$0")/../.."
FILES="src/ni26 tracks/t2_bci/scripts scripts/local/t2_d576_local.sh"
[ -z "$(git status --porcelain -- $FILES)" ] || { echo "commit the D576 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
D=$O/t2_d576; mkdir -p "$D"; L=$D/feeder.log
SC="systemd-run --user --scope -p MemoryHigh=12G -p MemoryMax=16G --quiet --"
log() { echo "### $(date +%T) $*" | tee -a "$L"; }
idle() { while gpu-submit --queue 2>/dev/null | awk 'NR > 1 && ($2 == "running" || $2 == "queued")' | grep -q .; do sleep 30; done; }
run() {   # name, job text: wait for an idle queue, submit, wait for this job's own done / fail marker
  local nm=$1 txt=$2 j n0
  idle
  n0=$(grep -cE "step $nm rc=" "$L" 2>/dev/null)   # a re-run: only a marker written AFTER this submit counts (the log is append-only)
  j=$(mktemp "$D/job_$nm.XXXX.sh")
  printf 'set -uo pipefail; cd %q; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\nrc=0\n%s\necho "### $(date +%%T) step %s rc=$rc" >> %q\nexit $rc\n' \
    "$(pwd)" "$txt" "$nm" "$L" > "$j"
  log "submit $nm (git $(git rev-parse --short HEAD))"
  gpu-submit -L "neural_interface_challenge:t2-d576-$nm" bash "$j" > /dev/null
  until [ "$(grep -cE "step $nm rc=" "$L")" -gt "${n0:-0}" ]; do sleep 20; done
  grep -E "step $nm rc=" "$L" | tail -1 | grep -q "rc=0" || { log "!! step $nm failed"; exit 6; }
}
B="$PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$E'"
# ---- step ctl: the determinism control (Track3 F1) ----
if [ ! -f "$D/ctl.json" ]; then
  run ctl "$SC $B --out '$D/cnn_fbc_ctl/runs' --protocols lc9_0 --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds 831 --peruser --tta-vp 8 --tta-copies >> '$D/ctl.log' 2>&1 || rc=5"
  $PY - "$D/cnn_fbc_ctl/runs" "$O/t2_d558/cnn_fbc/runs" "$D/ctl.json" <<'PYEOF'
import glob, json, os, sys
import numpy as np
a_dir, b_dir, out = sys.argv[1:4]
res = {"files": 0, "arrays": 0, "identical": 0, "max_abs_diff": 0.0, "differ": []}
for f in sorted(glob.glob(os.path.join(a_dir, "*.npz"))):
    g = os.path.join(b_dir, os.path.basename(f))
    if not os.path.exists(g):
        continue
    res["files"] += 1
    za, zb = np.load(f), np.load(g)
    for k in za.files:
        if k in zb.files and za[k].dtype.kind == "f" and za[k].shape == zb[k].shape and za[k].size:
            d = float(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)).max())
            res["arrays"] += 1
            res["identical"] += int(d == 0.0)
            res["max_abs_diff"] = max(res["max_abs_diff"], d)
            if d > 0:
                res["differ"].append([os.path.basename(f), k, d])
json.dump(res, open(out, "w"), indent=1)
print(f"[d576] determinism control: {res['files']} files, {res['identical']} / {res['arrays']} float arrays identical, max |diff| {res['max_abs_diff']:.3e}; "
      f"differing keys: {sorted({k for _, k, _ in res['differ']})}")
PYEOF
  log "control: $(cat "$D/ctl.json" | tr -d '\n' | cut -c1-200)"
fi
# ---- steps g0..g9: the fold-clean generators ----
for k in 0 1 2 3 4 5 6 7 8 9; do
  [ -s "$D/bank_lc9_${k}_final3.npz" ] && continue
  MF="$SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$E' --out '$D' --protocol lc9_$k"
  run "g$k" "$MF --tag wfm30k --seed 0 --steps 30000 --bs 64 --lr 2e-4 --norm white --p-eq 1.0 --sample eu100 --calib 0 --bank-mult 1 >> '$D/gen_lc9_$k.log' 2>&1 || rc=5
[ \$rc -eq 0 ] && { $MF --tag final3 --from-ckpt '$D/gen_lc9_${k}_wfm30k.pt' --sample eu100 --calib 3 --bank-mult 8 >> '$D/gen_lc9_$k.log' 2>&1 || rc=6; }
[ \$rc -eq 0 ] && [ -s '$D/bank_lc9_${k}_final3.npz' ] || rc=7"
done
# ---- step arm: FBCNet + syn 0.2 at 831-833, routed ----
R="r1_pulxx_vp2_e180_vr_rc75_mf20"
if [ "$(ls "$D/route_fbc_syn" 2>/dev/null | wc -l)" -lt 30 ]; then
  A="--protocols lc9_0,lc9_1,lc9_2,lc9_3,lc9_4,lc9_5,lc9_6,lc9_7,lc9_8,lc9_9 --models fbcnet_bboth --aligns none --recipe $R --seeds 831,832,833 --peruser --tta-vp 8 --tta-copies --syn-bank '$D/bank_{protocol}_final3.npz'"
  run arm "mkdir -p '$D/cnn_fbc_syn/runs'
$SC $B --out '$D/cnn_fbc_syn/runs' --shard 0 --nshards 3 $A >> '$D/arm_0.log' 2>&1 &
$SC $B --out '$D/cnn_fbc_syn/runs' --shard 1 --nshards 3 $A >> '$D/arm_1.log' 2>&1 &
$SC $B --out '$D/cnn_fbc_syn/runs' --shard 2 --nshards 3 $A >> '$D/arm_2.log' 2>&1 &
wait
for p in lc9_0 lc9_1 lc9_2 lc9_3 lc9_4 lc9_5 lc9_6 lc9_7 lc9_8 lc9_9; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$E' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$D/cnn_fbc_syn/runs' --protocols \$p --out '$D/route_fbc_syn' --model fbcnet_bboth --align none --recipe ${R}tv8c --seeds 831,832,833 >> '$D/arm_route.log' 2>&1 || rc=7; done
[ \$(ls '$D/route_fbc_syn' 2>/dev/null | wc -l) -ge 30 ] || rc=8"
fi
# ---- the root ----
RT=$O/t2_d576_syn_831
if [ ! -e "$RT" ]; then
  mkdir -p "$RT/route_fbc"
  for x in "$O/t2_d571_fbw20d557_831"/*; do b=$(basename "$x"); [ "$b" = route_fbc ] || ln -s "$x" "$RT/$b"; done
  n=0
  for f in "$D/route_fbc_syn"/*rc75_mf20tv8cfb3__s*.npz; do ln -s "$f" "$RT/route_fbc/$(basename "$f" | sed 's/rc75_mf20tv8cfb3/rc75tv8cfb3/')"; n=$((n + 1)); done
  log "root $RT: $(ls "$RT" | wc -l) entries (the fbw20d557 root has $(ls "$O/t2_d571_fbw20d557_831" | wc -l)); route_fbc = $n syn files under the stack's names"
fi
log "D576 feeder done"
