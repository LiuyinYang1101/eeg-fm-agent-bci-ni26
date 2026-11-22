#!/usr/bin/env bash
# T2-D638 identity (registered amendment 2026-11-19 15:43:14, BEFORE any cell number; zero GPU): the NO-OP root t2_d638_id_860 through the
# SAME chain as every cell (route splice -> the base teacher -> fbteach -> lpteach -> the lp root) must reproduce W21's lab chain on set E
# BITWISE: base teacher == t2_d615/teacher_p1, fbteach == t2_d571_fbw20d557_860's, lpteach == t2_d571_lpw20d557_860's, and the mix
# (t2_lab mix --base w20 --idea w20_fb, PRE_R, CKS 60,100,150) == the reference root's mix. A failure = STOP (no cell is read).
#   mem-guard bash scripts/local/t2_d638_identity.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done          # = t2_d571_local.sh (F7 / F8): no inherited NI26_* env
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
O=$HOME/workspace/outputs/neural_interface_challenge
L=$O/t2_d638/identity.log
mkdir -p "$O/t2_d638"
step() { echo "### $(date +%T) $1" >> "$L"; shift; "$@" >> "$L" 2>&1; }   # set -e: a failing step stops the script
echo "### $(date +%T) d638 identity start (git $(git rev-parse --short HEAD))" >> "$L"
step "base id" $PY -u scripts/local/t2_d638_splice.py base id
step "fbteach id" env NI26_D559_ROOT=t2_d638_id_860 NI26_D559_CKS=60,100,150 $PY -u scripts/local/t2_lab.py fbteach --set E \
  --out "$O/t2_d638_fbid_860/teacher_p1" --fm-rows own --proto 0.5
step "lpteach id" $PY -u scripts/local/t2_lab.py lpteach --set E --out "$O/t2_d638_lpid_860/teacher_p1" --src "$O/t2_d638_fbid_860/teacher_p1" --alpha 0.5 --k 10
step "lproot id" $PY -u scripts/local/t2_d638_splice.py lproot id
step "check teachers" $PY -u scripts/local/t2_d638_splice.py check-teachers
for side in ref id; do
  r=$([ $side = ref ] && echo t2_d571_lpw20d557_860 || echo t2_d638_lpid_860)
  out=$O/t2_d638/mix_${side}_chrono.npz
  [ -f "$out" ] || step "mix $side" env NI26_LAB_PRE_R=1 NI26_D559_ROOT=$r NI26_D559_CKS=60,100,150 $PY -u scripts/local/t2_lab.py mix --set E \
    --base w20 --idea w20_fb --out "$out"
done
step "compare mixes" $PY - "$O/t2_d638/mix_ref_chrono.npz" "$O/t2_d638/mix_id_chrono.npz" <<'PY'
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ks = sorted(k for k in a.files if k.startswith(("ix_", "mx_")))
assert ks == sorted(k for k in b.files if k.startswith(("ix_", "mx_"))) and len(ks) == 60, (len(ks),)
d = max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in ks)
print(f"[d638 identity] the id root's mix vs the reference root's mix: {len(ks) // 2} fold x draw blocks, max |d| {d:.3e}")
if d != 0.0:
    raise SystemExit("STOP: the identity mix differs from the reference (no cell may be read)")
print("[d638 identity] PASS: the no-op root reproduces W21's lab chain on set E bitwise (base / fb / lp teachers, mix)")
PY
echo "### $(date +%T) d638 identity done" >> "$L"
grep -E "d638 identity|d638 splice|STOP" "$L" | tail -8
