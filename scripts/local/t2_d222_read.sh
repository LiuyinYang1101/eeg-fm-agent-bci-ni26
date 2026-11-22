#!/usr/bin/env bash
# T2-D222 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the T2-D221 near miss sw2 (ShallowNet 80 / 80 filters under
# vp2-180 + vr) confirmed on FRESH seeds = T2-D215's (ShallowNet 156-167; FBCNet = D215's vr runs 192-203, unchanged), vs T2-D215's vr runs (c_vr), at w4321.
# Adopted iff bar d141 AND new >= 0 on these 12 seeds alone (the D221 seeds picked the arm, so they are reported beside it, never pooled into the verdict),
# and then the rebuilt upload-5 package passes the D140 bars at <= 18.1 ms per window.
#   mem-guard bash scripts/local/t2_d222_read.sh      (after scripts/local/t2_d215_read.sh, which writes the reference's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215/route_fbc_r1_pulxx_vp2_e180_vr/kit__*__s192.npz "$O"/t2_d215/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s156.npz >/dev/null || { echo "run t2_d215_read.sh first"; exit 2; }
d=( "$S"/ni26-t2p-d222-sh-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d222-sh: ${d[*]}"; exit 2; }
for p in kit seen1; do
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d222/route_sh_w2" --model shallow_w2 --align ea --recipe r1_pulx_vp2_e180_vr --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d222/spec.json" --out "$O/t2_d222/read.json" > "$O/t2_d222/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d222/read.json" "$O/t2_d221/read.json" <<'PYEOF' | tee "$O/t2_d222/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1]))["arms"]["sw2r"]
q = json.load(open(sys.argv[2]))["arms"]["sw2"]
ok = bool(r["met"]) and r["new"]["d"] >= 0
f = lambda x: ", ".join(f"{k} {x[k]['d']:+.4f}" for k in ("mix", "kit_test", "seen", "new")) + f" (mix {x['mix']['up']} / 12)"
print(f"sw2r vs c_vr (fresh seeds 156-167): {f(r)}; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0}")
print(f"beside it, T2-D221 sw2 (seeds 168-179, the selection seeds): {f(q)}")
print("VERDICT:", "ADOPT sw2 -> rebuild upload 5's candidate with shallow_w2 and check <= 18.1 ms per window" if ok else "NOT ADOPTED (the ShallowNet width stays 40 / 40)")
PYEOF
