#!/usr/bin/env bash
# T2-D221 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: CNN width x 2 under vp2-180 + vr (FBCNet 64 spatial filters per band,
# ShallowNet 80 / 80 filters), on T2-D215r's seeds (FBCNet 204-215, ShallowNet 168-179), vs T2-D215r's vr runs (c_vr), at w4321. Arms w2 (both wider),
# fw2 (FBCNet only), sw2 (ShallowNet only). Adopt-candidate iff bar d141 AND new >= 0 (several -> the highest mix); adoption needs the T2-D221r replicate on
# T2-D215's seeds (mix > 0, new >= 0) and the package's runtime bar (<= 18.1 ms per window).
#   mem-guard bash scripts/local/t2_d221_read.sh      (after scripts/local/t2_d215r_read.sh, which writes the reference's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215r/route_fbc_r1_pulxx_vp2_e180_vr/kit__*__s204.npz "$O"/t2_d215r/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s168.npz >/dev/null || { echo "run t2_d215r_read.sh first"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d221-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d221-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  blend --runs "$R" --protocols $p --out "$O/t2_d221/route_fbc_f64" --model fbcnet_bboth_f64 --recipe r1_pulxx_vp2_e180_vr --seeds $(seq -s, 204 215)
done
R=$(one sh) || exit 2
for p in kit seen1; do
  blend --runs "$R" --protocols $p --out "$O/t2_d221/route_sh_w2" --model shallow_w2 --align ea --recipe r1_pulx_vp2_e180_vr --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d221/spec.json" --out "$O/t2_d221/read.json" > "$O/t2_d221/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d221/read.json" <<'PYEOF' | tee "$O/t2_d221/verdict.txt"
import json, sys
A = json.load(open(sys.argv[1]))["arms"]
ok = {}
for a in ("w2", "fw2", "sw2"):
    r = A[a]
    ok[a] = bool(r["met"]) and r["new"]["d"] >= 0
    print(f"{a} vs c_vr: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> "
          + ("CANDIDATE" if ok[a] else "no"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D221r replicate (T2-D215's seeds) before any adoption" if best else "NOT ADOPTED (no arm passes)")
PYEOF
