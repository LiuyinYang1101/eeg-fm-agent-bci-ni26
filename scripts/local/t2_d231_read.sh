#!/usr/bin/env bash
# T2-D231 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the re-colouring dose, FBCNet pooled windows re-coloured at 0.25 (rc25) /
# 0.75 (rc75) under vp2-180 + vr (ShallowNet = c_vr's; copies = the vr line's), on T2-D215r's seeds (FBCNet 204-215), vs T2-D215r's vr runs (c_vr), at w4321;
# T2-D230's rc5 / rc10 on the same seeds beside. A dose is a candidate to replace rc5 iff bar d141 AND new >= 0 AND its mix > rc5's (+0.0059);
# the best such -> the T2-D231r replicate on T2-D215's seeds before it replaces rc5.
#   mem-guard bash scripts/local/t2_d231_read.sh      (after t2_d215r_read.sh and t2_d230_read.sh, which write the reference's and rc5's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215r/route_fbc_r1_pulxx_vp2_e180_vr/kit__*__s204.npz "$O"/t2_d230/route_fbc_r1_pulxx_vp2_e180_vr_rc5/seen1__*__s204.npz >/dev/null || { echo "run t2_d215r_read.sh and t2_d230_read.sh first"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d231-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d231-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc25 r1_pulxx_vp2_e180_vr_rc75; do
    blend --runs "$R" --protocols $p --out "$O/t2_d231/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d231/spec.json" --out "$O/t2_d231/read.json" > "$O/t2_d231/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d231/read.json" <<'PYEOF' | tee "$O/t2_d231/verdict.txt"
import json, sys
A = json.load(open(sys.argv[1]))["arms"]
ref = A["rc5"]["mix"]["d"]
ok = {}
for a in ("rc25", "rc75", "rc5", "rc10"):
    r = A[a]
    ok[a] = a in ("rc25", "rc75") and bool(r["met"]) and r["new"]["d"] >= 0 and r["mix"]["d"] > ref
    print(f"{a} vs c_vr: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0}"
          + (f"; mix > rc5's {ref:+.4f} {r['mix']['d'] > ref} -> " + ("CANDIDATE" if ok[a] else "no") if a in ("rc25", "rc75") else " (T2-D230, descriptive)"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D231r replicate (T2-D215's seeds) before it replaces rc5" if best else "rc5 STAYS (no dose beats it)")
PYEOF
