#!/usr/bin/env bash
# T2-D233 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: ShallowNet's pooled training windows (its EA inputs) re-coloured toward a
# random other training user at 0.75 (sh75) / 1.0 (sh10) vs none, under vp2-180 + vr (copies = the vr line's), on T2-D215r's seeds (ShallowNet 168-179), at w4321 with
# the adopted FBCNet (T2-D231's rc75, seeds 204-215); anchor c_rc75 = {fbc_rc75, sh_vr (T2-D215r's), stl, mor}.
# Adopt-candidate iff bar d141 AND new >= 0 (both -> the higher mix); adoption needs the T2-D233r replicate on T2-D215's seeds (mix > 0, new >= 0).
#   mem-guard bash scripts/local/t2_d233_read.sh      (after t2_d215r_read.sh and t2_d231_read.sh, which write the reference's and rc75's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215r/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s168.npz "$O"/t2_d231/route_fbc_r1_pulxx_vp2_e180_vr_rc75/kit__*__s204.npz >/dev/null || { echo "run t2_d215r_read.sh and t2_d231_read.sh first"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d233-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d233-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
R=$(one sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr_rc75 r1_pulx_vp2_e180_vr_rc10; do
    blend --runs "$R" --protocols $p --out "$O/t2_d233/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d233/spec.json" --out "$O/t2_d233/read.json" > "$O/t2_d233/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d233/read.json" <<'PYEOF' | tee "$O/t2_d233/verdict.txt"
import json, sys
A = json.load(open(sys.argv[1]))["arms"]
ok = {}
for a in ("sh75", "sh10"):
    r = A[a]
    ok[a] = bool(r["met"]) and r["new"]["d"] >= 0
    print(f"{a} vs c_rc75: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> "
          + ("CANDIDATE" if ok[a] else "no"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D233r replicate (T2-D215's seeds) before any adoption" if best else "NOT ADOPTED (no arm passes)")
PYEOF
