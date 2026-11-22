#!/usr/bin/env bash
# T2-D295 read (backlog 65, local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: (a) FBCNet re-coloured toward the external
# covariance bank (rcx75; ShallowNet stays vr) and (b) a rigid cap shift of 10 mm RMS per virtual participant on both families (cap10), FBCNet
# rc75 line seeds 204-215, ShallowNet vr line seeds 168-179, copies = the lines', at w4321 with stl_rcll + mor, vs c_rcll. fcap / scap descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 (both -> the higher mix); KILL (the arm) iff its kit val delta over indices 0-5 < -0.01.
# Adoption needs the T2-D295r replicate on T2-D247r's seeds (mix > 0, new >= 0).
#   mem-guard bash scripts/local/t2_r65_read.sh      (after `s1 fetch` of ni26-t2p-r65-{fbc-kit,fbc-seen1,sh}-<id>)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d231/route_fbc_r1_pulxx_vp2_e180_vr_rc75/kit__*__s204.npz "$O"/t2_d215r/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s168.npz >/dev/null || { echo "the anchor's route files are missing"; exit 2; }
one() { local d=( "$S"/ni26-t2p-r65-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for r65-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rcx75 r1_pulxx_vp2_e180_vr_rc75_cap10; do
    blend --runs "$R" --protocols $p --out "$O/t2_r65/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  r=r1_pulx_vp2_e180_vr_cap10
  blend --runs "$R" --protocols $p --out "$O/t2_r65/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_r65/spec.json" --out "$O/t2_r65/read.json" > "$O/t2_r65/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_r65/read.json" <<'PYEOF' | tee "$O/t2_r65/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("rcx", "cap", "fcap", "scap"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    ok[a] = a in ("rcx", "cap") and bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok[a] else "no")
             if a in ("rcx", "cap") else " (descriptive)"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D295r replicate (T2-D247r's seeds) before any adoption" if best else "NOT ADOPTED (no arm passes)")
PYEOF
