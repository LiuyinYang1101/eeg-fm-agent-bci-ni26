#!/usr/bin/env bash
# T2-D294 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: FM -> CNN distillation (alpha 0.5, T 2) on both families,
# kd = the PROTEUS windows, kdx = + 12,000 unlabelled external windows (FBCNet rc75 line, seeds 204-215; ShallowNet vr line, seeds 168-179), copies =
# the lines', at w4321 with stl_rcll + mor, vs T2-D253's anchor c_rcll = {fbc_rc75 (T2-D231), sh_vr (T2-D215r), stl_rcll, mor}. fkd* / skd* descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 (both -> the higher mix); KILL (the arm) iff its kit val delta over indices 0-5 < -0.01.
# Adoption needs the T2-D294r replicate on T2-D247r's seeds (mix > 0, new >= 0).
#   mem-guard bash scripts/local/t2_d294_read.sh      (after `s1 fetch` of ni26-t2p-d294-{fbc-kit,fbc-seen1,sh}-<id>)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d231/route_fbc_r1_pulxx_vp2_e180_vr_rc75/kit__*__s204.npz "$O"/t2_d215r/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s168.npz >/dev/null || { echo "the anchor's route files are missing"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d294-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d294-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc75_kd r1_pulxx_vp2_e180_vr_rc75_kdx; do
    blend --runs "$R" --protocols $p --out "$O/t2_d294/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr_kd r1_pulx_vp2_e180_vr_kdx; do
    blend --runs "$R" --protocols $p --out "$O/t2_d294/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d294/spec.json" --out "$O/t2_d294/read.json" > "$O/t2_d294/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d294/read.json" <<'PYEOF' | tee "$O/t2_d294/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("kd", "kdx", "fkd", "fkdx", "skd", "skdx"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    ok[a] = a in ("kd", "kdx") and bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok[a] else "no")
             if a in ("kd", "kdx") else " (descriptive)"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D294r replicate (T2-D247r's seeds, teacher seed 12) before any adoption" if best else "NOT ADOPTED (no arm passes)")
PYEOF
