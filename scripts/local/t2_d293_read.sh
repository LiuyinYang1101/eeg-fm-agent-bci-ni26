#!/usr/bin/env bash
# T2-D293 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: user-adversarial (DANN) pooled CNNs, lambda 0.1 (ua1) /
# 0.3 (ua3) on both families (FBCNet rc75 line, seeds 204-215; ShallowNet vr line, seeds 168-179), copies = the lines', at w4321 with
# stl_rcll + mor, vs T2-D253's anchor c_rcll = {fbc_rc75 (T2-D231), sh_vr (T2-D215r), stl_rcll, mor}. Single-family arms fua* / sua* descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 (both -> the higher mix); KILL (the arm) iff its kit val delta over indices 0-5 < -0.01.
# Adoption needs the T2-D293r replicate on T2-D247r's seeds (mix > 0, new >= 0).
#   mem-guard bash scripts/local/t2_d293_read.sh      (after `s1 fetch` of ni26-t2p-d293-{fbc-kit,fbc-seen1,sh}-<id>)
#   LOCAL=1 mem-guard bash scripts/local/t2_d293_read.sh   (the 5090 run, scripts/local/t2_d293_local.sh: the anchor's two CNN members are the
#      lambda 0 controls re-run in the same dirs at the same seeds, so the read is local vs local; spec_local.json, *_local outputs)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
L=${LOCAL:-0}; X=$([ "$L" = 1 ] && echo _local); SPEC="$O/t2_d293/spec$X.json"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d231/route_fbc_r1_pulxx_vp2_e180_vr_rc75/kit__*__s204.npz "$O"/t2_d215r/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s168.npz >/dev/null || { echo "the anchor's route files are missing"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d293-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d293-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc75_ua1 r1_pulxx_vp2_e180_vr_rc75_ua3 $([ "$L" = 1 ] && echo r1_pulxx_vp2_e180_vr_rc75); do
    blend --runs "$R" --protocols $p --out "$O/t2_d293/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr_ua1 r1_pulx_vp2_e180_vr_ua3 $([ "$L" = 1 ] && echo r1_pulx_vp2_e180_vr); do
    blend --runs "$R" --protocols $p --out "$O/t2_d293/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
if [ "$L" = 1 ]; then
  $PY -I - "$O/t2_d293/spec.json" "$SPEC" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1]))
for m, a, b in (("fbc_rc75", "/t2_d231/route_fbc_", "/t2_d293/route_fbc_"), ("sh_vr", "/t2_d215r/route_sh_", "/t2_d293/route_sh_")):
    for p in ("kit", "seen1"):
        assert a in J["members"][m][p], (m, p)
        J["members"][m][p] = J["members"][m][p].replace(a, b)
json.dump(J, open(sys.argv[2], "w"), indent=1)
print("spec_local: fbc_rc75 / sh_vr = the local lambda 0 controls (t2_d293 routes)")
PYEOF
fi
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$SPEC" --out "$O/t2_d293/read$X.json" > "$O/t2_d293/read$X.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d293/read$X.json" <<'PYEOF' | tee "$O/t2_d293/verdict$X.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("ua1", "ua3", "fua1", "fua3", "sua1", "sua3"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    ok[a] = a in ("ua1", "ua3") and bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok[a] else "no")
             if a in ("ua1", "ua3") else " (descriptive)"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch the T2-D293r replicate (T2-D247r's seeds) before any adoption" if best else "NOT ADOPTED (no arm passes)")
PYEOF
