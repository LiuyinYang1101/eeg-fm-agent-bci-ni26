#!/usr/bin/env bash
# T2-D296 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: CNN test-time averaging over the vp bank (backlog 71).
# One training per seed wrote the plain file and its <recipe>tv8 twin (scripts/local/t2_d296_local.sh), so the anchor c_rcll uses THIS run's
# plain CNN files (paired) + stl_rcll + mor (T2-D293's spec members), at w4321. Candidate arm: tv8 (both pooled CNNs TTA'd); ftv8 / stv8 descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 AND not killed (kit val delta over indices 0-5 < -0.01). Adoption needs the T2-D296r replicate
# (T2-D247r's seeds) and the template + kit twin passing by Oct 31 (a predict-path change: Finish on warm-up by Nov 1).
#   mem-guard bash scripts/local/t2_d296_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-d296-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d296-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
mkdir -p "$O/t2_d296"
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc75 r1_pulxx_vp2_e180_vr_rc75tv8; do
    blend --runs "$R" --protocols $p --out "$O/t2_d296/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr r1_pulx_vp2_e180_vrtv8; do
    blend --runs "$R" --protocols $p --out "$O/t2_d296/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
$PY -I - "$O/t2_d293/spec.json" "$O/t2_d296/spec.json" "$O/t2_d296" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("stl_rcll", "mor")}
for name, fam, rec, off, stem, sfx in (("fbc_rc75", "fbc", "r1_pulxx_vp2_e180_vr_rc75", 204, "fbcnet_bboth__none", "fb3"),
                                       ("fbc_tv8", "fbc", "r1_pulxx_vp2_e180_vr_rc75tv8", 204, "fbcnet_bboth__none", "fb3"),
                                       ("sh_vr", "sh", "r1_pulx_vp2_e180_vr", 168, "shallow__ea", "fb2"),
                                       ("sh_tv8", "sh", "r1_pulx_vp2_e180_vrtv8", 168, "shallow__ea", "fb2")):
    m[name] = {"offset": off, **{p: f"{D}/route_{fam}_{rec}/{p}__{stem}__n00__{rec}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda f, s: {f: 0.4, s: 0.3, "stl_rcll": 0.2, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=m,
            arms={"c_rcll": w("fbc_rc75", "sh_vr"), "tv8": w("fbc_tv8", "sh_tv8"), "ftv8": w("fbc_tv8", "sh_vr"), "stv8": w("fbc_rc75", "sh_tv8")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_rcll = this run's plain CNN files + stl_rcll + mor; arms tv8 (candidate), ftv8 / stv8 (descriptive)")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d296/spec.json" --out "$O/t2_d296/read.json" > "$O/t2_d296/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d296/read.json" <<'PYEOF' | tee "$O/t2_d296/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = False
for a in ("tv8", "ftv8", "stv8"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    if a == "tv8":
        ok = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok else "no")
             if a == "tv8" else " (descriptive)"))
print("VERDICT:", "candidate tv8 -> the T2-D296r replicate (T2-D247r's seeds) + the template / kit twin by Oct 31, before any adoption" if ok
      else "NOT ADOPTED (tv8 does not pass)")
PYEOF
