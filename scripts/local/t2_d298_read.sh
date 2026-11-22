#!/usr/bin/env bash
# T2-D298 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md (backlog 72): the per-user copies through the vp bank (pu_vp 1.0,
# T2-D200's pv10 on the current line) + the pooled CNNs' test-time averaging (tv8, T2-D296), on FBCNet rc75 (204-215) + ShallowNet vr (168-179).
# Anchor c_rcll = T2-D296's LOCAL plain CNN files + stl_rcll + mor (w4321); the pv10 run's pooled nets are checked equal to the anchor's.
# Candidate arm: pvtv (TTA pooled + augmented copies, both families); pv (plain pooled + augmented copies) and tv8 (= T2-D296) descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 AND not killed (kit val delta over indices 0-5 < -0.01); then the T2-D298r replicate; the tv8 half is a
# predict-path change (the user's twin decision, Finish on warm-up by Nov 1), the pv10 half weights only.
#   mem-guard bash scripts/local/t2_d298_read.sh      (after scripts/local/t2_d298_local.sh; needs t2_d296's routes)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for $1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
[ -f "$O/t2_d296/spec.json" ] || { echo "t2_d296's spec / routes are missing"; exit 2; }
mkdir -p "$O/t2_d298"
$PY -I - "$(one d296-fbc-kit)" "$(one d298-fbc-kit)" "$(one d296-sh)" "$(one d298-sh)" <<'PYEOF' || exit 2
import sys, numpy as np
a, b, c, d = sys.argv[1:]
for x, y, f0, f1 in ((a, b, "kit__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75__s{}.npz", "kit__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_pv10__s{}.npz", ),
                     (c, d, "seen1__shallow__ea__n00__r1_pulx_vp2_e180_vr__s{}.npz", "seen1__shallow__ea__n00__r1_pulx_vp2_e180_vr_pv10__s{}.npz")):
    s = 204 if "fbcnet" in f0 else 168
    m = max(float(np.abs(np.load(f"{x}/{f0.format(s + i)}")["prob_last"] - np.load(f"{y}/{f1.format(s + i)}")["prob_last"]).max()) for i in range(12))
    print(f"pooled nets, pv10 run vs the anchor run ({f0.split('__')[1]}): max |dp| {m:.1e}")
    assert m < 1e-4, "the pv10 run's pooled nets must be the anchor's"
PYEOF
for p in kit seen1; do
  R=$(one d298-fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc75_pv10 r1_pulxx_vp2_e180_vr_rc75_pv10tv8; do
    blend --runs "$R" --protocols $p --out "$O/t2_d298/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
  done
done
R=$(one d298-sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr_pv10 r1_pulx_vp2_e180_vr_pv10tv8; do
    blend --runs "$R" --protocols $p --out "$O/t2_d298/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
$PY -I - "$O/t2_d296/spec.json" "$O/t2_d298/spec.json" "$O/t2_d298" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "fbc_tv8", "sh_tv8", "stl_rcll", "mor")}
for name, fam, rec, off, stem, sfx in (("fbc_pv", "fbc", "r1_pulxx_vp2_e180_vr_rc75_pv10", 204, "fbcnet_bboth__none", "fb3"),
                                       ("fbc_pvtv", "fbc", "r1_pulxx_vp2_e180_vr_rc75_pv10tv8", 204, "fbcnet_bboth__none", "fb3"),
                                       ("sh_pv", "sh", "r1_pulx_vp2_e180_vr_pv10", 168, "shallow__ea", "fb2"),
                                       ("sh_pvtv", "sh", "r1_pulx_vp2_e180_vr_pv10tv8", 168, "shallow__ea", "fb2")):
    m[name] = {"offset": off, **{p: f"{D}/route_{fam}_{rec}/{p}__{stem}__n00__{rec}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda f, s: {f: 0.4, s: 0.3, "stl_rcll": 0.2, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=m,
            arms={"c_rcll": w("fbc_rc75", "sh_vr"), "pvtv": w("fbc_pvtv", "sh_pvtv"), "pv": w("fbc_pv", "sh_pv"), "tv8": w("fbc_tv8", "sh_tv8")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_rcll = T2-D296's local plain CNN files + stl_rcll + mor; arms pvtv (candidate), pv / tv8 (descriptive)")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d298/spec.json" --out "$O/t2_d298/read.json" > "$O/t2_d298/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d298/read.json" <<'PYEOF' | tee "$O/t2_d298/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = False
for a in ("pvtv", "pv", "tv8"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    if a == "pvtv":
        ok = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok else "no")
             if a == "pvtv" else " (descriptive)"))
print("VERDICT:", "candidate pvtv -> the T2-D298r replicate (T2-D247r's seeds) before any adoption; the tv8 half = the user's twin decision" if ok
      else "NOT ADOPTED (pvtv does not pass)")
PYEOF
