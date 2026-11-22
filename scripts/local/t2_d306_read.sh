#!/usr/bin/env bash
# T2-D306 read (local CPU), as pre-registered: the softclip arms vs T2-D296's same-seed local plain run (deterministic), in the MOR-free package
# weights (0.4444 / 0.3333 / 0.2222 with stl_rcll). Arms sc_ch / sc_glob = both CNN families softclipped; each a candidate iff bar d141 AND new >= 0
# AND not killed (kit val delta over indices 0-5 < -0.01); both -> the higher mix.   mem-guard bash scripts/local/t2_d306_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
[ -f "$O/t2_d296/spec.json" ] || { echo "t2_d296's spec / routes are missing"; exit 2; }
mkdir -p "$O/t2_d306"
for m in ch glob; do
  R=$S/ni26-t2p-d306-$m-5090/runs
  [ -d "$R" ] || { echo "no runs for $m"; exit 2; }
  for p in kit seen1; do
    blend --runs "$R" --protocols $p --out "$O/t2_d306/route_${m}_fbc" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds $(seq -s, 204 215)
    blend --runs "$R" --protocols $p --out "$O/t2_d306/route_${m}_sh" --model shallow --align ea --recipe r1_pulx_vp2_e180_vr --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
done
$PY -I - "$O/t2_d296/spec.json" "$O/t2_d306/spec.json" "$O/t2_d306" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "stl_rcll")}
for md in ("ch", "glob"):
    m[f"fbc_{md}"] = {"offset": 204, **{p: f"{D}/route_{md}_fbc/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75fb3__s{{s}}.npz" for p in ("kit", "seen1")}}
    m[f"sh_{md}"] = {"offset": 168, **{p: f"{D}/route_{md}_sh/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda f, s: {f: 0.4 / 0.9, s: 0.3 / 0.9, "stl_rcll": 0.2 / 0.9}
spec = dict(n=12, anchor="c_nm", bar="d141", members=m, arms={"c_nm": w("fbc_rc75", "sh_vr"), "sc_ch": w("fbc_ch", "sh_ch"), "sc_glob": w("fbc_glob", "sh_glob")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_nm = T2-D296's local plain CNNs + stl_rcll, MOR-free weights; arms sc_ch / sc_glob")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d306/spec.json" --out "$O/t2_d306/read.json" > "$O/t2_d306/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d306/read.json" <<'PYEOF' | tee "$O/t2_d306/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("sc_ch", "sc_glob"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_nm", i)] for i in range(6)) / 6
    ok[a] = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_nm: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok[a] else "no"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> the twin (W9's template + a softclip key) on the user's word" if best else "NOT ADOPTED (no arm passes)")
PYEOF
