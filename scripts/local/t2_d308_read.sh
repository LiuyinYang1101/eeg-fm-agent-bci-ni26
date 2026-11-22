#!/usr/bin/env bash
# T2-D308 read (local CPU), as pre-registered: virtual sessions (both CNN families + vs 0.5) vs T2-D296's same-seed local plain run, MOR-free package
# weights (0.4444 / 0.3333 / 0.2222 with stl_rcll); fvs / svs single-family descriptive. Candidate iff bar d141 AND new >= 0 AND not killed.
#   mem-guard bash scripts/local/t2_d308_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/sofia_s1/ni26-t2p-d308-vs-5090/runs
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
[ -f "$O/t2_d296/spec.json" ] && [ -d "$R" ] || { echo "missing t2_d296 spec or the D308 runs"; exit 2; }
mkdir -p "$O/t2_d308"
for p in kit seen1; do
  blend --runs "$R" --protocols $p --out "$O/t2_d308/route_fbc" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_vs5 --seeds $(seq -s, 204 215)
  blend --runs "$R" --protocols $p --out "$O/t2_d308/route_sh" --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_vs5 --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
done
$PY -I - "$O/t2_d296/spec.json" "$O/t2_d308/spec.json" "$O/t2_d308" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "stl_rcll")}
m["fbc_vs"] = {"offset": 204, **{p: f"{D}/route_fbc/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_vs5fb3__s{{s}}.npz" for p in ("kit", "seen1")}}
m["sh_vs"] = {"offset": 168, **{p: f"{D}/route_sh/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_vs5fb2__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda f, s: {f: 0.4 / 0.9, s: 0.3 / 0.9, "stl_rcll": 0.2 / 0.9}
spec = dict(n=12, anchor="c_nm", bar="d141", members=m, arms={"c_nm": w("fbc_rc75", "sh_vr"), "vs": w("fbc_vs", "sh_vs"), "fvs": w("fbc_vs", "sh_vr"), "svs": w("fbc_rc75", "sh_vs")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_nm = T2-D296's local plain CNNs + stl_rcll (MOR-free); arms vs (candidate), fvs / svs (descriptive)")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d308/spec.json" --out "$O/t2_d308/read.json" > "$O/t2_d308/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d308/read.json" <<'PYEOF' | tee "$O/t2_d308/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = False
for a in ("vs", "fvs", "svs"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_nm", i)] for i in range(6)) / 6
    if a == "vs":
        ok = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_nm: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok else "no") if a == "vs" else " (descriptive)"))
print("VERDICT:", "candidate vs -> one L_V draw rebuilt by Nov 4 on the user's word" if ok else "NOT ADOPTED (vs does not pass)")
PYEOF
