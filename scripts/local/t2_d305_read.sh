#!/usr/bin/env bash
# T2-D305 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: MIRepNet as a 4th member of the MOR-free package. Members at index i:
# fbc_rc75 204+i, sh_vr 168+i, stl_rcll i (T2-D253's spec) + MIRepNet seed i (local runs, no copies). Anchor c_nm = the MOR-free mix (0.4 / 0.3 / 0.2
# renormalised = 0.4444 / 0.3333 / 0.2222); arms mr8 / mrbb = 0.4 / 0.3 / 0.2 + MIRepNet 0.1 (8-30 Hz / broadband): the MOR's old slot, one weight.
# Each arm: candidate iff bar d141 AND new >= 0 AND not killed (kit val delta over indices 0-5 < -0.01); both -> the higher mix.
#   mem-guard bash scripts/local/t2_d305_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/sofia_s1/ni26-t2p-d305-mirep-5090/runs
[ -d "$R" ] || { echo "no D305 runs at $R"; exit 2; }
mkdir -p "$O/t2_d305"
$PY -I - "$O/t2_d253/spec.json" "$O/t2_d305/spec.json" "$R" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); R = sys.argv[3]
m = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "stl_rcll")}
for name, model in (("mr8", "mirepnet"), ("mrbb", "mirepnet_bb")):
    m[name] = {"offset": 0, **{p: f"{R}/{p}__{model}__ea__n00__mirep_ft__s{{s}}.npz" for p in ("kit", "seen1")}}
spec = dict(n=12, anchor="c_nm", bar="d141", members=m,
            arms={"c_nm": {"fbc_rc75": 0.4 / 0.9, "sh_vr": 0.3 / 0.9, "stl_rcll": 0.2 / 0.9},
                  "mr8": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mr8": 0.1},
                  "mrbb": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mrbb": 0.1}})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_nm = the MOR-free package; arms mr8 / mrbb = + MIRepNet at the MOR's 0.1 slot")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d305/spec.json" --out "$O/t2_d305/read.json" > "$O/t2_d305/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d305/read.json" <<'PYEOF' | tee "$O/t2_d305/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("mr8", "mrbb"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_nm", i)] for i in range(6)) / 6
    ok[a] = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_nm: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok[a] else "no"))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: A[a]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> the Nov 1 twin (the user's go, if read by Oct 31 14:00)" if best else "NOT ADOPTED (no arm passes): MIRepNet dropped")
PYEOF
