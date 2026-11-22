#!/usr/bin/env bash
# T2-D318 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md.
#   H-a package: anchor c_nm = the MOR-free package (fbc_rc75 204+i / sh_vr 168+i / stl_rcll i = 0.4444 / 0.3333 / 0.2222, T2-D305's
#                members) vs hg = 0.4 / 0.3 / 0.2 + HEEGNet seed i at 0.1, and eg = the same with the EEGNet control (beside).
#   H-a member:  HEEGNet alone vs EG alone (mix > 0 with >= 8 / 12 up); both alone vs the shipped FBCNet member alone (reported).
#   H-b stage 1: FBCNet + Lorentz MLR alone vs its linear twin alone (pooled, same job): d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_d318_read.sh [ha|hb|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
which="${1:-all}"
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
RA=${RA:-$O/sofia_s1/ni26-t2p-d318-ha-5090/runs}   # sofia: RA=.../ni26-t2p-d318-<id>/ha/runs
RB=${RB:-$O/sofia_s1/ni26-t2p-d318-hb-5090/runs}
mkdir -p "$O/t2_d318"
$PY -I - "$O/t2_d253/spec.json" "$O/t2_d318" "$RA" "$RB" "$which" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); out, RA, RB, which = sys.argv[2:6]
base = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "stl_rcll")}
mem = lambda R, stem: {"offset": 0, **{p: f"{R}/{p}__{stem}__s{{s}}.npz" for p in ("kit", "seen1")}}   # noqa: E731
hg, eg = mem(RA, "heeg__ea__n00__heeg_ref"), mem(RA, "eegnetctl__ea__n00__heeg_ref")
c_nm = {"fbc_rc75": 0.4 / 0.9, "sh_vr": 0.3 / 0.9, "stl_rcll": 0.2 / 0.9}
specs = {}
if which in ("ha", "all"):
    specs["ha_pkg"] = dict(n=12, anchor="c_nm", bar="d141", members=dict(base, hg=hg, eg=eg),
                           arms={"c_nm": c_nm, "hg": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "hg": 0.1},
                                 "eg": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "eg": 0.1}})
    specs["ha_mem"] = dict(n=12, anchor="eg_alone", bar="d141", members=dict(hg=hg, eg=eg, fbc_rc75=base["fbc_rc75"]),
                           arms={"eg_alone": {"eg": 1.0}, "hg_alone": {"hg": 1.0}, "fbc_alone": {"fbc_rc75": 1.0}})
if which in ("hb", "all"):
    lin = mem(RB, "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75")
    lm = mem(RB, "fbcnet_bboth_lmlr__none__n00__r1_pulxx_vp2_e180_vr_rc75")
    specs["hb_mem"] = dict(n=12, anchor="lin_alone", bar="d141", members=dict(lin=lin, lm=lm),
                           arms={"lin_alone": {"lin": 1.0}, "lm_alone": {"lm": 1.0}})
for k, s in specs.items():
    json.dump(s, open(f"{out}/spec_{k}.json", "w"), indent=1)
print("specs:", ", ".join(specs))
PYEOF
for k in ha_pkg ha_mem hb_mem; do
  [ -f "$O/t2_d318/spec_$k.json" ] || continue
  $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d318/spec_$k.json" --out "$O/t2_d318/read_$k.json" > "$O/t2_d318/read_$k.txt" 2>&1
  echo "$k rc $?"
done
$PY -I - "$O/t2_d318" "$which" <<'PYEOF' | tee "$O/t2_d318/verdict_${which}.txt"
import json, os, sys
d, which = sys.argv[1:3]
L = lambda k: json.load(open(f"{d}/read_{k}.json")) if os.path.exists(f"{d}/read_{k}.json") else None   # noqa: E731
f = lambda r: ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new")) + f" (mix {r['mix']['up']} / 12)"  # noqa: E731
P, M, B = L("ha_pkg"), L("ha_mem"), L("hb_mem")
if P and M:
    A = P["arms"]; kv = {(r["arm"], r["i"]): r["val"] for r in P["rows"] if r["proto"] == "kit"}
    ok = {}
    for a in ("hg", "eg"):
        r = A[a]; kill = sum(kv[(a, i)] - kv[("c_nm", i)] for i in range(6)) / 6
        ok[a] = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
        print(f"{a} vs c_nm: {f(r)}; kit val idx 0-5 {kill:+.4f}{' KILLED' if kill < -0.01 else ''}; d141 {'met' if r['met'] else 'not met'}; "
              f"new >= 0 {r['new']['d'] >= 0} -> package bar {'PASS' if ok[a] else 'no'}")
    hm = M["arms"]["hg_alone"]
    mem_ok = hm["mix"]["d"] > 0 and hm["mix"]["up"] >= 8
    print(f"member: HEEGNet alone vs EG alone: {f(hm)} -> member bar {'PASS' if mem_ok else 'no'}")
    print(f"        FBCNet (shipped, with copies) alone vs EG alone: {f(M['arms']['fbc_alone'])}  (reported)")
    for a in ("hg_alone", "fbc_alone"):
        lv = M["arms"][a]["level"]; eg_kt = lv["kit_test"] - M["arms"][a]["kit_test"]["d"]
        print(f"        level {a}: " + ", ".join(f"{q} {v:.4f}" for q, v in lv.items()) + f"  (EG alone kit test {eg_kt:.4f})")
    print("H-a VERDICT:", "CANDIDATE -> the fresh-seed replicate (HEEGNet seeds 12-23, mix >= +0.003)" if ok["hg"] and mem_ok else "NOT A CANDIDATE -> HEEGNet closed for T2")
if B:
    r = B["arms"]["lm_alone"]
    passed = bool(r["met"]) and r["new"]["d"] >= 0
    print(f"H-b: FBCNet + Lorentz MLR alone vs linear twin alone: {f(r)}; d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0}")
    print("H-b VERDICT:", "stage 1 PASS -> stage 2 (copies + package) to pre-register" if passed else "stage 1 not met -> closed")
PYEOF
