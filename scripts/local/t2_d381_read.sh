#!/usr/bin/env bash
# T2-D381 (a) read (pre-registered 2026-11-06 11:25; this script written before any D381 number): views -> members. PAIRED on the same nets
# (each bench run wrote plain / tv8 / tv8c): the tv8 twins (pooled net viewed, per-user copies plain) and the plain files, blended exactly as
# the shipped tv8c (fbc r 1/3 tag 3; sh / ifn r 0.5 tag 2 — the IFNet params checked 11:28 by a bitwise re-blend of seed 168).
#   spec_a (anchor nm35): V8 = fbc / sh / ifn tv8c -> tv8; V1 = -> plain.
#   spec_b (anchor S3f = M6's shape): S3f_V8 (tilt FBCNet 204-215 tv8 + sh / ifn tv8); then on S3f_V8 one ADD each (D377's constructions):
#     +EA6 (stl_ea of T2-D360r spread into the ST-L slot over 6), +SW2 (sw2, likewise), +USs (the ShallowNet share split in half with sh_us5 tv8).
# Screen: S3f_V8 >= -0.0015 typical vs S3f AND (b) frees >= 4.5 ms; an ADD arm >= +0.003 vs S3f, >= 9 / 12, P04 >= -0.010, new >= 0, <= 40 ms.
#   mem-guard bash scripts/local/t2_d381_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d381
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
for p in kit seen1; do
  for x in tv8 ""; do
    blend --runs "$S/ni26-t2p-d345r-ifnet-s168-5090/runs" --protocols $p --out "$R/route_ifn_$x" --model ifnet --align ea --recipe r1_pulx_vp2_e180_vr$x --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
  done
  blend --runs "$S/ni26-t2p-d378-fbc-tl15-1568960/runs" --protocols $p --out "$R/route_fbc_tl15f_tv8" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8 --seeds $(seq -s, 204 215)
  blend --runs "$S/ni26-t2p-d365-sh-us5-5090/runs" --protocols $p --out "$R/route_sh_us5_tv8" --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_us5tv8 --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
done
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d381"; D = f"{O}/t2_d296r"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
d378 = json.load(open(f"{O}/t2_d378/spec.json")); S3f = d378["arms"]["S3f"]; M.update({k: d378["members"][k] for k in ("stl_eaf", "lplh36", "lplh48", "fbc_tl15f")})
for k, f in (("sw2", "t2_d370/spec.json"), ("stl_ea", "t2_d360r/spec.json")):
    M[k] = json.load(open(f"{O}/{f}"))["members"][k]
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M["fbc_tv8"] = mem(192, D + "/route_fbc_r1_pulxx_vp2_e180_vr_rc75tv8/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8fb3")
M["fbc_pl"] = mem(192, D + "/route_fbc_r1_pulxx_vp2_e180_vr_rc75/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75fb3")
M["sh_tv8"] = mem(156, D + "/route_sh_r1_pulx_vp2_e180_vrtv8/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8fb2")
M["sh_pl"] = mem(156, D + "/route_sh_r1_pulx_vp2_e180_vr/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrfb2")
M["ifn_tv8"] = mem(168, R + "/route_ifn_tv8/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8fb2")
M["ifn_pl"] = mem(168, R + "/route_ifn_/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vrfb2")
M["fbc_tl15f_tv8"] = mem(204, R + "/route_fbc_tl15f_tv8/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8fb3")
M["sh_us5_tv8"] = mem(156, R + "/route_sh_us5_tv8/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_us5tv8fb2")
# the shipped members these twins stand in for must be the same nets: the same offsets as their tv8c twins
assert (M["fbc_tv8c"]["offset"], M["sh_tv8c"]["offset"], M["ifn"]["offset"], M["fbc_tl15f"]["offset"]) == (192, 156, 168, 204)
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
def add_st(a, new, slot):
    a = dict(a); assert new not in a and all(k in a for k in slot), (new, slot)
    tot = sum(a.pop(k) for k in slot)
    a.update({k: tot / (len(slot) + 1) for k in list(slot) + [new]})
    return a
def split(a, old, twin):
    a = dict(a); a[twin] = a[old] / 2; a[old] = a[old] / 2
    return a
A = {"nm35": nm, "V8": sw(nm, [("fbc_tv8c", "fbc_tv8"), ("sh_tv8c", "sh_tv8"), ("ifn", "ifn_tv8")]),
     "V1": sw(nm, [("fbc_tv8c", "fbc_pl"), ("sh_tv8c", "sh_pl"), ("ifn", "ifn_pl")])}
V8 = sw(S3f, [("fbc_tl15f", "fbc_tl15f_tv8"), ("sh_tv8c", "sh_tv8"), ("ifn", "ifn_tv8")])
slot = [k for k in S3f if k in ("lplh36", "lplh48", "stl_eaf", "vp12", "lp10")]; assert len(slot) == 5, sorted(S3f)
B = {"S3f": S3f, "S3f_V8": V8, "S3f_V8_EA6": add_st(V8, "stl_ea", slot), "S3f_V8_SW2": add_st(V8, "sw2", slot), "S3f_V8_USs": split(V8, "sh_tv8", "sh_us5_tv8")}
for a in list(A.values()) + list(B.values()):
    assert abs(sum(a.values()) - 1) < 1e-9, a
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": A}, open(f"{R}/spec_a.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": B}, open(f"{R}/spec_b.json", "w"), indent=1)
json.dump({f"{R}/spec_a.json": list(A), f"{R}/spec_b.json": list(B)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in S3f_V8 S3f_V8_EA6 S3f_V8_SW2 S3f_V8_USs; do
  echo "== user-cluster CI: $a vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_b.json" S3f $a > "$R/cluster_$a.txt" 2>&1; tail -6 "$R/cluster_$a.txt"
done
