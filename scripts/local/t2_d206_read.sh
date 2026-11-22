#!/usr/bin/env bash
# T2-D206 (zero compute), as pre-registered in docs/DECISIONS_T2.md: the family weights of the vp2-180 package.
# Arms = (FBCNet, ShallowNet, ST-L, MOR) on a 0.1 grid with FBCNet in {0.4..0.7}, ShallowNet in {0.1..0.3}, ST-L in {0.1..0.3}, MOR in {0.0..0.2}, summing to 1;
# anchor (0.5, 0.2, 0.2, 0.1) = the shipped package. Step 1 (pick, in-sample) on T2-D203's vp2_e180 route files (FBCNet seeds 156-167, ShallowNet 120-131):
# the arm with the largest mix delta vs the anchor (the anchor if none > 0). Step 2 (confirm) on T2-D204's vp2_e180 route files (FBCNet 168-179,
# ShallowNet 132-143; the same stl / mor files): adopt iff the pick vs the anchor meets bar d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_d206_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
D=$O/t2_d206
$PY -I - "$O" <<'PYEOF'
import json, itertools, sys
O = sys.argv[1]; D = f"{O}/t2_d206"
base = {k: v for k, v in json.load(open(f"{O}/t2_d193/spec.json"))["members"].items() if k in ("stl", "mor")}
grid = [(f, s, t, round(1 - f - s - t, 1)) for f, s, t in itertools.product((0.4, 0.5, 0.6, 0.7), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
        if -1e-9 <= 1 - f - s - t <= 0.2 + 1e-9]
for st, src, of, os_ in (("pick", "t2_d203", 156, 120), ("conf", "t2_d204", 168, 132)):
    m = dict(base)
    m["fbc"] = {"offset": of, **{p: f"{O}/{src}/route_fbc_r1_pulxx_vp2_e180/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180fb3__s{{s}}.npz" for p in ("kit", "seen1")}}
    m["sh"] = {"offset": os_, **{p: f"{O}/{src}/route_sh_r1_pulx_vp2_e180/{p}__shallow__ea__n00__r1_pulx_vp2_e180fb2__s{{s}}.npz" for p in ("kit", "seen1")}}
    arms = {}
    for f, s, t, mo in grid:
        w = {"fbc": f, "sh": s, "stl": t}
        if mo > 1e-9:
            w["mor"] = mo
        arms[f"w{int(round(f*10))}{int(round(s*10))}{int(round(t*10))}{int(round(mo*10))}"] = w
    assert "w5221" in arms
    json.dump({"n": 12, "anchor": "w5221", "bar": "d141", "members": m, "arms": arms}, open(f"{D}/spec_{st}.json", "w"), indent=2)
print(len(grid), "arms")
PYEOF
for st in pick conf; do $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$D/spec_$st.json" --out "$D/read_$st.json" > "$D/read_$st.txt" 2>&1; echo "$st rc $?"; done
$PY -I - "$D" <<'PYEOF' | tee "$D/verdict.txt"
import json, sys
D = sys.argv[1]
rp = json.load(open(f"{D}/read_pick.json"))["arms"]; rc = json.load(open(f"{D}/read_conf.json"))["arms"]
f = lambda a: f"mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}"
top = sorted((k for k in rp if k != "w5221"), key=lambda k: -rp[k]["mix"]["d"])[:5]
for k in top:
    print(f"pick  {k}: {f(rp[k])}")
pick = top[0] if top and rp[top[0]]["mix"]["d"] > 0 else "w5221"
print(f"PICK (step 1, in-sample; w = FBCNet / ShallowNet / ST-L / MOR x 10): {pick}")
if pick == "w5221":
    print("VERDICT: NOT ADOPTED (the anchor is the pick)")
else:
    a = rc[pick]; ok = a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
    print(f"confirm {pick} vs w5221 (T2-D204's seeds): {f(a)} -> {'MET' if ok else 'not met'} (bar d141 + new >= 0)")
    print("VERDICT:", f"ADOPT {pick}" if ok else "NOT ADOPTED")
PYEOF
