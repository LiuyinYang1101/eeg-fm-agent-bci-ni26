#!/usr/bin/env bash
# T2-D199 (zero compute), as pre-registered in docs/DECISIONS_T2.md: the per-user blend ratios of the vp2 CNNs.
# Step 1 (pick, in-sample): on T2-D195's vp2 runs (FBCNet seeds 96-107, ShallowNet 60-71) blend FBCNet at r in {1/4, 1/3, 1/2} and ShallowNet at
#   r in {1/3, 1/2, 2/3}; t2p_pkgshape over the 9 combinations vs the anchor (1/3, 1/2) = the package; pick = the largest mix delta (the anchor if none > 0).
# Step 2 (confirm, fresh seeds): on T2-D197's vp2 runs (FBCNet 108-119, ShallowNet 72-83) the pick vs the anchor; adopt iff bar d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_d199_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
D=$O/t2_d199
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
for st in pick conf; do
  if [ $st = pick ]; then F=( "$S"/ni26-t2p-d195-fbc-r1_pulxx_vp2-{kit,seen1}-[0-9]*/ ); H=( "$S"/ni26-t2p-d195-sh-r1_pulx_vp2-[0-9]*/ ); SF=$(seq -s, 96 107); SS=$(seq -s, 60 71)
  else F=( "$S"/ni26-t2p-d197-fbc-r1_pulxx_vp2-{kit,seen1}-[0-9]*/ ); H=( "$S"/ni26-t2p-d197-sh-r1_pulx_vp2-[0-9]*/ ); SF=$(seq -s, 108 119); SS=$(seq -s, 72 83); fi
  [ ${#F[@]} -eq 2 ] && [ ${#H[@]} -eq 1 ] || { echo "need 2 FBCNet + 1 ShallowNet dirs for $st"; exit 2; }
  for i in 0 1; do p=$([ $i = 0 ] && echo kit || echo seen1)
    for rt in "0.25 r4" "0.3333 r3" "0.5 r2"; do set -- $rt
      blend --runs "${F[$i]}runs" --protocols $p --out "$D/$st/fbc" --model fbcnet_bboth --recipe r1_pulxx_vp2 --r $1 --tag $2 --seeds $SF; done
  done
  for p in kit seen1; do
    for rt in "0.3333 r3" "0.5 r2" "0.6667 r67"; do set -- $rt
      blend --runs "${H[0]}runs" --protocols $p --out "$D/$st/sh" --model shallow --align ea --recipe r1_pulx_vp2 --r $1 --tag $2 --seeds $SS; done
  done
done
$PY -I - "$O" <<'PYEOF'
import json, sys
O = sys.argv[1]; D = f"{O}/t2_d199"
base = {k: v for k, v in json.load(open(f"{O}/t2_d193/spec.json"))["members"].items() if k in ("stl", "mor")}
def spec(st, of, os_, arms_extra=None):
    m = dict(base)
    for t in ("r4", "r3", "r2"):
        m[f"fbc_{t}"] = {"offset": of, **{p: f"{D}/{st}/fbc/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2fb{t}__s{{s}}.npz" for p in ("kit", "seen1")}}
    for t in ("r3", "r2", "r67"):
        m[f"sh_{t}"] = {"offset": os_, **{p: f"{D}/{st}/sh/{p}__shallow__ea__n00__r1_pulx_vp2fb{t}__s{{s}}.npz" for p in ("kit", "seen1")}}
    arms = {f"f{a}_s{b}": {f"fbc_{a}": 0.5, f"sh_{b}": 0.2, "stl": 0.2, "mor": 0.1} for a in ("r4", "r3", "r2") for b in ("r3", "r2", "r67")}
    json.dump({"n": 12, "anchor": "fr3_sr2", "bar": "d141", "members": m, "arms": arms}, open(f"{D}/spec_{st}.json", "w"), indent=2)
spec("pick", 96, 60); spec("conf", 108, 72)
PYEOF
for st in pick conf; do $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$D/spec_$st.json" --out "$D/read_$st.json" > "$D/read_$st.txt" 2>&1; echo "$st rc $?"; done
$PY -I - "$D" <<'PYEOF' | tee "$D/verdict.txt"
import json, sys
D = sys.argv[1]
rp = json.load(open(f"{D}/read_pick.json"))["arms"]; rc = json.load(open(f"{D}/read_conf.json"))["arms"]
f = lambda a: f"mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}"
for k in sorted(rp):
    if k != "fr3_sr2":
        print(f"pick  {k}: {f(rp[k])}")
cand = [k for k in rp if k != "fr3_sr2" and rp[k]["mix"]["d"] > 0]
pick = max(cand, key=lambda k: rp[k]["mix"]["d"]) if cand else "fr3_sr2"
print(f"PICK (step 1, in-sample): {pick}")
if pick == "fr3_sr2":
    print("VERDICT: NOT ADOPTED (the anchor r 1/3 / 1/2 is the pick)")
else:
    a = rc[pick]; ok = a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
    print(f"confirm {pick} vs anchor (fresh seeds 108-119 / 72-83): {f(a)} -> {'MET' if ok else 'not met'} (bar d141 + new >= 0)")
    print("VERDICT:", f"ADOPT {pick}" if ok else "NOT ADOPTED")
PYEOF
