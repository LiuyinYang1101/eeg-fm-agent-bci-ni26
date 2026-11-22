#!/usr/bin/env bash
# T2-D247r read (local CPU, zero compute), as pre-registered in T2-D247's row (docs/DECISIONS_T2.md), after `s1 fetch ni26` of
# ni26-t2p-d247r-stl-<ctrl|rcll>-<kit|seen1>-<jobid>: the ST-L control and the stacked ST-L (re-colouring 0.75 + LLRD 0.75) on FRESH seeds 12-23,
# in T2-D241r's line {fbc_rc75 of T2-D231r (192-203), sh_vr of T2-D215 (156-167), stl 0.2, mor}. Links the 48 result files into
# outputs/t2_d247r/, writes the spec (anchor c_ctrl), checks the arms differ, then t2p_pkgshape.
# Adopted iff c_rcll vs c_ctrl gives mix d > 0 AND new d >= 0.
#   mem-guard bash scripts/local/t2_d247r_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
declare -A K=([ctrl]="fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0mx0.4cgcpt" [rcll]="fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt")
for a in ctrl rcll; do
  mkdir -p "$O/t2_d247r/stl_$a"
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d247r-stl-$a-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for $a $p: ${d[*]}"; exit 2; }
    n=0
    for f in "${d[0]}"runs/${p}__fm-st_large-*"${K[$a]}"*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d247r/stl_$a/$(basename "$f")" && n=$((n + 1)); done
    [ "$n" -eq 12 ] || { echo "$a $p: $n / 12 result files with ${K[$a]}"; exit 3; }
  done
done
$PY -I - "$O/t2_d244/spec_r.json" "$O/t2_d247r" <<'PYEOF' || exit 5
import hashlib, json, sys
import numpy as np
src, out = sys.argv[1:3]
m = json.load(open(src))["members"]
cpt = {"kit": "cptstlarge-d89b", "seen1": "cptpt-d111-with"}
tag = {"ctrl": "ll1ema0dp0mx0.4cg", "rcll": "ll0.75ema0dp0mx0.4cgrc0.75"}
mem = {k: m[k] for k in ("fbc_rc75", "sh_vr", "mor")}
assert (mem["fbc_rc75"]["offset"], mem["sh_vr"]["offset"], mem["mor"]["offset"]) == (192, 156, 0)
for a in ("ctrl", "rcll"):
    mem[f"stl_{a}"] = {"offset": 12, **{p: f"{out}/stl_{a}/{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05{tag[a]}{cpt[p]}__s{{s}}.npz"
                                        for p in ("kit", "seen1")}}
line = lambda s: {"fbc_rc75": 0.4, "sh_vr": 0.3, s: 0.2, "mor": 0.1}
spec = {"n": 12, "anchor": "c_ctrl", "bar": "d141", "members": mem,
        "arms": {"c_ctrl": line("stl_ctrl"), "c_rcll": line("stl_rcll"), "stl_ctrl_alone": {"stl_ctrl": 1.0}, "stl_rcll_alone": {"stl_rcll": 1.0}}}
txt = json.dumps(spec, indent=1)
open(f"{out}/spec.json", "w").write(txt)
print(f"T2-D247r spec sha256 {hashlib.sha256(txt.encode()).hexdigest()[:12]}")
for p in ("kit", "seen1"):
    for i in range(12):
        a, b = np.load(mem["stl_ctrl"][p].format(s=12 + i)), np.load(mem["stl_rcll"][p].format(s=12 + i))
        assert np.array_equal(a["test_idx"], b["test_idx"]), (p, i)
        d = float(np.abs(a["prob_last"] - b["prob_last"]).max())
        if i == 0:
            print(f"rcll vs ctrl {p} seed 12: max |d prob| {d:.3f}")
        assert d > 1e-4, f"{p} seed {12 + i}: the arms are equal"
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d247r/spec.json" --out "$O/t2_d247r/read.json" > "$O/t2_d247r/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d247r/read.json" <<'PYEOF' | tee "$O/t2_d247r/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1]))
c = r["arms"]["c_rcll"]; al = r["arms"]["stl_rcll_alone"]["mix"]["d"] - r["arms"]["stl_ctrl_alone"]["mix"]["d"]
ok = c["mix"]["d"] > 0 and c["new"]["d"] >= 0
print(f"c_rcll vs c_ctrl (fresh seeds 12-23): mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, "
      f"new {c['new']['d']:+.4f}; ST-L alone (rcll - ctrl) mix {al:+.4f}")
print("VERDICT:", "ADOPTED (mix > 0 AND new >= 0 on fresh seeds)" if ok else "NOT ADOPTED (the ST-L recipe stays)")
PYEOF
