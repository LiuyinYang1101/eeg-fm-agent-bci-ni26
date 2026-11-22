#!/usr/bin/env bash
# T2-D247 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d247-stl-rcll-<kit|seen1>-<jobid>:
# the ST-L member with T2-D235's re-colouring 0.75 AND layer-wise lr decay X (X = 0.85 / 0.75 / 0.65, fixed by the row's rule from T2-D241 / D243),
# seeds 0-11, vs the stl member. Links the 24 result files into outputs/t2_d247/stl_rcll/, writes the spec from T2-D241's (members + stl_rcll;
# anchor c_rc75), checks the files differ from the stl member, then t2p_pkgshape.
# Kill iff ST-L alone (arm - stl) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_rcll vs c_rc75 meets bar d141 AND new >= 0; adoption needs
# T2-D247r on fresh seeds 12-23.
#   mem-guard bash scripts/local/t2_d247_read.sh <0.85|0.75|0.65>
set -uo pipefail
cd "$(dirname "$0")/../.."
X=${1:?the LLRD dose: 0.85|0.75|0.65}
case "$X" in 0.85|0.75|0.65) ;; *) echo "dose must be 0.85|0.75|0.65"; exit 1 ;; esac
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
K="fmE30b64le0.0001lh0.001wd0.05ll${X}ema0dp0mx0.4cgrc0.75cpt"
mkdir -p "$O/t2_d247/stl_rcll"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d247-stl-rcll-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for rcll $p: ${d[*]}"; exit 2; }
  n=0
  for f in "${d[0]}"runs/${p}__fm-st_large-*"$K"*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d247/stl_rcll/$(basename "$f")" && n=$((n + 1)); done
  [ "$n" -eq 12 ] || { echo "rcll $p: $n / 12 result files with $K"; exit 3; }
done
$PY -I - "$X" "$O/t2_d241/spec.json" "$O/t2_d247" <<'PYEOF' || exit 5
import hashlib, json, sys
import numpy as np
x, src, out = sys.argv[1:4]
m = json.load(open(src))["members"]
cpt = {"kit": "cptstlarge-d89b", "seen1": "cptpt-d111-with"}
mem = {k: m[k] for k in ("fbc_rc75", "sh_vr", "stl", "mor")}
mem["stl_rcll"] = {"offset": 0, **{p: f"{out}/stl_rcll/{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll{x}ema0dp0mx0.4cgrc0.75{cpt[p]}__s{{s}}.npz"
                                   for p in ("kit", "seen1")}}
line = lambda s: {"fbc_rc75": 0.4, "sh_vr": 0.3, s: 0.2, "mor": 0.1}
spec = {"n": 12, "anchor": "c_rc75", "bar": "d141", "members": mem,
        "arms": {"c_rc75": line("stl"), "c_rcll": line("stl_rcll"), "stl_alone": {"stl": 1.0}, "stl_rcll_alone": {"stl_rcll": 1.0}}}
txt = json.dumps(spec, indent=1)
open(f"{out}/spec.json", "w").write(txt)
print(f"LLRD {x} + rc 0.75; spec sha256 {hashlib.sha256(txt.encode()).hexdigest()[:12]}")
for p in ("kit", "seen1"):
    for s in range(12):
        a, b = np.load(mem["stl"][p].format(s=s)), np.load(mem["stl_rcll"][p].format(s=s))
        assert np.array_equal(a["test_idx"], b["test_idx"]), (p, s)
        d = float(np.abs(a["prob_last"] - b["prob_last"]).max())
        if s == 0:
            print(f"rcll {p} seed 0: vs the ST-L member max |d prob| {d:.3f}")
        assert d > 1e-4, f"{p} seed {s}: the runs equal the stl member"
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d247/spec.json" --out "$O/t2_d247/read.json" > "$O/t2_d247/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d247/read.json" <<'PYEOF' | tee "$O/t2_d247/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "stl_rcll_alone")]["test"] - rows[("kit", i, "stl_alone")]["test"] for i in range(6)) / 6
c = r["arms"]["c_rcll"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
print(f"rcll: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_rcll vs c_rc75: mix {c['mix']['d']:+.4f} "
      f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f} -> "
      + ("KILLED" if k <= -0.01 else ("CANDIDATE" if ok else "no")))
print("VERDICT:", "candidate rcll -> launch T2-D247r (fresh seeds 12-23) before any adoption" if ok else "NOT ADOPTED (the ST-L recipe stays)")
PYEOF
