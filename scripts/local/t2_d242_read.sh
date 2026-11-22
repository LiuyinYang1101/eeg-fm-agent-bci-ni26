#!/usr/bin/env bash
# T2-D242 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d242-stl-e60-<kit|seen1>-<a|b>-<jobid>
# (a = seeds 0-5, b = 6-11): the ST-L member fine-tuned 60 epochs on the base fixed by T2-D241's VERDICT line (none = the D112a recipe, dp10 / ll75 =
# D241's candidate arm), seeds 0-11, vs the base's ST-L. Links the 2 x 12 result files into outputs/t2_d242/stl_e60/, writes the spec from T2-D241's
# (members + stl_e60; anchor c_<base>, = c_rc75 for none), checks the files differ from the base's, then t2p_pkgshape.
# Kill iff ST-L alone (e60 - base) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_e60 vs c_<base> meets bar d141 AND new >= 0; adoption
# needs T2-D242r on fresh seeds 12-23.
#   mem-guard bash scripts/local/t2_d242_read.sh <none|dp10|ll75>      (after t2_d241_read.sh, whose links and spec this read reuses)
set -uo pipefail
cd "$(dirname "$0")/../.."
B=${1:?base: none|dp10|ll75 (T2-D241 VERDICT)}
declare -A K=([none]="ll1ema0dp0mx0.4" [dp10]="ll1ema0dp0.1mx0.4" [ll75]="ll0.75ema0dp0mx0.4")
[ -n "${K[$B]:-}" ] || { echo "base must be none|dp10|ll75"; exit 1; }
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
mkdir -p "$O/t2_d242/stl_e60"
for p in kit seen1; do
  for h in a b; do
    d=( "$S"/ni26-t2p-d242-stl-e60-$p-$h-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for e60 $p $h: ${d[*]}"; exit 2; }
    n=0
    for f in "${d[0]}"runs/${p}__fm-st_large-*fmE60b64*"${K[$B]}"cg*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d242/stl_e60/$(basename "$f")" && n=$((n + 1)); done
    [ "$n" -eq 6 ] || { echo "e60 $p $h: $n / 6 result files with fmE60 + ${K[$B]}"; exit 3; }
  done
done
[ "$(ls "$O/t2_d242/stl_e60" | wc -l)" -eq 24 ] || { echo "stl_e60: need 24 files"; exit 4; }
$PY -I - "$B" "$O/t2_d241/spec.json" "$O/t2_d242" <<'PYEOF' || exit 5
import hashlib, json, sys
import numpy as np
b, src, out = sys.argv[1:4]
m = json.load(open(src))["members"]
ref = "stl" if b == "none" else f"stl_{b}"
knob = {"none": "ll1ema0dp0", "dp10": "ll1ema0dp0.1", "ll75": "ll0.75ema0dp0"}[b]
cpt = {"kit": "cptstlarge-d89b", "seen1": "cptpt-d111-with"}
mem = {k: m[k] for k in ("fbc_rc75", "sh_vr", "stl", "mor", ref)}
mem["stl_e60"] = {"offset": 0, **{p: f"{out}/stl_e60/{p}__fm-st_large-mean__sz__n00__fmE60b64le0.0001lh0.001wd0.05{knob}mx0.4cg{cpt[p]}__s{{s}}.npz"
                                  for p in ("kit", "seen1")}}
anc = "c_rc75" if b == "none" else f"c_{b}"
line = lambda s: {"fbc_rc75": 0.4, "sh_vr": 0.3, s: 0.2, "mor": 0.1}
spec = {"n": 12, "anchor": anc, "bar": "d141", "members": mem,
        "arms": {anc: line(ref), "c_e60": line("stl_e60"), "ref_alone": {ref: 1.0}, "stl_e60_alone": {"stl_e60": 1.0}}}
txt = json.dumps(spec, indent=1)
open(f"{out}/spec.json", "w").write(txt)
print(f"base {b}: anchor {anc} (ST-L {ref}); spec sha256 {hashlib.sha256(txt.encode()).hexdigest()[:12]}")
for p in ("kit", "seen1"):
    for s in range(12):
        x, y = np.load(mem[ref][p].format(s=s)), np.load(mem["stl_e60"][p].format(s=s))
        assert np.array_equal(x["test_idx"], y["test_idx"]), (p, s)
        d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
        if s == 0:
            print(f"e60 {p} seed 0: vs the base ST-L max |d prob| {d:.3f}")
        assert d > 1e-4, f"{p} seed {s}: the runs equal the base"
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d242/spec.json" --out "$O/t2_d242/read.json" > "$O/t2_d242/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d242/read.json" "$B" <<'PYEOF' | tee "$O/t2_d242/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); b = sys.argv[2]; rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "stl_e60_alone")]["test"] - rows[("kit", i, "ref_alone")]["test"] for i in range(6)) / 6
c = r["arms"]["c_e60"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
anc = "c_rc75" if b == "none" else f"c_{b}"
print(f"e60 on base {b}: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_e60 vs {anc}: mix "
      f"{c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f} -> "
      + ("KILLED" if k <= -0.01 else ("CANDIDATE" if ok else "no")))
print("VERDICT:", "candidate e60 -> launch T2-D242r (fresh seeds 12-23) before any adoption" if ok else "NOT ADOPTED (30 epochs stay)")
PYEOF
