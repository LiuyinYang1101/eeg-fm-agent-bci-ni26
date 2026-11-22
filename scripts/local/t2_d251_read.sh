#!/usr/bin/env bash
# T2-D251 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d251-stl-le2-<kit|seen1>-<jobid>:
# the adopted ST-L (T2-D247r: rc 0.75 + LLRD 0.75) with the encoder lr doubled (--fm-lr-enc 2e-4, tag le0.0002), seeds 0-11, vs the stl_rcll runs of
# T2-D247 (same seeds). Links the 24 result files into outputs/t2_d251/stl_le2/, checks each run logged the knob (the result tag) and that the files
# differ from stl_rcll, writes outputs/t2_d251/spec.json from T2-D247's spec (members fbc_rc75 (T2-D231, seeds 204-215), sh_vr (T2-D215r, 168-179),
# stl_rcll, mor; offsets asserted; sha256 printed) + stl_le2, then t2p_pkgshape. Kill iff ST-L alone (le2 - rcll) kit test on seeds 0-5 <= -0.01;
# adopt-CANDIDATE iff c_le2 vs c_rcll meets bar d141 (mix >= +0.005 with >= 9 / 12 up AND kit test >= -0.005) AND new >= 0; adoption needs T2-D251r.
#   mem-guard bash scripts/local/t2_d251_read.sh      (after t2_d215r_read.sh and t2_d231_read.sh, which write the CNN route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
K="le0.0002lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75"
mkdir -p "$O/t2_d251/stl_le2"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d251-stl-le2-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for stl le2 $p: ${d[*]}"; exit 2; }
  n=0
  for f in "${d[0]}"runs/${p}__fm-st_large-*"$K"cpt*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d251/stl_le2/$(basename "$f")" && n=$((n + 1)); done
  [ "$n" -eq 12 ] || { echo "stl le2 $p: $n / 12 result files with $K"; exit 3; }
done
[ "$(ls "$O/t2_d251/stl_le2" | wc -l)" -eq 24 ] || { echo "stl le2: need 24 files"; exit 4; }
$PY -I - "$O/t2_d247/spec.json" "$O/t2_d251" <<'PYEOF' || exit 5
import glob, hashlib, json, os, sys
import numpy as np
s, out = json.load(open(sys.argv[1])), sys.argv[2]
offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs, {m: d["offset"] for m, d in mem.items()}
le2 = {}
for p in ("kit", "seen1"):
    f = sorted(glob.glob(f"{out}/stl_le2/{p}__*__s0.npz"))
    assert len(f) == 1, f
    le2[p] = f[0][:-len("0.npz")] + "{s}.npz"
    x, y = np.load(mem["stl_rcll"][p].format(s=0)), np.load(le2[p].format(s=0))
    assert np.array_equal(x["test_idx"], y["test_idx"]), p
    d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
    print(f"le2 {p} seed 0: vs the adopted ST-L (stl_rcll) max |d prob| {d:.3f}")
    assert d > 1e-4, "the runs equal the reference"
mem["stl_le2"] = dict(offset=0, **le2)
spec = dict(n=12, anchor="c_rcll", bar="d141", members=mem,
            arms={"c_rcll": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}, "c_le2": {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_le2": 0.2, "mor": 0.1},
                  "stl_rcll_alone": {"stl_rcll": 1.0}, "stl_le2_alone": {"stl_le2": 1.0}})
sp = os.path.join(out, "spec.json")
open(sp, "w").write(json.dumps(spec, indent=1) + "\n")
print(f"spec {sp} sha256 {hashlib.sha256(open(sp, 'rb').read()).hexdigest()[:12]}")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d251/spec.json" --out "$O/t2_d251/read.json" > "$O/t2_d251/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d251/read.json" <<'PYEOF' | tee "$O/t2_d251/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "stl_le2_alone")]["test"] - rows[("kit", i, "stl_rcll_alone")]["test"] for i in range(6)) / 6
c = r["arms"]["c_le2"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
a = r["arms"]["stl_le2_alone"]
print(f"le2: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_le2 vs c_rcll: mix {c['mix']['d']:+.4f} "
      f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, "
      f"seen1 val {c['seen1_val']['d']:+.4f}; ST-L alone (le2 - rcll) mix {a['mix']['d'] - r['arms']['stl_rcll_alone']['mix']['d']:+.4f}")
print("VERDICT:", "KILLED" if k <= -0.01 else ("CANDIDATE -> T2-D251r (fresh seeds 12-23) before any adoption" if ok else "NOT ADOPTED (the ST-L recipe stays)"))
PYEOF
