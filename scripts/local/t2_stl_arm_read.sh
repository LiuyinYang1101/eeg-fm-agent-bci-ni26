#!/usr/bin/env bash
# One-arm ST-L-member read (local CPU, zero compute) for T2-D257 on, as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
# ni26-t2p-<row>-stl-<arm>-<kit|seen1>-<jobid>: the arm's ST-L fine-tune on seeds 0-11 vs the adopted ST-L (T2-D247r's stl_rcll files, same seeds).
# Links the 24 result files ${p}__fm-st_large-mean__sz__n00__<pre>cpt*<post>__s*.npz into outputs/t2_<row>/stl_<arm>/, checks they differ from
# stl_rcll, writes outputs/t2_<row>/spec.json from T2-D253's spec (members fbc_rc75 (T2-D231, seeds 204-215), sh_vr (T2-D215r, 168-179), stl_rcll,
# mor; offsets asserted; sha256 printed) + stl_<arm>, then t2p_pkgshape with anchor c_rcll = {fbc_rc75 0.4, sh_vr 0.3, stl_rcll 0.2, mor 0.1}.
# Kill iff ST-L alone (arm - rcll) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_<arm> vs c_rcll meets bar d141 (mix >= +0.005 with
# >= 9 / 12 up AND kit test >= -0.005) AND new >= 0; adoption needs the row's fresh-seed replicate (seeds 12-23).
#   mem-guard bash scripts/local/t2_stl_arm_read.sh d257 aug fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75aug0.2-30
set -uo pipefail
cd "$(dirname "$0")/../.."
row=${1:?row}; arm=${2:?arm}; pre=${3:?tag before cpt}; post=${4:-}
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
R=$O/t2_$row
mkdir -p "$R/stl_$arm"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-$row-stl-$arm-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for stl $arm $p: ${d[*]}"; exit 2; }
  n=0
  for f in "${d[0]}"runs/${p}__fm-st_large-mean__sz__n00__${pre}cpt*${post}__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$R/stl_$arm/$(basename "$f")" && n=$((n + 1)); done
  [ "$n" -eq 12 ] || { echo "stl $arm $p: $n / 12 result files with ${pre}cpt*${post}"; exit 3; }
done
[ "$(ls "$R/stl_$arm" | wc -l)" -eq 24 ] || { echo "stl $arm: need 24 files"; exit 4; }
$PY -I - "$O/t2_d253/spec.json" "$R" "$arm" <<'PYEOF' || exit 5
import glob, hashlib, json, os, sys
import numpy as np
s, out, arm = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs, {m: d["offset"] for m, d in mem.items()}
pat = {}
for p in ("kit", "seen1"):
    f = sorted(glob.glob(f"{out}/stl_{arm}/{p}__*__s0.npz"))
    assert len(f) == 1, f
    pat[p] = f[0][:-len("0.npz")] + "{s}.npz"
    x, y = np.load(mem["stl_rcll"][p].format(s=0)), np.load(pat[p].format(s=0))
    assert np.array_equal(x["test_idx"], y["test_idx"]), p
    d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
    print(f"stl_{arm} {p} seed 0: vs the adopted ST-L (stl_rcll) max |d prob| {d:.3f}")
    assert d > 1e-4, "the runs equal the reference"
mem[f"stl_{arm}"] = dict(offset=0, **pat)
base = {"fbc_rc75": 0.4, "sh_vr": 0.3, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=mem,
            arms={"c_rcll": dict(base, stl_rcll=0.2), f"c_{arm}": dict(base, **{f"stl_{arm}": 0.2}),
                  "stl_rcll_alone": {"stl_rcll": 1.0}, f"stl_{arm}_alone": {f"stl_{arm}": 1.0}})
sp = os.path.join(out, "spec.json")
open(sp, "w").write(json.dumps(spec, indent=1) + "\n")
print(f"spec {sp} sha256 {hashlib.sha256(open(sp, 'rb').read()).hexdigest()[:12]}")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$R/spec.json" --out "$R/read.json" > "$R/read.txt" 2>&1; echo "rc $?"
$PY -I - "$R/read.json" "$arm" <<'PYEOF' | tee "$R/verdict.txt"
import json, sys
r, a = json.load(open(sys.argv[1])), sys.argv[2]; rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, f"stl_{a}_alone")]["test"] - rows[("kit", i, "stl_rcll_alone")]["test"] for i in range(6)) / 6
c = r["arms"][f"c_{a}"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
am = r["arms"][f"stl_{a}_alone"]["mix"]["d"] - r["arms"]["stl_rcll_alone"]["mix"]["d"]
print(f"{a}: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_{a} vs c_rcll: mix {c['mix']['d']:+.4f} "
      f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, "
      f"seen1 val {c['seen1_val']['d']:+.4f}; ST-L alone ({a} - rcll) mix {am:+.4f}")
print("VERDICT:", "KILLED" if k <= -0.01 else ("CANDIDATE -> the fresh-seed replicate (seeds 12-23) before any adoption" if ok else "NOT ADOPTED (the adopted ST-L stays)"))
PYEOF
