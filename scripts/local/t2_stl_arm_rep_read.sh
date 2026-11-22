#!/usr/bin/env bash
# One-arm ST-L fresh-seed replicate read (local CPU, zero compute) for T2-D272r on, as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch
# ni26` of ni26-t2p-<row>-stl-<arm>-<kit|seen1>-<jobid> (<row> = the replicate's, e.g. d272r): the arm's ST-L fine-tune on FRESH seeds 12-23 vs the
# adopted ST-L on the same seeds (T2-D247r's stl_rcll files). Links the 24 result files ${p}__fm-st_large-mean__sz__n00__<pre>cpt*<post>__s*.npz
# into outputs/t2_<row>/stl_<arm>/, checks they differ from stl_rcll, writes outputs/t2_<row>/spec.json from T2-D247r's spec (members fbc_rc75
# (T2-D231r, seeds 192-203), sh_vr (T2-D215, 156-167), stl_rcll (12-23), mor (0-11); offsets asserted; sha256 printed) + stl_<arm> (12-23), then
# t2p_pkgshape with anchor c_rcll = {fbc_rc75 0.4, sh_vr 0.3, stl_rcll 0.2, mor 0.1}. REPLICATED iff c_<arm> vs c_rcll gives mix d > 0 AND new d >= 0.
#   mem-guard bash scripts/local/t2_stl_arm_rep_read.sh d272r lplh fmE30b64le0.0001lh0.003wd0.05ll0.75ema0dp0mx0.4cgrc0.75lp10
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
$PY -I - "$O/t2_d247r/spec.json" "$R" "$arm" <<'PYEOF' || exit 5
import glob, hashlib, json, os, sys
import numpy as np
s, out, arm = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
offs = {"fbc_rc75": 192, "sh_vr": 156, "stl_rcll": 12, "mor": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs, {m: d["offset"] for m, d in mem.items()}
pat = {}
for p in ("kit", "seen1"):
    f = sorted(glob.glob(f"{out}/stl_{arm}/{p}__*__s12.npz"))
    assert len(f) == 1, f
    pat[p] = f[0][:-len("12.npz")] + "{s}.npz"
    assert all(os.path.exists(pat[p].format(s=12 + i)) for i in range(12)), "seeds 12-23"
    x, y = np.load(mem["stl_rcll"][p].format(s=12)), np.load(pat[p].format(s=12))
    assert np.array_equal(x["test_idx"], y["test_idx"]), p
    d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
    print(f"stl_{arm} {p} seed 12: vs the adopted ST-L (stl_rcll) max |d prob| {d:.3f}")
    assert d > 1e-4, "the runs equal the reference"
mem[f"stl_{arm}"] = dict(offset=12, **pat)
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
r, a = json.load(open(sys.argv[1])), sys.argv[2]
c = r["arms"][f"c_{a}"]
ok = c["mix"]["d"] > 0 and c["new"]["d"] >= 0
am = r["arms"][f"stl_{a}_alone"]["mix"]["d"] - r["arms"]["stl_rcll_alone"]["mix"]["d"]
print(f"{a} (fresh seeds 12-23): c_{a} vs c_rcll: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen "
      f"{c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, seen1 val {c['seen1_val']['d']:+.4f}; ST-L alone ({a} - rcll) mix {am:+.4f}")
print("VERDICT:", "REPLICATED (mix > 0 AND new >= 0 on fresh seeds)" if ok else "NOT REPLICATED")
PYEOF
