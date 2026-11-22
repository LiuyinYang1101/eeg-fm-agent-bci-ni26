#!/usr/bin/env bash
# One-arm MOR-member read (local CPU, zero compute) for T2-D258 / T2-D259, as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
# ni26-t2p-<row>-mor-<arm>-<kit|seen1>-<jobid>: the arm's MOR fine-tune on seeds 0-11 vs T2-D256's ctl MOR (today's code; bitwise = the 10-15 files).
# Links the 24 result files with <tag> into outputs/t2_<row>/mor_<arm>/, writes outputs/t2_<row>/spec.json from T2-D256's spec (members fbc_rc75 (204-215),
# sh_vr (168-179), stl_rcll, mor_ctl; offsets asserted; sha256 printed) + mor_<arm>, then t2p_pkgshape with anchor c_ctl = {fbc_rc75 0.4, sh_vr 0.3,
# stl_rcll 0.2, mor_ctl 0.1}. Kill iff MOR alone (arm - ctl) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_<arm> vs c_ctl meets bar d141
# (mix >= +0.005 with >= 9 / 12 up AND kit test >= -0.005) AND new >= 0; adoption needs the row's fresh-seed replicate.
#   mem-guard bash scripts/local/t2_mor_arm_read.sh d258 e24 fmE24b64le0.001lh0.001wd0.05ll1ema0dp0li16roled0.1
#   mem-guard bash scripts/local/t2_mor_arm_read.sh d259 rc5 fmE12b64le0.001lh0.001wd0.05ll1ema0dp0rc0.5li16roled0.1
set -uo pipefail
cd "$(dirname "$0")/../.."
row=${1:?row}; arm=${2:?arm}; t=${3:?tag}
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
R=$O/t2_$row
mkdir -p "$R/mor_$arm"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-$row-mor-$arm-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for mor $arm $p: ${d[*]}"; exit 2; }
  n=0
  for f in "${d[0]}"runs/${p}__fm-mor_p1v2-mean__wz__n00__${t}__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$R/mor_$arm/$(basename "$f")" && n=$((n + 1)); done
  [ "$n" -eq 12 ] || { echo "mor $arm $p: $n / 12 result files with $t"; exit 3; }
done
[ "$(ls "$R/mor_$arm" | wc -l)" -eq 24 ] || { echo "need 24 files"; exit 4; }
$PY -I - "$O/t2_d256/spec.json" "$R" "$arm" <<'PYEOF' || exit 5
import glob, hashlib, json, os, sys
import numpy as np
s, out, arm = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor_ctl": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs, {m: d["offset"] for m, d in mem.items()}
pat = {}
for p in ("kit", "seen1"):
    f = sorted(glob.glob(f"{out}/mor_{arm}/{p}__*__s0.npz"))
    assert len(f) == 1, f
    pat[p] = f[0][:-len("0.npz")] + "{s}.npz"
    x, y = np.load(mem["mor_ctl"][p].format(s=0)), np.load(pat[p].format(s=0))
    assert np.array_equal(x["test_idx"], y["test_idx"]), p
    d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
    print(f"mor_{arm} {p} seed 0: vs mor_ctl max |d prob| {d:.3f}")
    assert d > 1e-4, "the arm equals ctl"
mem[f"mor_{arm}"] = dict(offset=0, **pat)
base = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2}
spec = dict(n=12, anchor="c_ctl", bar="d141", members=mem,
            arms={"c_ctl": dict(base, mor_ctl=0.1), f"c_{arm}": dict(base, **{f"mor_{arm}": 0.1}),
                  "mor_ctl_alone": {"mor_ctl": 1.0}, f"mor_{arm}_alone": {f"mor_{arm}": 1.0}})
sp = os.path.join(out, "spec.json")
open(sp, "w").write(json.dumps(spec, indent=1) + "\n")
print(f"spec {sp} sha256 {hashlib.sha256(open(sp, 'rb').read()).hexdigest()[:12]}")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$R/spec.json" --out "$R/read.json" > "$R/read.txt" 2>&1; echo "rc $?"
$PY -I - "$R/read.json" "$arm" "$row" <<'PYEOF' | tee "$R/verdict.txt"
import json, sys
r, arm, row = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, f"mor_{arm}_alone")]["test"] - rows[("kit", i, "mor_ctl_alone")]["test"] for i in range(6)) / 6
c = r["arms"][f"c_{arm}"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
am = r["arms"][f"mor_{arm}_alone"]["mix"]["d"] - r["arms"]["mor_ctl_alone"]["mix"]["d"]
print(f"{arm}: kill check (MOR alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_{arm} vs c_ctl: mix {c['mix']['d']:+.4f} "
      f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, "
      f"seen1 val {c['seen1_val']['d']:+.4f}; MOR alone ({arm} - ctl) mix {am:+.4f}")
print("VERDICT:", "KILLED" if k <= -0.01 else (f"CANDIDATE -> T2-{row.upper()}r (fresh seeds 12-23) before any adoption" if ok else "NOT ADOPTED (the MOR fine-tune stays)"))
PYEOF
