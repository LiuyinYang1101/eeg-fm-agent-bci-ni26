#!/usr/bin/env bash
# T2-D241 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d241-stl-<dp10|ll75>-<kit|seen1>-<jobid>:
# ST-L fine-tuned with drop path 0.1 (dp10) / layer-wise lr decay 0.75 (ll75), seeds 0-11, vs the ST-L member on disk. Links the 2 x 24 result files
# into outputs/t2_d241/stl_<arm>/, checks each run logged its knob (the result tag) and that the files differ from the stl member, then t2p_pkgshape
# on the adopted line {fbc_rc75 (T2-D231, seeds 204-215), sh_vr (T2-D215r, 168-179), stl, mor} with the ST-L member swapped.
# Per arm: kill iff ST-L alone (arm - ctrl) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_<arm> vs c_rc75 meets bar d141 AND new >= 0
# (both -> the higher mix); adoption needs T2-D241r on fresh seeds 12-23.
#   mem-guard bash scripts/local/t2_d241_read.sh      (after t2_d215r_read.sh and t2_d231_read.sh, which write the CNN route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
declare -A K=([dp10]="ll1ema0dp0.1mx0.4" [ll75]="ll0.75ema0dp0mx0.4")
for a in dp10 ll75; do
  mkdir -p "$O/t2_d241/stl_$a"
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d241-stl-$a-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for stl $a $p: ${d[*]}"; exit 2; }
    n=0
    for f in "${d[0]}"runs/${p}__fm-st_large-*"${K[$a]}"cg*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d241/stl_$a/$(basename "$f")" && n=$((n + 1)); done
    [ "$n" -eq 12 ] || { echo "stl $a $p: $n / 12 result files with ${K[$a]}"; exit 3; }
  done
  [ "$(ls "$O/t2_d241/stl_$a" | wc -l)" -eq 24 ] || { echo "stl $a: need 24 files"; exit 4; }
done
$PY -I - "$O/t2_d241/spec.json" <<'PYEOF' || exit 5
import json, sys
import numpy as np
m = json.load(open(sys.argv[1]))["members"]
for a in ("dp10", "ll75"):
    for p in ("kit", "seen1"):
        x, y = np.load(m["stl"][p].format(s=0)), np.load(m[f"stl_{a}"][p].format(s=0))
        assert np.array_equal(x["test_idx"], y["test_idx"]), (a, p)
        d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
        print(f"{a} {p} seed 0: vs the ST-L member max |d prob| {d:.3f}")
        assert d > 1e-4, "the runs equal the reference"
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d241/spec.json" --out "$O/t2_d241/read.json" > "$O/t2_d241/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d241/read.json" <<'PYEOF' | tee "$O/t2_d241/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ok = {}
for a in ("dp10", "ll75"):
    k = sum(rows[("kit", i, f"stl_{a}_alone")]["test"] - rows[("kit", i, "stl_alone")]["test"] for i in range(6)) / 6
    c = r["arms"][f"c_{a}"]
    met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
    ok[a] = k > -0.01 and met and c["new"]["d"] >= 0
    print(f"{a}: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_{a} vs c_rc75: mix {c['mix']['d']:+.4f} "
          f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f} -> "
          + ("KILLED" if k <= -0.01 else ("CANDIDATE" if ok[a] else "no")))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: r["arms"][f"c_{a}"]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch T2-D241r (fresh seeds 12-23) before any adoption" if best else "NOT ADOPTED (the ST-L recipe stays)")
PYEOF
