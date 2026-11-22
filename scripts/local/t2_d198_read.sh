#!/usr/bin/env bash
# T2-D198 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d198-stl-<kit|seen1>-<jobid>:
# links the 24 result files into outputs/t2_d198/stl_vp2/, checks every one logged the bank, t2p_pkgshape on the adopted vp2 package (T2-D195's CNN
# files, seeds 96-107 / 60-71) with the ST-L member swapped; kill iff ST-L alone (vp2 - ctrl) kit test on seeds 0-5 <= -0.01; adopt iff
# c_vp2_stlvp vs c_vp2 meets bar d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_d198_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
mkdir -p "$O/t2_d198/stl_vp2"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d198-stl-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for stl $p: ${d[*]}"; exit 2; }
  n=$(grep -h "virtual participants on the target windows" "${d[0]}"*.log | wc -l); [ "$n" -eq 12 ] || { echo "stl $p: bank logged $n / 12 times"; exit 3; }
  for f in "${d[0]}"runs/${p}__fm-st_large-*vp2*__s*.npz; do ln -sfn "$f" "$O/t2_d198/stl_vp2/$(basename "$f")"; done
done
[ "$(ls "$O/t2_d198/stl_vp2" | wc -l)" -eq 24 ] || { echo "need 24 files"; exit 4; }
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d198/spec.json" --out "$O/t2_d198/read.json" > "$O/t2_d198/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d198/read.json" <<'PYEOF' | tee "$O/t2_d198/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "stl_vp2_alone")]["test"] - rows[("kit", i, "stl_alone")]["test"] for i in range(6)) / 6
a = r["arms"]["c_vp2_stlvp"]
ok = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
print(f"kill check (ST-L alone, kit test, seeds 0-5): {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}")
print(f"c_vp2_stlvp vs c_vp2: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, "
      f"new {a['new']['d']:+.4f} -> {'MET' if ok else 'not met'} (bar d141 + new >= 0)")
print("VERDICT:", "KILLED" if k <= -0.01 else ("ADOPT" if ok else "NOT ADOPTED"))
PYEOF
