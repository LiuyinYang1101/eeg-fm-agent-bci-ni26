#!/usr/bin/env bash
# T2-D144 50k read + T2-D151 member gate (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
# the bench dirs (D144: 1521500-05; D151: 1521506 / 07). Specs written mechanically from the rows (outputs/t2_d144/spec_50k.json,
# outputs/t2_d151/spec_member.json), then:
# D144: per arm, kit val selects the config (A: le 1e-3 / 3e-4 x plain / ctp; B: le 1e-3 / 3e-4); kill unless the selected kit test delta
#       vs mor_p1v2 >= -0.050 (6 seeds); both pass -> the higher selected kit test mean goes to 100k; step-2 gate = selected kit test mean >= 0.4681
# D151: per arm vs mortp, gate = mix delta >= +0.010 with >= 8 / 12 up AND kit test delta >= -0.005; both pass -> the higher mix goes on
#   mem-guard bash scripts/local/t2_d144_d151_reads.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
for x in "t2_d144 spec_50k read_50k" "t2_d151 spec_member read_member"; do
  set -- $x
  echo "### $(date +%T) pkgshape $1/$2"
  $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/$1/$2.json" --out "$O/$1/$3.json" > "$O/$1/$3.txt" 2>&1; echo "rc $?"
done
$PY -I - "$O/t2_d144/read_50k.json" "$O/t2_d151/read_member.json" <<'PYEOF' | tee "$O/t2_d144/verdict_50k.txt"
import json, sys
a = json.load(open(sys.argv[1])); rows = a["arms"]
print("T2-D144 50k (member alone vs mor_p1v2, 6 seeds; kit test of mor_p1v2 = 0.4681 per D141)")
sel = {}
for arm in ("A", "B"):
    c = {k: v for k, v in rows.items() if k.startswith(arm + "_")}
    k = max(c, key=lambda k: c[k]["kit_val"]["d"])
    sel[arm] = (k, c[k]["kit_test"]["d"])
    for kk in sorted(c):
        print(f"  {kk:16s} kit val {c[kk]['kit_val']['d']:+.4f}  kit test {c[kk]['kit_test']['d']:+.4f}  mix {c[kk]['mix']['d']:+.4f}{'  <- selected' if kk == k else ''}")
    print(f"  arm {arm}: selected {k}, kit test delta {sel[arm][1]:+.4f} -> {'CONTINUES (>= -0.050)' if sel[arm][1] >= -0.050 else 'KILLED (< -0.050)'}; step-2 gate (>= 0) {'met' if sel[arm][1] >= 0 else 'not met'}")
ok = [arm for arm in sel if sel[arm][1] >= -0.050]
print(f"  -> cycle 2: {max(ok, key=lambda arm: sel[arm][1]) if ok else 'none (TP-2 closed for T2)'}")
b = json.load(open(sys.argv[2]))["arms"]
print("T2-D151 member gate (vs mortp, 12 seeds): mix >= +0.010 with >= 8 / 12 up AND kit test >= -0.005")
p = []
for k in ("wA", "wB"):
    x = b[k]; g = x["mix"]["d"] >= 0.010 and x["mix"]["up"] >= 8 and x["kit_test"]["d"] >= -0.005
    p += [k] if g else []
    print(f"  {k}: mix {x['mix']['d']:+.4f} ({x['mix']['up']} / 12 up), kit test {x['kit_test']['d']:+.4f}, seen {x['seen']['d']:+.4f}, new {x['new']['d']:+.4f} -> {'PASS' if g else 'fail'}")
print(f"  -> package read: {max(p, key=lambda k: b[k]['mix']['d']) if p else 'none (T2 MOR line closed for pretraining)'}")
PYEOF
echo "### $(date +%T) done"
