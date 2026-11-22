#!/usr/bin/env bash
# T2-X1 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the MOR member with the amplitude side input
# (--aligns none --fm-side amp) vs its fresh control (the shipped recipe, --aligns wz), MOR seeds 12-23, inside the w4321 package with
# T2-D214's anchor CNN routes (FBCNet 192-203, ShallowNet 156-167) and the ST-L member. Kill iff the MOR alone (x1 - c) kit test on
# seeds 12-17 <= -0.01; adopt iff c_x1 vs c_c meets bar d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_x1_read.sh      (after scripts/local/t2_d214_read.sh, which writes the anchor's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
ls "$O"/t2_d214/route_fbc_r1_pulxx_vp2_e180/kit__*__s192.npz "$O"/t2_d214/route_sh_r1_pulx_vp2_e180/seen1__*__s156.npz >/dev/null || { echo "run t2_d214_read.sh first"; exit 2; }
for a in c x1; do for p in kit seen1; do
  d=( "$S"/ni26-t2p-x1-mor-$a-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for mor $a $p: ${d[*]}"; exit 2; }
  [ "$(ls "${d[0]}runs"/"${p}"__*.npz 2>/dev/null | wc -l)" -eq 12 ] || { echo "mor $a $p: not 12 result files"; exit 2; }
  ln -sfn "${d[0]}runs" "$O/t2_x1/mor_${a}_$p"
done; done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_x1/spec.json" --out "$O/t2_x1/read.json" > "$O/t2_x1/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_x1/read.json" <<'PYEOF' | tee "$O/t2_x1/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "mor_x1_alone")]["test"] - rows[("kit", i, "mor_c_alone")]["test"] for i in range(6)) / 6
a, m1, m0 = r["arms"]["c_x1"], r["arms"]["mor_x1_alone"], r["arms"]["mor_c_alone"]   # every arm's d is vs c_c: alone x1 - c = m1 - m0
met = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
print(f"x1: kill check (MOR alone vs c, kit test, seeds 12-17) {k:+.4f}; c_x1 vs c_c: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), "
      f"kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f} -> {'MET' if met else 'not met'}")
print("   (descriptive) MOR alone, x1 - c: " + ", ".join(f"{q} {m1[q]['d'] - m0[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new")))
print("VERDICT:", "ADOPT x1 (template work next: MOR align none + AmpSide in MORMemberT2, D140 bars)" if met else "NOT ADOPTED (MOR stays wz, no side input)")
PYEOF
