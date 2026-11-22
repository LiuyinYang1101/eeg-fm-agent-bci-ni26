#!/usr/bin/env bash
# T2-D215 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: virtual references (a random partial re-reference in each
# vp2 bank draw) under vp2-180, read at w4321 against T2-D214's anchor c (the same seeds: FBCNet 192-203, ShallowNet 156-167). Kill iff the
# FBCNet member alone (vr - c) kit test on seeds 192-197 <= -0.01; adopt iff c_vr vs c_c meets bar d141 AND new >= 0.
#   mem-guard bash scripts/local/t2_d215_read.sh      (after scripts/local/t2_d214_read.sh, which writes the anchor's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d214/route_fbc_r1_pulxx_vp2_e180/kit__*__s192.npz "$O"/t2_d214/route_sh_r1_pulx_vp2_e180/seen1__*__s156.npz >/dev/null || { echo "run t2_d214_read.sh first"; exit 2; }
r=r1_pulxx_vp2_e180_vr; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d215-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d215/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 192 203)
done
r=r1_pulx_vp2_e180_vr
d=( "$S"/ni26-t2p-d215-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d215/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 156 167); done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d215/spec.json" --out "$O/t2_d215/read.json" > "$O/t2_d215/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d215/read.json" <<'PYEOF' | tee "$O/t2_d215/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "fbc_vr_alone")]["test"] - rows[("kit", i, "fbc_c_alone")]["test"] for i in range(6)) / 6
a = r["arms"]["c_vr"]
met = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
print(f"vr: kill check (FBCNet alone vs c, kit test, seeds 192-197) {k:+.4f}; c_vr vs c_c: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), "
      f"kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f} -> {'MET' if met else 'not met'}")
print("VERDICT:", "ADOPT vr" if met else "NOT ADOPTED (no virtual references)")
PYEOF
