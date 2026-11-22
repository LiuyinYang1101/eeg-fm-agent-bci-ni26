#!/usr/bin/env bash
# T2-D195 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the nine dirs
# ni26-t2p-d195-fbc-<r1_pulxx|r1_pulxx_vp|r1_pulxx_vp2>-<kit|seen1>-<jobid> and ni26-t2p-d195-sh-<r1_pulx|r1_pulx_vp|r1_pulx_vp2>-<jobid>:
# FBCNet blends at r 1/3 (seeds 96-107), ShallowNet blends at r 1/2 (seeds 60-71); t2p_pkgshape; then
#   kill (vp2 only): the FBCNet member alone (vp2 - ctrl), kit test, seeds 96-101 <= -0.01;
#   A: c_vp2 vs c0v (fresh seeds): bar d141 AND new >= 0;
#   B: c_vp vs c0v pooled over 24 seeds (T2-D193's 12 + these 12; the mean of the two reads' deltas, up-counts summed):
#      mix >= +0.005 with >= 18 / 24 up AND kit test >= -0.005 AND new >= 0;
#   both met -> the arm with the larger fresh-seed mix delta.
#   mem-guard bash scripts/local/t2_d195_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for r in r1_pulxx r1_pulxx_vp r1_pulxx_vp2; do for p in kit seen1; do
  d=( "$S"/ni26-t2p-d195-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d195/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds 96,97,98,99,100,101,102,103,104,105,106,107
done; done
for r in r1_pulx r1_pulx_vp r1_pulx_vp2; do
  d=( "$S"/ni26-t2p-d195-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d195/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds 60,61,62,63,64,65,66,67,68,69,70,71
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d195/spec.json" --out "$O/t2_d195/read.json" > "$O/t2_d195/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d195/read.json" "$O/t2_d193/read.json" <<'PYEOF' | tee "$O/t2_d195/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); r0 = json.load(open(sys.argv[2]))
rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "fbc_vp2_alone")]["test"] - rows[("kit", i, "fbc_ctrl_alone")]["test"] for i in range(6)) / 6
f = lambda a: f"mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}"
print(f"kill check vp2 (FBCNet member alone, kit test, seeds 96-101): {k:+.4f} -> {'KILL vp2' if k <= -0.01 else 'continue'}")
a2, a1, a0 = r["arms"]["c_vp2"], r["arms"]["c_vp"], r0["arms"]["c_vp"]
okA = k > -0.01 and a2["mix"]["d"] >= 0.005 and a2["mix"]["up"] >= 9 and a2["kit_test"]["d"] >= -0.005 and a2["new"]["d"] >= 0
print(f"A  c_vp2 vs c0v (seeds 96-107 / 60-71): {f(a2)} -> {'MET' if okA else 'not met'} (bar d141 + new >= 0)")
print(f"   c_vp  vs c0v (fresh replicate):       {f(a1)}")
P = {m: (a0[m]["d"] + a1[m]["d"]) / 2 for m in ("mix", "kit_test", "seen", "new")}; up = a0["mix"]["up"] + a1["mix"]["up"]
okB = P["mix"] >= 0.005 and up >= 18 and P["kit_test"] >= -0.005 and P["new"] >= 0
print(f"B  c_vp pooled 24 seeds (D193 + D195): mix {P['mix']:+.4f} ({up} / 24), kit test {P['kit_test']:+.4f}, seen {P['seen']:+.4f}, new {P['new']:+.4f} "
      f"-> {'MET' if okB else 'not met'} (mix >= +0.005, >= 18 / 24, kit test >= -0.005, new >= 0)")
pick = ("vp2" if a2["mix"]["d"] >= a1["mix"]["d"] else "vp") if okA and okB else ("vp2" if okA else ("vp" if okB else None))
print("VERDICT:", f"ADOPT {pick}" if pick else "NOT ADOPTED (virtual participants closed for T2)")
PYEOF
