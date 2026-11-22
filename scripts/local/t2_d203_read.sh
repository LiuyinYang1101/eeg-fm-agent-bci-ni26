#!/usr/bin/env bash
# T2-D203 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the nine dirs
# ni26-t2p-d203-fbc-r1_pulxx_<vp2|vp2_e120|vp2_e180>-<kit|seen1>-<jobid> and ni26-t2p-d203-sh-r1_pulx_<...>-<jobid> (FBCNet seeds 156-167, ShallowNet 120-131):
#   kill e180 iff the FBCNet member alone (e180 - vp2) kit test on seeds 156-161 <= -0.01;
#   A: c_vp2_e180 vs c_vp2 (fresh seeds): bar d141 AND new >= 0;
#   B: c_vp2_e120 vs c_vp2 pooled over 24 seeds (T2-D202's 12 + these 12; the mean of the two Deltas, up-counts summed):
#      mix >= +0.005 with >= 18 / 24 up AND kit test >= -0.005 AND new >= 0;
#   both met -> the larger fresh-seed mix delta.
#   mem-guard bash scripts/local/t2_d203_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for v in vp2 vp2_e120 vp2_e180; do r=r1_pulxx_$v; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d203-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d203/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 156 167)
done; done
for v in vp2 vp2_e120 vp2_e180; do r=r1_pulx_$v
  d=( "$S"/ni26-t2p-d203-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d203/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 120 131); done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d203/spec.json" --out "$O/t2_d203/read.json" > "$O/t2_d203/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d203/read.json" "$O/t2_d202/read.json" <<'PYEOF' | tee "$O/t2_d203/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); r0 = json.load(open(sys.argv[2]))
rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
f = lambda a: f"mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}"
k = sum(rows[("kit", i, "fbc_vp2_e180_alone")]["test"] - rows[("kit", i, "fbc_vp2_alone")]["test"] for i in range(6)) / 6
print(f"kill check e180 (FBCNet alone, kit test, seeds 156-161): {k:+.4f} -> {'KILL e180' if k <= -0.01 else 'continue'}")
a2, a1, a0 = r["arms"]["c_vp2_e180"], r["arms"]["c_vp2_e120"], r0["arms"]["c_vp2_e120"]
okA = k > -0.01 and a2["mix"]["d"] >= 0.005 and a2["mix"]["up"] >= 9 and a2["kit_test"]["d"] >= -0.005 and a2["new"]["d"] >= 0
print(f"A  c_vp2_e180 vs c_vp2: {f(a2)} -> {'MET' if okA else 'not met'} (bar d141 + new >= 0)")
print(f"   c_vp2_e120 vs c_vp2 (fresh replicate): {f(a1)}")
P = {m: (a0[m]["d"] + a1[m]["d"]) / 2 for m in ("mix", "kit_test", "seen", "new")}; up = a0["mix"]["up"] + a1["mix"]["up"]
okB = P["mix"] >= 0.005 and up >= 18 and P["kit_test"] >= -0.005 and P["new"] >= 0
print(f"B  c_vp2_e120 pooled 24 seeds (D202 + D203): mix {P['mix']:+.4f} ({up} / 24), kit test {P['kit_test']:+.4f}, seen {P['seen']:+.4f}, new {P['new']:+.4f} "
      f"-> {'MET' if okB else 'not met'} (mix >= +0.005, >= 18 / 24, kit test >= -0.005, new >= 0)")
pick = ("e180" if a2["mix"]["d"] >= a1["mix"]["d"] else "e120") if okA and okB else ("e180" if okA else ("e120" if okB else None))
print("VERDICT:", f"ADOPT vp2_{pick}" if pick else "NOT ADOPTED (vp2 at 60 epochs stands)")
PYEOF
