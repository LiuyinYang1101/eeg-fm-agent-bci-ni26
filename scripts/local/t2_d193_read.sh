#!/usr/bin/env bash
# T2-D193 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the six dirs
# ni26-t2p-d193-fbc-<r1_pulxx|r1_pulxx_vp>-<kit|seen1>-<jobid> and ni26-t2p-d193-sh-<r1_pulx|r1_pulx_vp>-<jobid>:
# FBCNet blends at r 1/3 (seeds 84-95), ShallowNet blends at r 1/2 (seeds 48-59); t2p_pkgshape; the kill check (FBCNet member alone,
# kit test, seeds 84-89: vp - ctrl <= -0.01 -> KILL); the decision (c_vp vs c0v: bar d141 AND new >= 0)
#   mem-guard bash scripts/local/t2_d193_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for r in r1_pulxx r1_pulxx_vp; do for p in kit seen1; do
  d=( "$S"/ni26-t2p-d193-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d193/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds 84,85,86,87,88,89,90,91,92,93,94,95
done; done
for r in r1_pulx r1_pulx_vp; do
  d=( "$S"/ni26-t2p-d193-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d193/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds 48,49,50,51,52,53,54,55,56,57,58,59
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d193/spec.json" --out "$O/t2_d193/read.json" > "$O/t2_d193/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d193/read.json" <<'PYEOF' | tee "$O/t2_d193/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "fbc_vp_alone")]["test"] - rows[("kit", i, "fbc_ctrl_alone")]["test"] for i in range(6)) / 6
print(f"kill check (FBCNet member alone, kit test, seeds 84-89): {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}")
a = r["arms"]["c_vp"]; ok = a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
print(f"c_vp vs c0v: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f} "
      f"-> {'MET' if ok else 'not met'} (bar d141 + new >= 0)")
print("VERDICT:", "KILLED" if k <= -0.01 else ("ADOPT" if ok else "NOT ADOPTED"))
PYEOF
