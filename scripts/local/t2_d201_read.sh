#!/usr/bin/env bash
# T2-D201 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the fresh-seed replicate of T2-D200's pv10 arm (FBCNet seeds
# 132-143, ShallowNet 96-107), decided on 24 seeds pooled with T2-D200's read (the mean of the two Deltas, up-counts summed):
# adopt iff mix >= +0.005 with >= 18 / 24 up AND kit test >= -0.005 AND new >= 0 (T2-D200's clauses on 24 seeds).
#   mem-guard bash scripts/local/t2_d201_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for v in vp2 vp2_pv10; do r=r1_pulxx_$v; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d201-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d201/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 132 143)
done; done
for v in vp2 vp2_pv10; do r=r1_pulx_$v
  d=( "$S"/ni26-t2p-d201-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d201/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 96 107); done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d201/spec.json" --out "$O/t2_d201/read.json" > "$O/t2_d201/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d201/read.json" "$O/t2_d200/read.json" <<'PYEOF' | tee "$O/t2_d201/verdict.txt"
import json, sys
a1 = json.load(open(sys.argv[1]))["arms"]["c_vp2_pv10"]; a0 = json.load(open(sys.argv[2]))["arms"]["c_vp2_pv10"]
f = lambda a: f"mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}"
print(f"replicate (seeds 132-143 / 96-107): {f(a1)}")
P = {m: (a0[m]["d"] + a1[m]["d"]) / 2 for m in ("mix", "kit_test", "seen", "new")}; up = a0["mix"]["up"] + a1["mix"]["up"]
ok = P["mix"] >= 0.005 and up >= 18 and P["kit_test"] >= -0.005 and P["new"] >= 0
print(f"pooled 24 seeds (D200 + D201): mix {P['mix']:+.4f} ({up} / 24), kit test {P['kit_test']:+.4f}, seen {P['seen']:+.4f}, new {P['new']:+.4f} "
      f"-> {'MET' if ok else 'not met'} (mix >= +0.005, >= 18 / 24, kit test >= -0.005, new >= 0)")
print("VERDICT:", "ADOPT pv10" if ok else "NOT ADOPTED (the copies stay unaugmented)")
PYEOF
