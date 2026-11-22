#!/usr/bin/env bash
# T2-D215r read (local CPU, zero compute; DESCRIPTIVE), as pre-registered in docs/DECISIONS_T2.md: the virtual-reference recipes on T2-D216's fresh
# seeds (FBCNet 204-215, ShallowNet 168-179) vs T2-D216's anchor c, at w4321; prints the replicate and the pooled 24-index means with T2-D215.
#   mem-guard bash scripts/local/t2_d215r_read.sh      (after scripts/local/t2_d216_read.sh, which writes the anchor's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d216/route_fbc_r1_pulxx_vp2_e180/kit__*__s204.npz "$O"/t2_d216/route_sh_r1_pulx_vp2_e180/seen1__*__s168.npz >/dev/null || { echo "run t2_d216_read.sh first"; exit 2; }
r=r1_pulxx_vp2_e180_vr; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d215r-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d215r/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 204 215)
done
r=r1_pulx_vp2_e180_vr
d=( "$S"/ni26-t2p-d215r-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d215r/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 168 179); done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d215r/spec.json" --out "$O/t2_d215r/read.json" > "$O/t2_d215r/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d215r/read.json" "$O/t2_d215/read.json" <<'PYEOF' | tee "$O/t2_d215r/verdict.txt"
import json, sys
a, b = (json.load(open(f))["arms"]["c_vr"] for f in sys.argv[1:3])
print("replicate c_vr vs c_c: " + ", ".join(f"{q} {a[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new")) + f" (mix {a['mix']['up']} / 12)")
pool = {q: (a[q]["d"] + b[q]["d"]) / 2 for q in ("mix", "kit_test", "seen", "new")}
print("pooled 24 indices with T2-D215: " + ", ".join(f"{q} {v:+.4f}" for q, v in pool.items()) + f" (mix {a['mix']['up'] + b['mix']['up']} / 24)")
print("FLAG:", "pooled mix or new < 0 -> tell the user before any vr build" if pool["mix"] < 0 or pool["new"] < 0 else "none (T2-D215 stands)")
PYEOF
