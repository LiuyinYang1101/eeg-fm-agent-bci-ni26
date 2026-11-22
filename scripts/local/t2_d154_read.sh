#!/usr/bin/env bash
# T2-D154 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the six
# ni26-t2p-d154-fbc-<recipe>-<protocol>-<jobid> dirs (recipes r1_pul / r1_pue / r1_pus; kit + seen1):
# 1 the D124c blend (r 1/3, T2-D124b's cached fingerprints) of each recipe's FBCNet B3 + its per-user copies -> t2_d154/route_fbc_<recipe>
# 2 t2p_pkgshape on t2_d154/spec.json (anchor c0 = T2-D146's sofia r1 controls; arms c_pul / c_pue / c_pus)
# 3 the deciding arm = the highest mean of the kit_val and seen1_val deltas (no test index); its bar d141 line decides
#   mem-guard bash scripts/local/t2_d154_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
echo "### $(date +%T) blends"
for r in r1_pul r1_pue r1_pus; do
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d154-fbc-$r-$p-*/ )
    [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $r $p: ${d[*]}"; exit 2; }
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d154/route_fbc_$r" --model fbcnet_bboth --recipe $r
  done
  echo "$r: $(ls "$O"/t2_d154/route_fbc_$r/*.npz | wc -l) blended files"
done
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d154/spec.json" --out "$O/t2_d154/read.json" > "$O/t2_d154/read.txt" 2>&1
echo "rc $?"
$PY -I - "$O/t2_d154/read.json" <<'EOF' | tee -a "$O/t2_d154/read.txt"
import json, sys
a = json.load(open(sys.argv[1]))["arms"]
v = {k: (x["kit_val"]["d"] + x["seen1_val"]["d"]) / 2 for k, x in a.items()}
for k in sorted(v, key=v.get, reverse=True):
    print(f"selection (mean val delta): {k} {v[k]:+.4f}")
print(f"deciding arm (pre-registered: highest mean val delta): {max(v, key=v.get)}")
EOF
echo "### $(date +%T) done"
