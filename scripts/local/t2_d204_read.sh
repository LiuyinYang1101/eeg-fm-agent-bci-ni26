#!/usr/bin/env bash
# T2-D204 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the nine dirs
# ni26-t2p-d204-fbc-r1_pulxx_<vp2_e180|vp2_e240|vp2_e360>-<kit|seen1>-<jobid> and ni26-t2p-d204-sh-r1_pulx_<...>-<jobid>:
# FBCNet blends at r 1/3 (seeds 168-179), ShallowNet at r 1/2 (seeds 132-143); t2p_pkgshape (anchor c_vp2_e180 = the adopted T2-D203 recipe); per arm k in
# {vp2_e240, vp2_e360}: kill iff the FBCNet member alone (k - vp2_e180) kit test on seeds 168-173 <= -0.01; adopt iff c_k vs c_vp2_e180 meets bar d141
# AND new >= 0; several met -> the largest mix delta.
#   mem-guard bash scripts/local/t2_d204_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for v in vp2_e180 vp2_e240 vp2_e360; do r=r1_pulxx_$v; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d204-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d204/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 168 179)
done; done
for v in vp2_e180 vp2_e240 vp2_e360; do r=r1_pulx_$v
  d=( "$S"/ni26-t2p-d204-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d204/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 132 143); done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d204/spec.json" --out "$O/t2_d204/read.json" > "$O/t2_d204/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d204/read.json" <<'PYEOF' | tee "$O/t2_d204/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ok = {}
for v in ("vp2_e240", "vp2_e360"):
    k = sum(rows[("kit", i, f"fbc_{v}_alone")]["test"] - rows[("kit", i, "fbc_vp2_e180_alone")]["test"] for i in range(6)) / 6
    a = r["arms"][f"c_{v}"]
    met = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
    ok[v] = a["mix"]["d"] if met else None
    print(f"{v}: kill check (FBCNet alone vs e180, kit test, seeds 168-173) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; "
          f"c_{v} vs c_vp2_e180: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, "
          f"new {a['new']['d']:+.4f} -> {'MET' if met else 'not met'} (bar d141 + new >= 0)")
best = max((v for v in ok if ok[v] is not None), key=lambda v: ok[v], default=None)
print("VERDICT:", f"ADOPT {best}" if best else "NOT ADOPTED (vp2_e180 stands)")
PYEOF
