#!/usr/bin/env bash
# T2-D200 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the nine dirs
# ni26-t2p-d200-fbc-r1_pulxx_<vp2|vp2_pv5|vp2_pv10>-<kit|seen1>-<jobid> and ni26-t2p-d200-sh-r1_pulx_<vp2|vp2_pv5|vp2_pv10>-<jobid>:
# FBCNet blends at r 1/3 (seeds 120-131), ShallowNet blends at r 1/2 (seeds 84-95); t2p_pkgshape (anchor c_vp2 = the adopted package, copies
# unaugmented); per arm k in {vp2_pv5 (copies through the bank at 0.5), vp2_pv10 (the full bank)}: kill iff the FBCNet member alone (k - vp2)
# kit test on seeds 120-125 <= -0.01; adopt iff c_k vs c_vp2 meets bar d141 AND new >= 0; both met -> the larger mix delta.
#   mem-guard bash scripts/local/t2_d200_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for v in vp2 vp2_pv5 vp2_pv10; do r=r1_pulxx_$v; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d200-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d200/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds 120,121,122,123,124,125,126,127,128,129,130,131
done; done
for v in vp2 vp2_pv5 vp2_pv10; do r=r1_pulx_$v
  d=( "$S"/ni26-t2p-d200-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d200/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds 84,85,86,87,88,89,90,91,92,93,94,95
  done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d200/spec.json" --out "$O/t2_d200/read.json" > "$O/t2_d200/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d200/read.json" <<'PYEOF' | tee "$O/t2_d200/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ok = {}
for v in ("vp2_pv5", "vp2_pv10"):
    k = sum(rows[("kit", i, f"fbc_{v}_alone")]["test"] - rows[("kit", i, "fbc_vp2_alone")]["test"] for i in range(6)) / 6
    a = r["arms"][f"c_{v}"]
    met = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
    ok[v] = a["mix"]["d"] if met else None
    print(f"{v}: kill check (FBCNet alone vs vp2, kit test, seeds 120-125) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; "
          f"c_{v} vs c_vp2: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, "
          f"new {a['new']['d']:+.4f} -> {'MET' if met else 'not met'} (bar d141 + new >= 0)")
best = max((v for v in ok if ok[v] is not None), key=lambda v: ok[v], default=None)
print("VERDICT:", f"ADOPT {best}" if best else "NOT ADOPTED (the copies stay unaugmented)")
PYEOF
