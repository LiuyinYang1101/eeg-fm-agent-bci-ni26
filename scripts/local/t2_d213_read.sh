#!/usr/bin/env bash
# T2-D213 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the per-user copies' FT dose under vp2-180, read at w4321.
# FBCNet copies 80 (anchor c) / 40 (lo) / 160 (hi) epochs, ShallowNet copies 40 / 20 / 80, paired lo with lo and hi with hi; FBCNet seeds 180-191,
# ShallowNet 144-155. Per arm k in {lo, hi}: kill iff the FBCNet member alone (k - c) kit test on seeds 180-185 <= -0.01; adopt iff c_k vs c_c meets
# bar d141 AND new >= 0; both met -> the larger mix delta.
#   mem-guard bash scripts/local/t2_d213_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for v in vp2_e180 vp2_e180_pu40 vp2_e180_pu160; do r=r1_pulxx_$v; for p in kit seen1; do
  d=( "$S"/ni26-t2p-d213-fbc-$r-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for fbc $r $p: ${d[*]}"; exit 2; }
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d213/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 180 191)
done; done
for v in vp2_e180 vp2_e180_pu20 vp2_e180_pu80; do r=r1_pulx_$v
  d=( "$S"/ni26-t2p-d213-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for sh $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d213/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 144 155); done
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d213/spec.json" --out "$O/t2_d213/read.json" > "$O/t2_d213/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d213/read.json" <<'PYEOF' | tee "$O/t2_d213/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ok = {}
for v in ("lo", "hi"):
    k = sum(rows[("kit", i, f"fbc_{v}_alone")]["test"] - rows[("kit", i, "fbc_c_alone")]["test"] for i in range(6)) / 6
    a = r["arms"][f"c_{v}"]
    met = k > -0.01 and a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005 and a["new"]["d"] >= 0
    ok[v] = a["mix"]["d"] if met else None
    print(f"{v}: kill check (FBCNet alone vs c, kit test, seeds 180-185) {k:+.4f}; c_{v} vs c_c: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), "
          f"kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f} -> {'MET' if met else 'not met'}")
best = max((v for v in ok if ok[v] is not None), key=lambda v: ok[v], default=None)
print("VERDICT:", f"ADOPT {best}" if best else "NOT ADOPTED (copies 80 / 40 stay)")
PYEOF
