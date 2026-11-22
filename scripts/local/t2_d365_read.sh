#!/usr/bin/env bash
# T2-D365 read (pre-registered 2026-11-06 04:44; written before any number): the tv8c per-user blends of the local d365 runs exactly as
# t2_d342_blend.sh / T2-D358 (FBCNet r 1/3 tag 3, ShallowNet EA r 0.5 tag 2), then t2_d320_rescreen.py (SKλ) on nm35 of t2_d345r/full with
# the pre-registered arms: R (primary) = both CNN members -> their us5 twins; Rf / Rs = one family only; A = each CNN family's weight split
# 1 : 1 between its shipped member and its twin (the mixed-recipe read). Outputs: outputs/t2_d365/{route_*,spec.json,sk.json,rescreen_sk.csv}.
#   mem-guard bash scripts/local/t2_d365_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs; S=$O/sofia_s1; R=$O/t2_d365
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
mkdir -p "$R"
for p in kit seen1; do
  blend --runs "$S/ni26-t2p-d365-fbc-us5-5090/runs" --protocols $p --out "$R/route_fbc_us5_tv8c" --model fbcnet_bboth \
    --recipe r1_pulxx_vp2_e180_vr_rc75_us5tv8c --seeds $(seq -s, 192 203)
  blend --runs "$S/ni26-t2p-d365-sh-us5-5090/runs" --protocols $p --out "$R/route_sh_us5_tv8c" --model shallow --align ea \
    --recipe r1_pulx_vp2_e180_vr_us5tv8c --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
done
$PY - "$O" <<'EOF'
import json, sys
O = sys.argv[1]
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
M["fbc_us5"] = {"offset": 192, **{p: f"{O}/t2_d365/route_fbc_us5_tv8c/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_us5tv8cfb3__s{{s}}.npz" for p in ("kit", "seen1")}}
M["sh_us5"] = {"offset": 156, **{p: f"{O}/t2_d365/route_sh_us5_tv8c/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_us5tv8cfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
def swap(a, pairs, split=False):
    a = dict(a)
    for old, new in pairs:
        w = a.pop(old)
        if split:
            a[old], a[new] = w / 2, w / 2
        else:
            a[new] = w
    return a
P = [("fbc_tv8c", "fbc_us5"), ("sh_tv8c", "sh_us5")]
arms = {"nm35": nm, "R": swap(nm, P), "Rf": swap(nm, P[:1]), "Rs": swap(nm, P[1:]), "A": swap(nm, P, split=True)}
for k, a in arms.items():
    assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d365/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d365/spec.json": list(arms)}, open(f"{O}/t2_d365/sk.json", "w"))
print({k: {m: round(w, 4) for m, w in a.items()} for k, a in arms.items() if k != "nm35"})
EOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 2 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in R A; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a > "$R/cluster_$a.txt" 2>&1; tail -12 "$R/cluster_$a.txt"; done
