#!/usr/bin/env bash
# T2-D367 read (pre-registered 2026-11-06 05:35; written before any number): the tv8c per-user blend of the local d367 FBCNet runs exactly as
# T2-D358 / D365 (FBCNet r 1/3 tag 3), then t2_d320_rescreen.py (SKλ) on nm35 of t2_d345r/full with the pre-registered arms:
# Rf (primary) = fbc_tv8c -> its SSL-init twin; Af = the FBCNet weight split 1 : 1. Plus the user-cluster CI and the SSL loss curves.
# Outputs: outputs/t2_d367/{route_fbc_ssl1_tv8c,spec.json,sk.json,rescreen_sk.csv,cluster_*.txt}.
#   mem-guard bash scripts/local/t2_d367_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs; S=$O/sofia_s1; R=$O/t2_d367; D=$S/ni26-t2p-d367-fbc-ssl-5090
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
mkdir -p "$R"
for p in kit seen1; do
  $PY -c "import json; s=json.load(open('$D/ssl_$p.pt.json')); print('ssl $p:', s['n_export'], '+', s['n_crops'], 'windows,', s['steps'], 'steps, loss', s['loss_curve'][0], '->', s['loss_curve'][-1], '(chance', round(s['ln_2bs'], 3), ')')"
  blend --runs "$D/runs" --protocols $p --out "$R/route_fbc_ssl1_tv8c" --model fbcnet_bboth \
    --recipe r1_pulxx_vp2_e180_vr_rc75inissl1tv8c --seeds $(seq -s, 192 203)
done
$PY - "$O" <<'EOF'
import json, sys
O = sys.argv[1]
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
M["fbc_ssl"] = {"offset": 192, **{p: f"{O}/t2_d367/route_fbc_ssl1_tv8c/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75inissl1tv8cfb3__s{{s}}.npz" for p in ("kit", "seen1")}}
rf = dict(nm); rf["fbc_ssl"] = rf.pop("fbc_tv8c")
af = dict(nm); w = af.pop("fbc_tv8c"); af["fbc_tv8c"], af["fbc_ssl"] = w / 2, w / 2
arms = {"nm35": nm, "Rf": rf, "Af": af}
for k, a in arms.items():
    assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d367/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d367/spec.json": list(arms)}, open(f"{O}/t2_d367/sk.json", "w"))
EOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 2 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in Rf Af; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a > "$R/cluster_$a.txt" 2>&1; tail -12 "$R/cluster_$a.txt"; done
