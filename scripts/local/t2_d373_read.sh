#!/usr/bin/env bash
# T2-D373 read (pre-registered 2026-11-06 07:24; written before any number): the stack of T2-D370's three singles on nm35 + SKλ, 12 draws.
#   mem-guard bash scripts/local/t2_d373_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d373
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import json, sys
O = sys.argv[1]
s = json.load(open(f"{O}/t2_d370/spec.json")); M = s["members"]; A = s["arms"]; nm = A["nm35"]
l2 = dict(A["L2"])                                    # r0 / r12 -> lplh0 / lplh12
slot = [k for k in ("lplh0", "lplh12", "r24", "vp12", "lp10") if k in l2]
s2 = {k: v for k, v in l2.items() if k not in slot}
s2.update({k: sum(l2[x] for x in slot) / (len(slot) + 1) for k in slot + ["sw2"]})
s3 = {**{k: v * 0.95 for k, v in s2.items()}, "mor_rc": 0.05}
arms = {"nm35": nm, "S3": s3, "S2": s2, "L2": A["L2"], "Ca": A["Ca"], "mor_rc_05": A["mor_rc_05"]}
for k, a in arms.items():
    assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d373/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d373/spec.json": list(arms)}, open(f"{O}/t2_d373/sk.json", "w"))
print({k: {m: round(w, 4) for m, w in a.items()} for k, a in arms.items() if k in ("S3", "S2")})
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')].set_index('arm')
print(d[['typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(float_format=lambda v: f'{v:+.4f}'))
s=d.loc['L2','typical']+d.loc['Ca','typical']+d.loc['mor_rc_05','typical']; print(f'sum of singles {s:+.4f}; S3 - sum {d.loc[\"S3\",\"typical\"]-s:+.4f}; S3 / sum {d.loc[\"S3\",\"typical\"]/s:.2f}')"
for a in S3 S2; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a > "$R/cluster_$a.txt" 2>&1; tail -12 "$R/cluster_$a.txt"; done
