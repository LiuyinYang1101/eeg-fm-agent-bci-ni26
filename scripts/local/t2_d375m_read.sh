#!/usr/bin/env bash
# T2-D375 step 1.5 read (pre-registered 2026-11-06 08:43; written before any number): nm35 of t2_d345r/full under SKλ, 12 draws:
# A_cen = nm35 + fbc_cen 0.05 + sh_cen 0.05 (every other weight x 0.9) vs A_lin = the same with the raw-probe controls; + user-cluster CIs.
#   mem-guard bash scripts/local/t2_d375m_read.sh <fetched sofia run dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?the fetched ni26-t2p-d375-member-<job> dir}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d375m
mkdir -p "$R"
$PY - "$O" "$RUN" <<'PYEOF'
import json, os, sys
O, RUN = sys.argv[1:3]
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
for k, off in (("fbc_cen", 192), ("fbc_lin", 192), ("sh_cen", 156), ("sh_lin", 156)):
    M[k] = {"offset": off, **{p: f"{RUN}/{k}/{p}__s{{s}}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(M[k][p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), k
add = lambda a, ms: {**{k: v * (1 - 0.05 * len(ms)) for k, v in a.items()}, **{m: 0.05 for m in ms}}  # noqa: E731
arms = {"nm35": nm, "A_cen": add(nm, ["fbc_cen", "sh_cen"]), "A_lin": add(nm, ["fbc_lin", "sh_lin"]),
        "A_fcen": add(nm, ["fbc_cen"]), "A_scen": add(nm, ["sh_cen"]), "m_fcen": {"fbc_cen": 1.0}, "m_flin": {"fbc_lin": 1.0}, "m_scen": {"sh_cen": 1.0}, "m_slin": {"sh_lin": 1.0}}
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d375m/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d375m/spec.json": list(arms)}, open(f"{O}/t2_d375m/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')].set_index('arm')
print(d[['typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(float_format=lambda v: f'{v:+.4f}'))
print(f\"A_cen - A_lin = {d.loc['A_cen','typical']-d.loc['A_lin','typical']:+.4f}\")"
for a in A_cen A_lin; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a > "$R/cluster_$a.txt" 2>&1; tail -12 "$R/cluster_$a.txt"; done
