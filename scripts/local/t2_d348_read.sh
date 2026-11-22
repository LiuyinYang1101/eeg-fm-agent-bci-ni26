#!/usr/bin/env bash
# T2-D348 read (pre-registered 2026-11-05 20:29; this script written 2026-11-06 before any D348 package number): the one-member X4 shape
# (fbc_tv8c 0.4 / sh_tv8c 0.3 / ST-L 0.3, T2-D326's) under SKλ, 12 draws: anchor X4_5k = stl_rcll (r0, the shipped 5k CPT) vs X4_01k / X4_02k
# (PRIMARY) / X4_10k (the same seeds 0-11); the member-alone levels (the dose curve); and, REPORTED ONLY (not the selector), nm35 with r0 -> cpt02k.
#   mem-guard bash scripts/local/t2_d348_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d348
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; S = f"{O}/sofia_s1"
D = {"01k": ("ni26-t2p-d348-ft-kit-s01k-1568299", "ni26-t2p-d348-ft-se1-s01k-1566560"), "02k": ("ni26-t2p-d348-ft-kit-s02k-1566550", "ni26-t2p-d348-ft-se1-s02k-1566566"),
     "10k": ("ni26-t2p-d348-ft-kit-s10k-1566555", "ni26-t2p-d348-ft-se1-s10k-1566569")}
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M0 = s["members"]; nm = s["arms"]["nm35"]
M = {k: M0[k] for k in M0}
for k, (kd, sd) in D.items():
    M[f"cpt{k}"] = {"offset": 0}
    for p, d in (("kit", kd), ("seen1", sd)):
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (k, p, st)
        M[f"cpt{k}"][p] = st[0] + "__s{s}.npz"
        assert all(os.path.exists(M[f"cpt{k}"][p].format(s=i)) for i in range(12)), (k, p)
    print(k, os.path.basename(M[f"cpt{k}"]["kit"]))
x4 = lambda st: {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, st: 0.3}  # noqa: E731
arms = {"X4_5k": x4("r0"), "X4_01k": x4("cpt01k"), "X4_02k": x4("cpt02k"), "X4_10k": x4("cpt10k"),
        "m5k": {"r0": 1.0}, "m01k": {"cpt01k": 1.0}, "m02k": {"cpt02k": 1.0}, "m10k": {"cpt10k": 1.0}}
sw = dict(nm); sw["cpt02k"] = sw.pop("r0"); arms["nm35"] = nm; arms["nm35_02k"] = sw
json.dump({"n": 12, "anchor": "X4_5k", "members": M, "arms": arms}, open(f"{O}/t2_d348/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d348/spec.json": list(arms)}, open(f"{O}/t2_d348/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: X4_02k vs X4_5k"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" X4_5k X4_02k > "$R/cluster_X4_02k.txt" 2>&1; tail -12 "$R/cluster_X4_02k.txt"
