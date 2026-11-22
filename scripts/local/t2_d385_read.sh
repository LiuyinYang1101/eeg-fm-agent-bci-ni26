#!/usr/bin/env bash
# T2-D385 read (pre-registered 2026-11-06 15:04; this script written before any D385 number): M6's own ST-L members at 100 epochs.
#   S3f100 (THE DECISION ARM, anchor S3f = M6's shape) = S3f with lplh36 -> LP-FT + lh3 e100 (seeds 72-83), lplh48 -> LP-FT + lh3 e100 (84-95),
#          stl_eaf -> EA e100 (96-107); vp12 / lp10 kept (30 epochs). Singles (one swap each) descriptive; the user-cluster CI.
# Bar: S3f100 vs S3f typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0.
#   mem-guard bash scripts/local/t2_d385_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d385
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d385"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert {"lplh36", "lplh48", "stl_eaf", "vp12", "lp10"} <= set(S3f), sorted(S3f)
def stl(arm, off):
    m = {"offset": off}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{S}/ni26-t2p-d385-{arm}-{p}-[0-9]*"); assert len(ds) == 1, (arm, p, ds)
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (arm, p, st)
        assert "__fmE100b" in st[0], (arm, p, st[0])                                       # the stem carries the fine-tune length
        assert ("lh0.003" in st[0] and "lp10" in st[0]) == arm.startswith("lplh") and ("__ea__" in st[0]) == arm.startswith("ea"), (arm, st[0])
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=off + i)) for i in range(12)), (arm, p)
    return m
M["lpa100"] = stl("lplh_e100_s72", 72); M["lpb100"] = stl("lplh_e100_s84", 84); M["ea100"] = stl("ea_e100_s96", 96)
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
P = [("lplh36", "lpa100"), ("lplh48", "lpb100"), ("stl_eaf", "ea100")]
A = {"S3f": S3f, "S3f100": sw(S3f, P), "LPa": sw(S3f, P[:1]), "LPb": sw(S3f, P[1:2]), "EA100": sw(S3f, P[2:])}
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: S3f100 vs S3f (the decision arm)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S3f S3f100 > "$R/cluster_S3f100.txt" 2>&1; tail -8 "$R/cluster_S3f100.txt"
