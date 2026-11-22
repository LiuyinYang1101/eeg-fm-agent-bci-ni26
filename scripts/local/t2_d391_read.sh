#!/usr/bin/env bash
# T2-D391 read (pre-registered 2026-11-06 21:57; this script written 22:31, before any D391 number): ONE ST-L member with EA + LP-FT
# (lp 10, head lr 3e-3), FRESH seeds 108-119 (sofia 1572569 kit / 1572570 seen1), swapped for M6's EA member.
#   S3f_lpea (THE DECISION ARM, anchor S3f = M6's shape) = S3f with stl_eaf -> lpea; the user-cluster CI.
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0.
#   mem-guard bash scripts/local/t2_d391_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d391
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d391"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert "stl_eaf" in S3f, sorted(S3f)
m = {"offset": 108}
for p, job in (("kit", 1572569), ("seen1", 1572570)):
    ds = glob.glob(f"{S}/ni26-t2p-d391-lpea-{p}-{job}"); assert len(ds) == 1, (p, ds)
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (p, st)
    assert "__ea__" in st[0] and "lh0.003" in st[0] and "lp10" in st[0] and "__fmE30b" in st[0], (p, st[0])   # EA + LP-FT, 30 epochs
    m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=108 + i)) for i in range(12)), (p, "seeds 108-119 incomplete")
M["lpea"] = m
a = dict(S3f); a["lpea"] = a.pop("stl_eaf")
A = {"S3f": S3f, "S3f_lpea": a}
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: S3f_lpea vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S3f S3f_lpea > "$R/cluster.txt" 2>&1; tail -8 "$R/cluster.txt"
