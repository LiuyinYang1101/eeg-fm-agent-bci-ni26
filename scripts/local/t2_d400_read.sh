#!/usr/bin/env bash
# T2-D400 read (pre-registered 2026-11-07 03:06; this script written before any D400 number): the spectral tilt on ALL FIVE of M6's ST-L
# members, each a tilt twin of the existing member on THE SAME seeds (so S3f's own members are the paired controls).
#   S3f_T5 (THE DECISION ARM) = S3f with lplh36 / lplh48 / stl_eaf / vp12 / lp10 -> their tilt twins; anchor S3f; singles T_<m> descriptive.
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI.
#   mem-guard bash scripts/local/t2_d400_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d400
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d400"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
pairs = {"lplh36": "lplh36", "lplh48": "lplh48", "stl_eaf": "stleaf", "vp12": "vp12", "lp10": "lp10"}
for m, tag in pairs.items():
    off = M[m]["offset"]; t = {"offset": off}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{S}/ni26-t2p-d400-{tag}_tl-{p}-[0-9]*"); assert len(ds) == 1, (tag, p, ds)
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (tag, p, st)
        assert "tl0.15" in st[0], (tag, st[0])
        # the twin differs from its control (the existing member) by the tilt tag only
        ctl = os.path.basename(M[m][p]).rsplit("__s", 1)[0]
        assert os.path.basename(st[0]).replace("tl0.15", "") == ctl, (tag, p, os.path.basename(st[0]), ctl)
        t[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(t[p].format(s=off + i)) for i in range(12)), (tag, p)
    M[f"{m}_tl"] = t
def sw(a, pr):
    a = dict(a)
    for o, n in pr: assert o in a and n not in a; a[n] = a.pop(o)
    return a
A = {"S3f": S3f, "S3f_T5": sw(S3f, [(m, f"{m}_tl") for m in pairs]), **{f"T_{m}": sw(S3f, [(m, f"{m}_tl")]) for m in pairs}}
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: S3f_T5 vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S3f S3f_T5 > "$R/cluster.txt" 2>&1; grep -E 'P0[4589]|P10|typical:|returning' "$R/cluster.txt"
