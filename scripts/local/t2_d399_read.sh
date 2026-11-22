#!/usr/bin/env bash
# T2-D399 read (backlog 94; pre-registered 2026-11-07 before any number; this script written before any number): the CNN spectral tilt on the
# ST-L fine-tune, PAIRED on fresh seeds 124-135 (tilt 0.15 vs no tilt, the shipped plain line otherwise). On M6's shape (S3f), runtime-neutral:
#   TL = S3f with vp12 -> the tilt member (THE DECISION ARM) vs CTL = S3f with vp12 -> the paired no-tilt member (anchor); member alone too.
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI.
#   mem-guard bash scripts/local/t2_d399_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d399
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d399"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert "vp12" in S3f
def stl(arm, tilt):
    m = {"offset": 124}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{S}/ni26-t2p-d399-{arm}-{p}-[0-9]*"); assert len(ds) == 1, (arm, p, ds)
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (arm, p, st)
        assert ("tl0.15" in st[0]) == tilt and "__fmE30b" in st[0], (arm, st[0])
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=124 + i)) for i in range(12)), (arm, p)
    return m
M["stl_tl"] = stl("stl_tl15_s124", True); M["stl_ctl"] = stl("stl_ctl_s124", False)
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a; a[n] = a.pop(o); return a
A = {"CTL": sw(S3f, "vp12", "stl_ctl"), "TL": sw(S3f, "vp12", "stl_tl"), "m_ctl": {"stl_ctl": 1.0}, "m_tl": {"stl_tl": 1.0}}
json.dump({"n": 12, "anchor": "CTL", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: TL vs CTL"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" CTL TL > "$R/cluster.txt" 2>&1; grep -E 'P0[4589]|P10|typical:|returning' "$R/cluster.txt"
