#!/usr/bin/env bash
# T2-D402 read (backlog 96; pre-registered 2026-11-07 before any number; this script written before any D402 number): G3's PACKAGE-LOSS
# training of M6's vp12 ST-L member (CE + 1 x NLL((1 - 0.35) p_CNN_oof + 0.35 p_member), T2-D253's out-of-fold CNN mix), PAIRED on the
# member's own seeds 12-23 (the existing vp12 member = the beta-0 control).
#   PK (THE DECISION ARM) = S3f with vp12 -> vp12_pk vs S3f (anchor); m_ctl / m_pk = the member alone (descriptive)
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI. Window check (descriptive, pre-registered):
#   of the windows whose decoded argmax changes, fixed share >= 0.55, and >= 60 % of the fixed windows are ones the CNN sub-mix gets wrong.
#   mem-guard bash scripts/local/t2_d402_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d402
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d402"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert "vp12" in S3f
off = M["vp12"]["offset"]; t = {"offset": off}
for p in ("kit", "seen1"):
    ds = glob.glob(f"{S}/ni26-t2p-d402-vp12_pk-{p}-[0-9]*"); assert len(ds) == 1, (p, ds)
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (p, st)
    assert "pk1w0.35" in st[0], st[0]
    ctl = os.path.basename(M["vp12"][p]).rsplit("__s", 1)[0]
    assert os.path.basename(st[0]).replace("pk1w0.35", "") == ctl, (p, os.path.basename(st[0]), ctl)   # differs by the package loss only
    t[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(t[p].format(s=off + i)) for i in range(12)), p
M["vp12_pk"] = t
pk = dict(S3f); pk["vp12_pk"] = pk.pop("vp12")
A = {"S3f": S3f, "PK": pk, "m_ctl": {"vp12": 1.0}, "m_pk": {"vp12_pk": 1.0}}
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: PK vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S3f PK > "$R/cluster.txt" 2>&1; grep -E 'P0[4589]|P10|typical:|returning' "$R/cluster.txt"
echo "== window check (descriptive)"
$PY -I - "$E" "$R/spec.json" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd() / "tracks/t2_bci/scripts"))
from t2p_report import causal_sinkhorn
d = pd.read_csv(f"{sys.argv[1]}/index.csv", dtype={"subject": str, "session": str}); spec = json.load(open(sys.argv[2]))
M, A = spec["members"], spec["arms"]; strm, ordr, y = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy()
cnn = {m: w for m, w in A["S3f"].items() if m in ("fbc_tl15f", "sh_tv8c", "ifn")}; assert abs(sum(cnn.values()) - 0.65) < 1e-6, cnn
def mix(w, proto, split, i):
    ix, P = None, 0.0
    for m, wm in w.items():
        z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
        ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
        assert ix is None or np.array_equal(ix, ix_); ix, P = ix_, P + wm * p.astype(np.float64)
    return ix, P
fixed = broken = fixed_cnnwrong = 0
for i in range(spec["n"]):
    for proto in ("kit", "seen1"):
        for split in ("test", "val"):
            ix, Pa = mix(A["S3f"], proto, split, i); _, Pb = mix(A["PK"], proto, split, i); _, Pc = mix(cnn, proto, split, i)
            sk = lambda P: causal_sinkhorn(P, (strm[ix], ordr[ix]), 10, 3, 0.95).argmax(1)
            a, b, c = sk(Pa), sk(Pb), Pc.argmax(1); yy = y[ix]
            f_ = (a != yy) & (b == yy); fixed += int(f_.sum()); broken += int(((a == yy) & (b != yy)).sum()); fixed_cnnwrong += int((f_ & (c != yy)).sum())
ch = fixed + broken
print(f"[D402] changed-and-scored windows {ch}: fixed {fixed} ({fixed / max(ch, 1):.2f}), broken {broken}; fixed where the CNN sub-mix is wrong "
      f"{fixed_cnnwrong / max(fixed, 1):.2f} (pre-registered: fixed share >= 0.55, CNN-wrong share of fixed >= 0.60)")
PYEOF
