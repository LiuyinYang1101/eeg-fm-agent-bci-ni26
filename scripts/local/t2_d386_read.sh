#!/usr/bin/env bash
# T2-D386 read (pre-registered 2026-11-06; this script written before any D386 number): the MOR dose read for the in-house FM project's B.
# Five MOR members, each T2-D256's rc 0.75 MOR line on FRESH seeds 24-35: p1v2 (the package's MOR, pinned), mor100 (B's init = dose 0),
# B5k / B10k / B20k (the in-house FM project's annealed continuations on its pool + the ni26 pools).
#   spec_a (THE DOSE CURVE, member alone, anchor m_mor100): m_B5k / m_B10k / m_B20k / m_p1v2.
#   spec_b (package, anchor S3f = M6's shape, SKλ): SW_<x> = vp12 -> the MOR member at its 0.07 (runtime-neutral: THE ONLY CANDIDATE ARMS);
#          AD_<x> = + the MOR member at 0.05 (D377's add_frac; over 40 ms: informational only).
# Bar (a swap arm): SW_x vs S3f typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0.
#   mem-guard bash scripts/local/t2_d386_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d386
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d386"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert "vp12" in S3f, sorted(S3f)
X = ("p1v2", "mor100", "B5k", "B10k", "B20k")
for x in X:
    m = {"offset": 24}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{S}/ni26-t2p-d386-{x}-{p}-[0-9]*"); assert len(ds) == 1, (x, p, ds)
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-mor_p1v2-mean__*__s*.npz")}); assert len(st) == 1, (x, p, st)
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=24 + i)) for i in range(12)), (x, p)
    M[f"mor_{x}"] = m
stems = {x: M[f"mor_{x}"]["kit"] for x in X}
assert len(set(stems.values())) == 5, "the five MOR members must be five different runs (the CPT sha is in the stem)"
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a, (o, n); a[n] = a.pop(o); return a
def add(a, n, frac=0.05):
    tot = sum(a.values()); a = {k: v * (tot - frac) / tot for k, v in a.items()}; a[n] = frac; return a
A = {"m_mor100": {"mor_mor100": 1.0}, **{f"m_{x}": {f"mor_{x}": 1.0} for x in X if x != "mor100"}}
B = {"S3f": S3f, **{f"SW_{x}": sw(S3f, "vp12", f"mor_{x}") for x in X}, **{f"AD_{x}": add(S3f, f"mor_{x}") for x in X}}
for a in list(A.values()) + list(B.values()):
    assert abs(sum(a.values()) - 1) < 1e-9, a
json.dump({"n": 12, "anchor": "m_mor100", "members": M, "arms": A}, open(f"{R}/spec_a.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": B}, open(f"{R}/spec_b.json", "w"), indent=1)
json.dump({f"{R}/spec_a.json": list(A), f"{R}/spec_b.json": list(B)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for x in p1v2 mor100 B5k B10k B20k; do
  echo "== user-cluster CI: SW_$x vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_b.json" S3f SW_$x > "$R/cluster_SW_$x.txt" 2>&1; tail -3 "$R/cluster_SW_$x.txt"
done
