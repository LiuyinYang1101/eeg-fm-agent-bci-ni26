#!/usr/bin/env bash
# T2-D379 read (pre-registered 2026-11-06 09:49; this script written 2026-11-06 11:20, before any D379 number): the 100-epoch ST-L slot on
# FRESH seeds 60-71 (sofia 1569198 kit / 1569199 seen1). nm35 of t2_d345r/full, SKλ, 12 draws.
#   SL100f   (THE DECISION ARM) = nm35 with r0 / r12 / r24 / vp12 -> fresh e100 restarts j = 0..3 (draw i: seed 60 + (i + 3j) mod 12), lp10 kept.
#   m30 / m100f (informational) = the ST-L member alone at 30 (r0) vs fresh 100 epochs (restart 0).
#   S3f_vp100 (INFORMATIONAL ONLY, not a decision arm; added after D378 / D380, before any D379 number): M6's shape (nm35 + the D378 swaps)
#            with its one untouched ST-L member vp12 -> e100 restart 0. A combination candidate would still need its own pre-registered row.
# Bar (the row's): typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI reported.
#   mem-guard bash scripts/local/t2_d379_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d379
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
O = sys.argv[1]; S = f"{O}/sofia_s1"
base = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(base["members"]); nm = base["arms"]["nm35"]
assert set(nm) >= {"r0", "r12", "r24", "vp12", "lp10"}, sorted(nm)
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M.update({k: d378["members"][k] for k in ("stl_eaf", "lplh36", "lplh48", "fbc_tl15f")})
S3f = d378["arms"]["S3f"]
Rx = re.compile(r"\]\s+ep (\d+) loss ([\d.]+) val ([\d.na]+) test ([\d.na]+)")
src = {}
for p, job in (("kit", 1569198), ("seen1", 1569199)):
    ds = glob.glob(f"{S}/ni26-t2p-*-{job}"); assert len(ds) == 1, (p, job, ds)
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (p, st)
    src[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(src[p].format(s=60 + i)) for i in range(12)), (p, "seeds 60-71 incomplete")
    cur = {}
    for f in glob.glob(f"{ds[0]}/*.log"):
        for line in open(f):
            m = Rx.search(line)
            if m:
                cur.setdefault(int(m.group(1)), []).append([float(m.group(2))] + [float(x) if x != "nan" else np.nan for x in m.group(3, 4)])
    es = [e for e in (1, 10, 30, 50, 70, 100) if e in cur]
    print(f"curves e100f {p} (epoch: loss / val / test):", {e: np.nanmean(np.array(cur[e]), 0).round(4).tolist() for e in es})
for j in range(4):
    dj = f"{O}/t2_d379/restarts/e100f{j}"; os.makedirs(dj, exist_ok=True); M[f"e100f{j}"] = {"offset": 0}
    for p in ("kit", "seen1"):
        for i in range(12):
            ln = f"{dj}/{p}__s{i}.npz"
            if os.path.lexists(ln): os.remove(ln)
            os.symlink(src[p].format(s=60 + (i + 3 * j) % 12), ln)
        M[f"e100f{j}"][p] = f"{dj}/{p}__s{{s}}.npz"
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
A0 = {"nm35": nm, "SL100f": sw(nm, [("r0", "e100f0"), ("r12", "e100f1"), ("r24", "e100f2"), ("vp12", "e100f3")]), "m30": {"r0": 1.0}, "m100f": {"e100f0": 1.0}}
A1 = {"S3f": S3f, "S3f_vp100": sw(S3f, [("vp12", "e100f0")])}
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": A0}, open(f"{O}/t2_d379/spec_a.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A1}, open(f"{O}/t2_d379/spec_b.json", "w"), indent=1)
json.dump({f"{O}/t2_d379/spec_a.json": list(A0), f"{O}/t2_d379/spec_b.json": list(A1)}, open(f"{O}/t2_d379/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: SL100f vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_a.json" nm35 SL100f > "$R/cluster_SL100f.txt" 2>&1; tail -12 "$R/cluster_SL100f.txt"
echo "== (informational) user-cluster CI: S3f_vp100 vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_b.json" S3f S3f_vp100 > "$R/cluster_S3f_vp100.txt" 2>&1; tail -12 "$R/cluster_S3f_vp100.txt"
