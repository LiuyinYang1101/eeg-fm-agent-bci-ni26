#!/usr/bin/env bash
# T2-D383 read (pre-registered 2026-11-06 13:43; this script written before any D383 number and before any T2-D379 number): the next stack on M6.
#   STK2 (THE DECISION ARM, anchor S3f) = S3f with sh_tv8c -> the FRESH e360 ShallowNet (seeds 204-215, sofia 1570114; blend r 0.5 tag 2, tv8c)
#        AND vp12 -> T2-D379's e100 restart 0 (draw i: seed 60 + i; sofia 1569198 / 99). Singles SHf / VPe descriptive.
# Bar: STK2 vs S3f typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI reported.
#   mem-guard bash scripts/local/t2_d383_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d383
mkdir -p "$R"
d=( "$S"/ni26-t2p-d383-sh_e360_s204-1570114/ ); [ -d "${d[0]}runs" ] || { echo "need the d383 ShallowNet dir" >&2; exit 2; }
for p in kit seen1; do
  $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" --runs "${d[0]}runs" --protocols $p --out "$R/route_sh_e360f" \
    --model shallow --align ea --recipe r1_pulx_vp2_e360_vrtv8c --r 0.5 --tag 2 --seeds $(seq -s, 204 215) > /dev/null
done
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d383"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert {"sh_tv8c", "vp12"} <= set(S3f), sorted(S3f)
M["sh_e360f"] = {"offset": 204, **{p: f"{R}/route_sh_e360f/{p}__shallow__ea__n00__r1_pulx_vp2_e360_vrtv8cfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
assert all(os.path.exists(M["sh_e360f"][p].format(s=204 + i)) for p in ("kit", "seen1") for i in range(12))
src = {}
for p, job in (("kit", 1569198), ("seen1", 1569199)):
    ds = glob.glob(f"{S}/ni26-t2p-*-{job}"); assert len(ds) == 1, (p, job, ds)
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (p, st)
    src[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(src[p].format(s=60 + i)) for i in range(12)), (p, "D379 seeds 60-71 incomplete")
M["e100f0"] = {"offset": 60, **src}          # draw i: seed 60 + i = T2-D379's restart 0
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
A = {"S3f": S3f, "STK2": sw(S3f, [("sh_tv8c", "sh_e360f"), ("vp12", "e100f0")]), "SHf": sw(S3f, [("sh_tv8c", "sh_e360f")]), "VPe": sw(S3f, [("vp12", "e100f0")])}
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: STK2 vs S3f (the decision arm)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S3f STK2 > "$R/cluster_STK2.txt" 2>&1; tail -8 "$R/cluster_STK2.txt"
