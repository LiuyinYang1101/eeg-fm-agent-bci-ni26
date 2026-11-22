#!/usr/bin/env bash
# T2-D378 read (pre-registered 2026-11-06 09:3x; written before any fresh number): the 3-swap stack on FRESH members, nm35 of t2_d345r/full,
# SKλ, 12 draws. Members: the tilt FBCNet seeds 204-215 (blended as T2-D358: r 1/3 tag 3, tv8c), the EA ST-L 36-47 (D378), the LP-FT + lh3
# ST-L 36-47 / 48-59 (T2-D376). Arms: S3f = all three swaps; the fresh singles (additivity) + the user-cluster CI.
#   mem-guard bash scripts/local/t2_d378_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d378
mkdir -p "$R"
for p in kit seen1; do
  $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$(pwd)/outputs/t2_d124b/fp_proteus.npy" --runs "$S/ni26-t2p-d378-fbc-tl15-1568960/runs" --protocols $p \
    --out "$R/route_fbc_tl15f_tv8c" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 204 215) > /dev/null
done
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; S = f"{O}/sofia_s1"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
def stl(kd, sd, off):
    m = {"offset": off}
    for p, d in (("kit", kd), ("seen1", sd)):
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (d, st)
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=off + i)) for i in range(12)), (d, p)
    return m
M["stl_eaf"] = stl("ni26-t2p-d378-stlea-kit-1568961", "ni26-t2p-d378-stlea-seen1-1568962", 36)
M["lplh36"] = stl("ni26-t2p-d376-lplh_s36-kit-1568928", "ni26-t2p-d376-lplh_s36-seen1-1568929", 36)
M["lplh48"] = stl("ni26-t2p-d376-lplh_s48-kit-1568930", "ni26-t2p-d376-lplh_s48-seen1-1568931", 48)
M["fbc_tl15f"] = {"offset": 204, **{p: f"{O}/t2_d378/route_fbc_tl15f_tv8c/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3__s{{s}}.npz" for p in ("kit", "seen1")}}
assert all(os.path.exists(M["fbc_tl15f"][p].format(s=204 + i)) for p in ("kit", "seen1") for i in range(12))
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: a[n] = a.pop(o)
    return a
P = {"TL": [("fbc_tv8c", "fbc_tl15f")], "EA": [("r24", "stl_eaf")], "L2": [("r0", "lplh36"), ("r12", "lplh48")]}
arms = {"nm35": nm, "S3f": sw(nm, P["TL"] + P["EA"] + P["L2"])}
arms.update({f"{k}f": sw(nm, v) for k, v in P.items()})
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d378/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d378/spec.json": list(arms)}, open(f"{O}/t2_d378/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')].set_index('arm')
print(d[['typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(float_format=lambda v: f'{v:+.4f}'))
s=d.loc['TLf','typical']+d.loc['EAf','typical']+d.loc['L2f','typical']; print(f'sum of fresh singles {s:+.4f}; S3f / sum {d.loc[\"S3f\",\"typical\"]/s:.2f}')"
echo "== user-cluster CI: S3f vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 S3f > "$R/cluster_S3f.txt" 2>&1; tail -12 "$R/cluster_S3f.txt"
