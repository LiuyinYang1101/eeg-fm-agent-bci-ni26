#!/usr/bin/env bash
# T2-D413c STEP 1 read (= T2-D413 Step 1; pre-registered 10:08 / 11:23 / D413c; written before any D413c Step 1 number): M6's tilt FBCNet + MeanFlow synthetic
# new-people windows (syn 0.5) vs the shipped line, PAIRED on fresh seeds 300-311 (both arms trained in the same local job).
# Blends as shipped (r 1/3, tag 3, tv8c). On M6's shape (S3f):
#   SWmf = fbc_tl15f -> fbc_mff (THE DECISION ARM)   SWctl = fbc_tl15f -> fbc_ctlf (the paired control; anchor)   S3f = M6 (sanity only)
# KILL (first, the panel's): FBCNet ALONE, kit protocol, test cells, causal SKλ (10 it, n0 3, λ 0.95), seeds 300-305: mean Δ (mf − ctl) < −0.005.
# PASS: SWmf vs SWctl typical (W25) >= +0.003 with >= 9 / 12, P04 >= −0.010, new >= 0; W50 + the user-cluster CI reported, never judged.
#   mem-guard bash scripts/local/t2_d413c_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d413c/read
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d413c-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d413c-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  blend --runs "$(one fbc_tl15mf_s300)" --protocols $p --out "$R/route_fbc_mff" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75_tl15_mf50tv8c --seeds $(seq -s, 300 311)
  blend --runs "$(one fbc_tl15_s300)" --protocols $p --out "$R/route_fbc_ctlf" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 300 311)
done
$PY - "$O" "$E" <<'PYEOF'
import json, os, sys
import numpy as np, pandas as pd
O, E = sys.argv[1], sys.argv[2]; R = f"{O}/t2_d413c/read"
sys.path.insert(0, "tracks/t2_bci/scripts")
from t2p_report import causal_sinkhorn
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M["fbc_mff"] = mem(300, R + "/route_fbc_mff/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15_mf50tv8cfb3")
M["fbc_ctlf"] = mem(300, R + "/route_fbc_ctlf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3")
# the kill check: FBCNet alone, kit test cells, SKλ, seeds 300-305
ix_ = pd.read_csv(f"{E}/index.csv"); strm, ordr, y, cell = ix_.stream.to_numpy(), ix_.order.to_numpy(), ix_.y.to_numpy(), ix_.cell.to_numpy()
def cba(f):
    z = np.load(f); ix = z["test_idx"]; P = causal_sinkhorn(z["prob_last"].astype(np.float64), (strm[ix], ordr[ix]), 10, 3, 0.95); pr = P.argmax(1)
    return np.mean([np.mean([np.mean(pr[(cell[ix] == c) & (y[ix] == k)] == k) for k in np.unique(y[ix][cell[ix] == c])]) for c in np.unique(cell[ix])])
kd = [cba(M["fbc_mff"]["kit"].format(s=300 + i)) - cba(M["fbc_ctlf"]["kit"].format(s=300 + i)) for i in range(6)]
kill = float(np.mean(kd)) < -0.005
print(f"== KILL CHECK (FBCNet alone, kit test, SKλ, seeds 300-305): mean Δ {np.mean(kd):+.4f} ({sum(v > 0 for v in kd)} / 6 up) -> {'KILLED' if kill else 'continue'}")
json.dump({"kill_delta": float(np.mean(kd)), "per_seed": [float(v) for v in kd], "killed": bool(kill)}, open(f"{R}/kill.json", "w"), indent=1)
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a, (o, n); a[n] = a.pop(o); return a
A = {"SWctl": sw(S3f, "fbc_tl15f", "fbc_ctlf"), "SWmf": sw(S3f, "fbc_tl15f", "fbc_mff"), "S3f": S3f}
json.dump({"n": 12, "anchor": "SWctl", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI + W50: SWmf vs SWctl (the decision contrast)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" SWctl SWmf > "$R/cluster_SWmf.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster_SWmf.txt"
$PY - "$R" <<'PYEOF'
import json, sys
import pandas as pd
R = sys.argv[1]; k = json.load(open(f"{R}/kill.json"))
d = pd.read_csv(f"{R}/rescreen_sk.csv"); d = d[(d.status == "ok") & (d.dec == "sk") & (d.arm == "SWmf")].iloc[0]
ok = d.typical >= 0.003 and d.typical_up >= 9 and d.P04 >= -0.010 and d.newT >= 0
v = "KILLED (kill check)" if k["killed"] else ("PASS -> the fresh-seed replicate T2-D413r (seeds 312-323 + a re-drawn generator)" if ok else "NOT MET")
print(f"== VERDICT (pre-registered bar): {v}  [typical {d.typical:+.4f} ({int(d.typical_up)} / 12), P04 {d.P04:+.4f}, new {d.newT:+.4f}; kill Δ {k['kill_delta']:+.4f}]")
PYEOF
