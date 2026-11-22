#!/usr/bin/env bash
# T2-D412 read (pre-registered 2026-11-09; this script written before any D412 number): M6's tilt FBCNet with CLASS-CONDITIONAL re-colouring
# (rc_cls 1, the GA panel's rcc) vs the shipped class-pooled rc 0.75, PAIRED on fresh seeds 276-287 (both arms trained in the same local job).
# Blends as shipped (r 1/3, tag 3, tv8c). On M6's shape (S3f):
#   SWcc = fbc_tl15f -> fbc_ccf (THE DECISION ARM)   SWctl = fbc_tl15f -> fbc_ctlf (the paired control; anchor)   S3f = M6 (sanity only)
# KILL (first, the panel's): FBCNet ALONE, kit protocol, test cells, causal SKλ (10 it, n0 3, λ 0.95), seeds 276-281: mean Δ (cc − ctl) < −0.005.
# PASS: SWcc vs SWctl typical (W25) >= +0.003 with >= 9 / 12, P04 >= −0.010, new >= 0; W50 + the user-cluster CI reported, never judged.
#   mem-guard bash scripts/local/t2_d412_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d412
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d412-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d412-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  blend --runs "$(one fbc_tl15cc_s276)" --protocols $p --out "$R/route_fbc_ccf" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75cc_tl15tv8c --seeds $(seq -s, 276 287)
  blend --runs "$(one fbc_tl15_s276)" --protocols $p --out "$R/route_fbc_ctlf" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 276 287)
done
$PY - "$O" "$E" <<'PYEOF'
import json, os, sys
import numpy as np, pandas as pd
O, E = sys.argv[1], sys.argv[2]; R = f"{O}/t2_d412"
sys.path.insert(0, "tracks/t2_bci/scripts")
from t2p_report import causal_sinkhorn
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M["fbc_ccf"] = mem(276, R + "/route_fbc_ccf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75cc_tl15tv8cfb3")
M["fbc_ctlf"] = mem(276, R + "/route_fbc_ctlf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3")
# the kill check: FBCNet alone, kit test cells, SKλ, seeds 276-281
ix_ = pd.read_csv(f"{E}/index.csv"); strm, ordr, y, cell = ix_.stream.to_numpy(), ix_.order.to_numpy(), ix_.y.to_numpy(), ix_.cell.to_numpy()
def cba(f):
    z = np.load(f); ix = z["test_idx"]; P = causal_sinkhorn(z["prob_last"].astype(np.float64), (strm[ix], ordr[ix]), 10, 3, 0.95); pr = P.argmax(1)
    return np.mean([np.mean([np.mean(pr[(cell[ix] == c) & (y[ix] == k)] == k) for k in np.unique(y[ix][cell[ix] == c])]) for c in np.unique(cell[ix])])
kd = [cba(M["fbc_ccf"]["kit"].format(s=276 + i)) - cba(M["fbc_ctlf"]["kit"].format(s=276 + i)) for i in range(6)]
kill = float(np.mean(kd)) < -0.005
print(f"== KILL CHECK (FBCNet alone, kit test, SKλ, seeds 276-281): mean Δ {np.mean(kd):+.4f} ({sum(v > 0 for v in kd)} / 6 up) -> {'KILLED' if kill else 'continue'}")
json.dump({"kill_delta": float(np.mean(kd)), "per_seed": [float(v) for v in kd], "killed": bool(kill)}, open(f"{R}/kill.json", "w"), indent=1)
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a, (o, n); a[n] = a.pop(o); return a
A = {"SWctl": sw(S3f, "fbc_tl15f", "fbc_ctlf"), "SWcc": sw(S3f, "fbc_tl15f", "fbc_ccf"), "S3f": S3f}
json.dump({"n": 12, "anchor": "SWctl", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI + W50: SWcc vs SWctl (the decision contrast)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" SWctl SWcc > "$R/cluster_SWcc.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster_SWcc.txt"
$PY - "$R" <<'PYEOF'
import json, sys
import pandas as pd
R = sys.argv[1]; k = json.load(open(f"{R}/kill.json"))
d = pd.read_csv(f"{R}/rescreen_sk.csv"); d = d[(d.status == "ok") & (d.dec == "sk") & (d.arm == "SWcc")].iloc[0]
ok = d.typical >= 0.003 and d.typical_up >= 9 and d.P04 >= -0.010 and d.newT >= 0
v = "KILLED (kill check)" if k["killed"] else ("PASS -> the fresh-seed replicate T2-D412r" if ok else "NOT MET")
print(f"== VERDICT (pre-registered bar): {v}  [typical {d.typical:+.4f} ({int(d.typical_up)} / 12), P04 {d.P04:+.4f}, new {d.newT:+.4f}; kill Δ {k['kill_delta']:+.4f}]")
PYEOF
