#!/usr/bin/env bash
# T2-D390 read (pre-registered 2026-11-06; this script written before any D390 number): M6's tilt FBCNet with EA input alignment vs its shipped
# 'none', PAIRED on fresh seeds 252-263 (the same seeds, alignment only). Blends as shipped (r 1/3, tag 3, tv8c). On M6's shape (S3f):
#   SWea = fbc_tl15f -> fbc_eaf (THE DECISION ARM)   SWnone = fbc_tl15f -> fbc_nonef (the paired control); anchor SWnone.
# Bar: SWea vs SWnone typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI.
#   mem-guard bash scripts/local/t2_d390_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d390
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d390-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d390-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  blend --runs "$(one fbc_tl15ea_s252)" --protocols $p --out "$R/route_fbc_eaf" --model fbcnet_bboth --align ea --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 252 263)
  blend --runs "$(one fbc_tl15none_s252)" --protocols $p --out "$R/route_fbc_nonef" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 252 263)
done
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d390"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M["fbc_eaf"] = mem(252, R + "/route_fbc_eaf/{p}__fbcnet_bboth__ea__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3")
M["fbc_nonef"] = mem(252, R + "/route_fbc_nonef/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3")
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a, (o, n); a[n] = a.pop(o); return a
A = {"SWnone": sw(S3f, "fbc_tl15f", "fbc_nonef"), "SWea": sw(S3f, "fbc_tl15f", "fbc_eaf"), "S3f": S3f}
json.dump({"n": 12, "anchor": "SWnone", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: SWea vs SWnone (the decision contrast)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" SWnone SWea > "$R/cluster_SWea.txt" 2>&1; tail -8 "$R/cluster_SWea.txt"
