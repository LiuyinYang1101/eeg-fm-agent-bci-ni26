#!/usr/bin/env bash
# T2-D388 read (pre-registered 2026-11-06; this script written before any D388 number): the fresh-seed check of M6pv's pv10 copies on M6's nets.
# Members (blends as shipped, tv8c): the tilt FBCNet 240-251 with pv10 copies (fbc_pvf) and with plain copies (fbc_plf, the paired control);
# ShallowNet 216-227 with pv10 copies (sh_pvf) and plain copies (sh_plf). On the end-phase paired spec (outputs/t2_endphase: S1, M6 = S3f):
#   M6pv_f = S3f with fbc_tl15f -> fbc_pvf, sh_tv8c -> sh_pvf          M6_f = S3f with fbc_tl15f -> fbc_plf, sh_tv8c -> sh_plf (paired control)
# Bar: (i) M6pv_f vs S1 typical >= +0.015 with >= 9 / 12 AND P04 >= -0.010 (T2-D387's rule, on fresh seeds);
#      (ii) M6pv_f vs M6_f (same pooled nets, copies only): returning >= +0.003 AND P04 >= +0.003 AND typical >= -0.005 (the hedge is real).
#   mem-guard bash scripts/local/t2_d388_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d388
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d388-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d388-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  blend --runs "$(one fbc_tl15pv10_s240)" --protocols $p --out "$R/route_fbc_pvf" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_tl15_pv10tv8c --seeds $(seq -s, 240 251)
  blend --runs "$(one fbc_tl15_s240)" --protocols $p --out "$R/route_fbc_plf" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_tl15tv8c --seeds $(seq -s, 240 251)
  blend --runs "$(one sh_pv10_s216)" --protocols $p --out "$R/route_sh_pvf" --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_pv10tv8c --r 0.5 --tag 2 --seeds $(seq -s, 216 227)
  blend --runs "$(one sh_vr_s216)" --protocols $p --out "$R/route_sh_plf" --model shallow --align ea --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds $(seq -s, 216 227)
done
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d388"
base = json.load(open(f"{O}/t2_endphase/spec.json")); M = dict(base["members"]); A0 = base["arms"]
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M["fbc_pvf"] = mem(240, R + "/route_fbc_pvf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15_pv10tv8cfb3")
M["fbc_plf"] = mem(240, R + "/route_fbc_plf/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_tl15tv8cfb3")
M["sh_pvf"] = mem(216, R + "/route_sh_pvf/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_pv10tv8cfb2")
M["sh_plf"] = mem(216, R + "/route_sh_plf/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2")
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
M6 = A0["M6"]
A = {"S1": A0["S1"], "M6": M6, "M6pv_f": sw(M6, [("fbc_tl15f", "fbc_pvf"), ("sh_tv8c", "sh_pvf")]), "M6_f": sw(M6, [("fbc_tl15f", "fbc_plf"), ("sh_tv8c", "sh_plf")])}
json.dump({"n": 12, "anchor": "S1", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in M6pv_f M6_f; do echo "== $a vs S1"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S1 $a > "$R/cluster_$a.txt" 2>&1; grep -E 'typical:|returning' "$R/cluster_$a.txt"; done
echo "== (ii) M6pv_f vs M6_f (copies only)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" M6_f M6pv_f > "$R/cluster_pv_vs_pl.txt" 2>&1; grep -E 'typical:|returning|P04 alone' "$R/cluster_pv_vs_pl.txt"
