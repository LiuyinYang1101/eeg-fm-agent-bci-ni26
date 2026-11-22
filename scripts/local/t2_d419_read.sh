#!/usr/bin/env bash
# T2-D419 read (pre-registered 2026-11-09 before any D419 number): M7's pooled FBCNet at 90 pooled epochs (full cosine) vs the shipped 180, PAIRED on
# fresh seeds 324-335 (both arms in one local job), copies + tv8c unchanged. Blends as shipped (r 1/3, tag 3, tv8c). In M7's shape
# (outputs/t2_d408/spec_m7.json, arm M7): M7c = fbc_tv8c -> fbc_e180 (the ANCHOR) vs M7e90 = fbc_tv8c -> fbc_e90; causal SKλ (t2_d320_rescreen);
# W50 + user-cluster CI (t2_cluster_ci). BAR (R3, the manager's spec): typical >= +0.015 (over the 12 draws, >= 6 seeds) with W50 > 0 ADOPTS;
# typical <= 0 KILLS; in between -> one more dose (e60 if e90 > 0). Member alone (e90 vs e180, kit test, SKλ) reported.
#   mem-guard bash scripts/local/t2_d419_read.sh [e60]
# T2-D419b (2026-11-10, R3's extra dose): the optional argument is the dose (default e90 = D419 itself, outputs/t2_d419/); e60 -> outputs/t2_d419e60/,
# the same e180 control dir; the dose's in-between band is NOT ADOPTED (the extra dose is spent).
set -euo pipefail
cd "$(dirname "$0")/../.."
DOSE=${1:-e90}
case "$DOSE" in e90|e60) ;; *) echo "dose must be e90 or e60" >&2; exit 2;; esac
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d419${DOSE#e90}
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d419-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d419-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  blend --runs "$(one fbc_${DOSE}_s324)" --protocols $p --out "$R/route_fbc_$DOSE" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75${DOSE}tv8c --seeds $(seq -s, 324 335)
  blend --runs "$(one fbc_e180_s324)" --protocols $p --out "$R/route_fbc_e180" --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75tv8c --seeds $(seq -s, 324 335)
done
$PY - "$O" "$R" "$DOSE" <<'PYEOF'
import json, os, sys
O, R, D = sys.argv[1:4]
m7 = json.load(open(f"{O}/t2_d408/spec_m7.json")); M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
def mem(pat):
    m = {"offset": 324, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=324 + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
M[f"fbc_{D}"] = mem(R + f"/route_fbc_{D}/{{p}}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75{D}tv8cfb3")
M["fbc_e180"] = mem(R + "/route_fbc_e180/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3")
sw = lambda n: {(n if k == "fbc_tv8c" else k): v for k, v in A7.items()}
A = {"M7c": sw("fbc_e180"), f"M7{D}": sw(f"fbc_{D}")}
json.dump({"n": 12, "anchor": "M7c", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
Ms = {"n": 12, "anchor": "e180", "members": {f"fbc_{D}": M[f"fbc_{D}"], "fbc_e180": M["fbc_e180"]}, "arms": {"e180": {"fbc_e180": 1.0}, D: {f"fbc_{D}": 1.0}}}
json.dump(Ms, open(f"{R}/spec_member.json", "w"), indent=1)
json.dump({f"{R}/spec_member.json": ["e180", D]}, open(f"{R}/sk_member.json", "w"))
PYEOF
for t in "" _member; do
  mkdir -p "$R/out$t"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R/out$t" --jobs 4 \
    --specs "$R/spec$t.json" --sk-list "$R/sk$t.json" > "$R/run$t.log" 2>&1
done
$PY -c "
import pandas as pd
for t in ('_member', ''):
    d=pd.read_csv('$R/out'+t+'/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
    print('== ' + ('MEMBER ALONE ($DOSE vs e180)' if t else 'M7 SHAPE (M7$DOSE vs M7c)'))
    print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" M7c M7$DOSE > "$R/cluster.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster.txt"
$PY - "$R" "$DOSE" <<'PYEOF'
import re, sys
import pandas as pd
R, D = sys.argv[1:3]
d = pd.read_csv(f"{R}/out/rescreen_sk.csv"); r = d[(d.status == "ok") & (d.dec == "sk") & (d.arm == f"M7{D}")].iloc[0]
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", open(f"{R}/cluster.txt").read()).group(1))
mid = "IN BETWEEN -> one more dose (e60)" if D == "e90" else "NOT ADOPTED (T2-D419b: R3's extra dose is spent; the pooled-length line is CLOSED)"
v = "ADOPT (R3) -> the fresh-seed replicate T2-D419r before any build" if r.typical >= 0.015 and w50 > 0 else ("KILLED (R3: typical <= 0)" if r.typical <= 0 else mid)
print(f"== VERDICT (pre-registered R3 bar): {v}  [typical {r.typical:+.4f} ({int(r.typical_up)} / 12), W50 {w50:+.4f}, P04 {r.P04:+.4f}, newT {r.newT:+.4f}]")
PYEOF
