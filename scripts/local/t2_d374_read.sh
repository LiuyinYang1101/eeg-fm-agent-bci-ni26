#!/usr/bin/env bash
# T2-D374 read (pre-registered 2026-11-06 07:49; implementation + this script written 23:44, before any D374 number): TUNED DANN on FBCNet.
# Four arms on the rc75 line at seeds 192-203 (= nm35's fbc_tv8c, paired by design): marginal dm03 / dm1, class-conditional dc03 / dc1.
# ENGAGEMENT first (T2-D371): the [dann] log line's fresh linear user probe at the last epoch, mean over seeds; engaged iff <= 0.80
# (a fall >= 0.2 from 1.00); a non-engaged arm is NOT TESTED, whatever its score. Then, on nm35 under SKλ, 12 draws, anchor nm35:
#   SW_<t> = fbc_tv8c -> fbc_<t> (swap);  AD_<t> = FBCNet's weight split 1 : 1 between fbc_tv8c and fbc_<t> (added member).
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010 (either form); promote = a fresh-seed replicate.
#   mem-guard bash scripts/local/t2_d374_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d374
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d374-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d374-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}"; }
: > "$R/engagement.txt"
for t in dm03 dm1 dc03 dc1; do
  D=$(one fbc_${t}_s192)
  grep -h '\[dann\]' "$D"/*.log | sed -E 's/.*fresh linear user probe \(last epoch, 3-fold\) ([0-9.]+).*/\1/' | $PY -c "
import sys, numpy as np; v = np.array([float(x) for x in sys.stdin if x.strip()]); print('$t', len(v), 'fits; probe mean %.3f (min %.3f, max %.3f) -> %s' % (v.mean(), v.min(), v.max(), 'ENGAGED' if v.mean() <= 0.80 else 'NOT ENGAGED'))" | tee -a "$R/engagement.txt"
  for p in kit seen1; do blend --runs "$D/runs" --protocols $p --out "$R/route_fbc_$t" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_${t}tv8c --seeds $(seq -s, 192 203); done
done
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d374"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
assert M["fbc_tv8c"]["offset"] == 192
A = {"nm35": nm}
for t in ("dm03", "dm1", "dc03", "dc1"):
    m = {"offset": 192, **{p: f"{R}/route_fbc_{t}/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_{t}tv8cfb3__s{{s}}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=192 + i)) for p in ("kit", "seen1") for i in range(12)), t
    M[f"fbc_{t}"] = m
    sw = dict(nm); sw[f"fbc_{t}"] = sw.pop("fbc_tv8c"); A[f"SW_{t}"] = sw
    ad = dict(nm); w = ad["fbc_tv8c"]; ad["fbc_tv8c"] = w / 2; ad[f"fbc_{t}"] = w / 2; A[f"AD_{t}"] = ad
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== engagement"; cat "$R/engagement.txt"
