#!/usr/bin/env bash
# T2-D370 read (pre-registered 2026-11-06 07:01; written before any number): the audit's zero-GPU re-reads on nm35 of t2_d345r/full under SKλ.
#   mem-guard bash scripts/local/t2_d370_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d370
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; S = f"{O}/sofia_s1"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
st = "fm-st_large-mean__sz__n00__fmE30b64le0.0001lh{lh}wd0.05ll0.75ema0dp0mx0.4cgrc0.75{x}"
def stl(kdir, sdir, lh, x, xs, off):
    return {"offset": off, "kit": f"{S}/{kdir}/runs/kit__{st.format(lh=lh, x=x)}cptstlarge-d89b{xs}__s{{s}}.npz",
            "seen1": f"{S}/{sdir}/runs/seen1__{st.format(lh=lh, x=x)}cptpt-d111-with{xs}__s{{s}}.npz"}
M["lplh0"] = stl("ni26-t2p-d272-stl-lplh-kit-1537007", "ni26-t2p-d272-stl-lplh-seen1-1537008", "0.003", "lp10", "", 0)
M["lplh12"] = stl("ni26-t2p-d272r-stl-lplh-kit-1537009", "ni26-t2p-d272r-stl-lplh-seen1-1537010", "0.003", "lp10", "", 12)
M["sw2"] = stl("ni26-t2p-d255-stl-sw2-kit-1536154", "ni26-t2p-d255-stl-sw2-seen1-1536155", "0.001", "", "sw2", 0)
mor = "fm-mor_p1v2-mean__wz__n00__fmE12b64le0.001lh0.001wd0.05ll1ema0dp0{x}li16roled0.1"
M["mor_mx"] = {"offset": 0, "kit": f"{S}/ni26-t2p-t2d133-rolelora-1460136/runs/kit__{mor.format(x='mx0.4')}__s{{s}}.npz",
               "seen1": f"{S}/ni26-t2p-d142-mormix-1513629/runs/seen1__{mor.format(x='mx0.4')}__s{{s}}.npz"}
for k, x, kd, sd in (("mor_rc", "rc0.75", "ni26-t2p-d256-mor-kit-1536402", "ni26-t2p-d256-mor-seen1-1536403"),
                     ("mor_pl", "", "ni26-t2p-d256-mor-kit-1536402", "ni26-t2p-d256-mor-seen1-1536403"),
                     ("mor_rc5", "rc0.5", "ni26-t2p-d259-mor-rc5-kit-1536439", "ni26-t2p-d259-mor-rc5-seen1-1536441")):
    M[k] = {"offset": 0, "kit": f"{S}/{kd}/runs/kit__{mor.format(x=x)}__s{{s}}.npz", "seen1": f"{S}/{sd}/runs/seen1__{mor.format(x=x)}__s{{s}}.npz"}
for k in ("lplh0", "lplh12", "sw2", "mor_mx", "mor_rc", "mor_pl", "mor_rc5"):
    for p in ("kit", "seen1"):
        miss = [i for i in range(12) if not os.path.exists(M[k][p].format(s=M[k]["offset"] + i))]
        assert not miss, (k, p, miss)
def swap(a, pairs):
    a = dict(a)
    for old, new in pairs:
        a[new] = a.pop(old)
    return a
def add(a, m, w):
    return {**{k: v * (1 - w) for k, v in a.items()}, m: w}
slot = [k for k in ("r0", "r12", "r24", "vp12", "lp10") if k in nm]
ca = {k: v for k, v in nm.items() if k not in slot}
ca.update({k: sum(nm[x] for x in slot) / (len(slot) + 1) for k in slot + ["sw2"]})
arms = {"nm35": nm, "L2": swap(nm, [("r0", "lplh0"), ("r12", "lplh12")]), "Cs": swap(nm, [("r0", "sw2")]), "Ca": ca}
for m in ("mor_mx", "mor_rc", "mor_rc5", "mor_pl"):
    for w in (0.05, 0.10):
        arms[f"{m}_{int(w * 100):02d}"] = add(nm, m, w)
for k, a in arms.items():
    assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d370/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d370/spec.json": list(arms)}, open(f"{O}/t2_d370/sk.json", "w"))
print("arms:", list(arms))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in L2 Cs Ca mor_mx_05 mor_rc_05; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a > "$R/cluster_$a.txt" 2>&1; grep -E 'typical:|new \(' "$R/cluster_$a.txt"; done
