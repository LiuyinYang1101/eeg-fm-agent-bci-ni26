#!/usr/bin/env bash
# The END-PHASE SHEET read (descriptive, zero GPU; the manager's 17:3x ask, 2026-11-06): every shot candidate vs S1's internal line on ONE paired
# spec (the same 12 draws, SKλ), so the sheet's deltas come from one read instead of a chain of reads.
#   S1     = T2-D310's S1 arm (pv10 copies + tv8c FBCNet / ShallowNet, stl_rcll at 4 : 3 : 2, normalised as the package does)  (the anchor; sealed 0.808)
#   M5     = nm35 (t2_d345r/full)          M5h = nm35 with lp10 -> stl_ea (the P2 shot list's m5h_swap_lp10)
#   M6     = S3f (T2-D378, fresh members)  M7  = SL100f (T2-D379, fresh e100 restarts)
#   mixes: M4like (the P2 shot list's m4_like) for the returning-user hedge context
#   mem-guard bash scripts/local/t2_endphase_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_endphase
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import json, sys
O = sys.argv[1]; R = f"{O}/t2_endphase"
M = {}
def take(f, keys=None):
    J = json.load(open(f"{O}/{f}"))["members"]
    for k, v in J.items():
        if keys is None or k in keys:
            assert k not in M or M[k] == v, f"member {k} differs between specs ({f})"
            M[k] = v
take("t2_d310/spec.json", ("fbc_pvtv8c", "sh_pvtv8c", "stl_rcll"))
take("t2_d345r/full/spec.json")
take("t2_p2_shotlist/spec.json", ("stl_ea", "fbc_pvtv8c", "sh_pvtv8c"))
take("t2_d378/spec.json", ("stl_eaf", "lplh36", "lplh48", "fbc_tl15f"))
take("t2_d379/spec_a.json", ("e100f0", "e100f1", "e100f2", "e100f3"))
S1 = json.load(open(f"{O}/t2_d310/spec.json"))["arms"]["S1"]
assert abs(sum(S1.values()) - 0.9) < 1e-9, S1        # T2-D310 wrote S1 as the MOR-free 4 : 3 : 2 (sum 0.9); the package normalises it
S1 = {k: v / 0.9 for k, v in S1.items()}            # (its fm_weight is 0.2 / 0.9 = 0.2222, row 420): the shipped line = 4/9 : 3/9 : 2/9
nm = json.load(open(f"{O}/t2_d345r/full/spec.json"))["arms"]["nm35"]
sl = json.load(open(f"{O}/t2_p2_shotlist/spec.json"))["arms"]
d378 = json.load(open(f"{O}/t2_d378/spec.json"))["arms"]["S3f"]
d379 = json.load(open(f"{O}/t2_d379/spec_a.json"))["arms"]["SL100f"]
assert sl["nm35"] == nm, "the shot list's nm35 must be M5's"
A = {"S1": S1, "M5": nm, "M5h": sl["m5h_swap_lp10"], "M6": d378, "M7": d379, "M4like": sl["m4_like"]}
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
A["M6pv"] = sw(d378, [("fbc_tl15f", "fbc_pvtv8c"), ("sh_tv8c", "sh_pvtv8c")])    # T2-D387 (pre-registered before these two arms' numbers)
A["M6pvs"] = sw(d378, [("sh_tv8c", "sh_pvtv8c")])
for a in A.values():
    assert abs(sum(a.values()) - 1) < 1e-9 and all(k in M for k in a), a
json.dump({"n": 12, "anchor": "S1", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in M5 M5h M6 M7 M4like M6pv M6pvs; do
  echo "== user-cluster CI: $a vs S1"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" S1 $a > "$R/cluster_$a.txt" 2>&1; grep -E 'P0[4589]|P10|typical:|returning' "$R/cluster_$a.txt" | tail -8
done
for a in M6pv M6pvs; do   # T2-D387: vs M6 too
  echo "== user-cluster CI: $a vs M6"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" M6 $a > "$R/cluster_${a}_M6.txt" 2>&1; grep -E 'typical:|returning|P04 alone' "$R/cluster_${a}_M6.txt" | tail -3
done
