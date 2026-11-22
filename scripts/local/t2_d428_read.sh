#!/usr/bin/env bash
# T2-D428 read (pre-registered in docs/DECISIONS_T2.md before any D428 number): FORGETTING broadband EA (ni26.stream "eaf", gamma 0.98) for M7's ShallowNet.
# Inputs (scripts/local/t2_d424_local.sh): outputs/sofia_s1/ni26-t2p-d424-sh-ctl-5090/runs (ShallowNet-EA = the PAIRED control, same hardware) and
# ni26-t2p-d428-sh-eaf-5090/runs (the same with --aligns eaf), seeds 336-347, kit + seen1. Blends as shipped for ShallowNet (r 0.5, tag 2, tv8c;
# routing on the PLAIN fingerprints, as the package's router sees raw windows).
#   STAGE 1 (member alone, causal SKλ): sh_eaf vs sh_ctl; member typical <= -0.010 KILLS (no Stage 2 verdict).
#   STAGE 2 (M7's shape, outputs/t2_d408/spec_m7.json arm M7): M7c = sh_tv8c -> sh_ctl (the ANCHOR) vs M7ef = sh_tv8c -> sh_eaf; W50 + user-cluster CI.
#     BAR (the T2 goal's standard bar): typical >= +0.003 with >= 9 / 12 draws up and P04 >= -0.010 -> CANDIDATE (-> T2-D428r on fresh seeds,
#     runtime (≈ EA's), kit gate; the build is a template change = the user's typed line);
#     typical <= 0 KILLS; otherwise NOT ADOPTED and the line closes (no dose).
#   mem-guard bash scripts/local/t2_d428_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d428
mkdir -p "$R"
SEEDS=$(seq -s, 336 347)
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
for spec in "ctl:d424-sh-ctl:ea" "eaf:d428-sh-eaf:eaf"; do
  IFS=: read -r arm dn al <<< "$spec"
  d="$S/ni26-t2p-$dn-5090/runs"; [ -d "$d" ] || { echo "missing $d" >&2; exit 2; }
  for p in kit seen1; do
    blend --runs "$d" --protocols $p --out "$R/route_$arm" --model shallow --align "$al" --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds "$SEEDS"
  done
done
$PY - "$O" "$R" <<'PYEOF'
import json, os, sys
O, R = sys.argv[1], sys.argv[2]
m7 = json.load(open(f"{O}/t2_d408/spec_m7.json")); M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
assert M["sh_tv8c"]["kit"].endswith("vrtv8cfb2__s{s}.npz"), M["sh_tv8c"]
def mem(arm, al):
    m = {"offset": 336, **{p: f"{R}/route_{arm}/{p}__shallow__{al}__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=336 + i)) for p in ("kit", "seen1") for i in range(12)), arm
    return m
M["sh_ctl"], M["sh_eaf"] = mem("ctl", "ea"), mem("eaf", "eaf")
Ms = {"n": 12, "anchor": "ctl", "members": {k: M[k] for k in ("sh_ctl", "sh_eaf")}, "arms": {"ctl": {"sh_ctl": 1.0}, "eaf": {"sh_eaf": 1.0}}}
json.dump(Ms, open(f"{R}/spec_member.json", "w"), indent=1)
json.dump({f"{R}/spec_member.json": ["ctl", "eaf"]}, open(f"{R}/sk_member.json", "w"))
sw = lambda n: {(n if k == "sh_tv8c" else k): v for k, v in A7.items()}
A = {"M7c": sw("sh_ctl"), "M7ef": sw("sh_eaf")}
json.dump({"n": 12, "anchor": "M7c", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
for t in _member ""; do
  mkdir -p "$R/out$t"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R/out$t" --jobs 4 \
    --specs "$R/spec$t.json" --sk-list "$R/sk$t.json" > "$R/run$t.log" 2>&1
done
$PY -c "
import pandas as pd
for t in ('_member', ''):
    d=pd.read_csv('$R/out'+t+'/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
    print('== ' + ('STAGE 1: MEMBER ALONE (eaf vs ctl)' if t else 'STAGE 2: M7 SHAPE (M7ef vs M7c)'))
    print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
$PY - "$R" <<'PYEOF'
import os, re, subprocess, sys
import pandas as pd
R = sys.argv[1]
b = pd.read_csv(f"{R}/out_member/rescreen_sk.csv"); b = b[(b.status == "ok") & (b.dec == "sk")].set_index("arm")
mt = float(b.loc["eaf", "typical"])
print(f"== STAGE 1: member typical {mt:+.4f} ({int(b.loc['eaf', 'typical_up'])} / 12) -> {'pass' if mt > -0.010 else 'KILLED (<= -0.010)'}")
out = subprocess.run([sys.executable, "-I", "scripts/local/t2_cluster_ci.py", f"{R}/spec.json", "M7c", "M7ef"], capture_output=True, text=True,
                     env=dict(os.environ, OMP_NUM_THREADS="2")).stdout
open(f"{R}/cluster_eaf.txt", "w").write(out); print(out[out.find("per user"):] if "per user" in out else out[-1500:])
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
c = pd.read_csv(f"{R}/out/rescreen_sk.csv"); c = c[(c.status == "ok") & (c.dec == "sk") & (c.arm == "M7ef")].iloc[0]
if mt <= -0.010:
    v = "KILLED at STAGE 1 (member typical <= -0.010; the M7-shape numbers above are descriptive)"
elif c.typical >= 0.003 and int(c.typical_up) >= 9 and c.P04 >= -0.010:
    v = "CANDIDATE -> T2-D428r on fresh seeds + runtime + kit gate; the build is a template change (the user's typed line)"
elif c.typical <= 0:
    v = "KILLED (typical <= 0)"
else:
    v = "NOT ADOPTED: the forgetting-EA line closes (no dose)"
print(f"== VERDICT (pre-registered): {v}  [M7 shape typical {c.typical:+.4f} ({int(c.typical_up)} / 12), W50 {w50:+.4f}, P04 {c.P04:+.4f}, newT {c.newT:+.4f}]")
PYEOF
