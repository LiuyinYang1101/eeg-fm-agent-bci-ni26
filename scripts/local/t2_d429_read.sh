#!/usr/bin/env bash
# T2-D429 read (pre-registered in docs/DECISIONS_T2.md before any D429 number): FORGETTING EA (ni26.stream "eaf", gamma 0.98) on BOTH EA families of M7
# at once (ShallowNet + IFNet), FRESH seeds 348-359, kit + seen1 (scripts/local/t2_d429_local.sh: outputs/sofia_s1/ni26-t2p-d429-{sh,ifn}-{ctl,eaf}-5090).
# Blends as shipped (both families r 0.5, tag 2, tv8c; routing on the PLAIN fingerprints).
#   MEMBERS (descriptive): sh_eaf vs sh_ctl, ifn_eaf vs ifn_ctl (causal SKλ).
#   JUDGED (M7's shape, outputs/t2_d408/spec_m7.json arm M7): M7c2 = sh_tv8c -> sh_ctl AND ifn -> ifn_ctl (the ANCHOR) vs M7ef2 = the same with
#   sh_eaf + ifn_eaf; W50 + user-cluster CI. BAR (the standard bar): typical >= +0.003 with >= 9 / 12 draws up and P04 >= -0.010 -> CANDIDATE
#   (-> T2-D429r on fresh seeds, runtime, kit gate; a StreamAligner twin = a template change, the user's typed line); otherwise NOT ADOPTED, and the
#   forgetting-EA line closes for the season (no gamma dose).
#   mem-guard bash scripts/local/t2_d429_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d429
mkdir -p "$R"
SEEDS=$(seq -s, 348 359)
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
for spec in "sh_ctl:sh-ctl:shallow:ea" "sh_eaf:sh-eaf:shallow:eaf" "ifn_ctl:ifn-ctl:ifnet:ea" "ifn_eaf:ifn-eaf:ifnet:eaf"; do
  IFS=: read -r arm dn model al <<< "$spec"
  d="$S/ni26-t2p-d429-$dn-5090/runs"; [ -d "$d" ] || { echo "missing $d" >&2; exit 2; }
  for p in kit seen1; do
    blend --runs "$d" --protocols $p --out "$R/route_$arm" --model "$model" --align "$al" --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds "$SEEDS"
  done
done
$PY - "$O" "$R" <<'PYEOF'
import json, os, sys
O, R = sys.argv[1], sys.argv[2]
m7 = json.load(open(f"{O}/t2_d408/spec_m7.json")); M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
assert M["sh_tv8c"]["kit"].endswith("vrtv8cfb2__s{s}.npz") and M["ifn"]["kit"].endswith("vrtv8cfb2__s{s}.npz"), (M["sh_tv8c"], M["ifn"])
def mem(arm, model, al):
    m = {"offset": 348, **{p: f"{R}/route_{arm}/{p}__{model}__{al}__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=348 + i)) for p in ("kit", "seen1") for i in range(12)), arm
    return m
for arm, model, al in (("sh_ctl", "shallow", "ea"), ("sh_eaf", "shallow", "eaf"), ("ifn_ctl", "ifnet", "ea"), ("ifn_eaf", "ifnet", "eaf")):
    M[arm] = mem(arm, model, al)
specs = {}
for fam in ("sh", "ifn"):
    specs[fam] = {"n": 12, "anchor": "ctl", "members": {f"{fam}_ctl": M[f"{fam}_ctl"], f"{fam}_eaf": M[f"{fam}_eaf"]},
                  "arms": {"ctl": {f"{fam}_ctl": 1.0}, "eaf": {f"{fam}_eaf": 1.0}}}
    json.dump(specs[fam], open(f"{R}/spec_member_{fam}.json", "w"), indent=1)
json.dump({f"{R}/spec_member_{fam}.json": ["ctl", "eaf"] for fam in ("sh", "ifn")}, open(f"{R}/sk_member.json", "w"))
def sw(sh, ifn):
    out = {}
    for k, v in A7.items():
        out[sh if k == "sh_tv8c" else ifn if k == "ifn" else k] = v
    return out
A = {"M7c2": sw("sh_ctl", "ifn_ctl"), "M7ef2": sw("sh_eaf", "ifn_eaf")}
assert abs(sum(A["M7c2"].values()) - sum(A7.values())) < 1e-9
json.dump({"n": 12, "anchor": "M7c2", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
for t in _member ""; do
  mkdir -p "$R/out$t"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R/out$t" --jobs 4 \
    --specs "$R/spec$t*.json" --sk-list "$R/sk$t.json" > "$R/run$t.log" 2>&1
done
$PY -c "
import pandas as pd
for t in ('_member', ''):
    d=pd.read_csv('$R/out'+t+'/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
    print('== ' + ('MEMBERS ALONE (descriptive; eaf vs ctl per family)' if t else 'JUDGED: M7 SHAPE (M7ef2 vs M7c2)'))
    print(d[['spec','arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
$PY - "$R" <<'PYEOF'
import os, re, subprocess, sys
import pandas as pd
R = sys.argv[1]
out = subprocess.run([sys.executable, "-I", "scripts/local/t2_cluster_ci.py", f"{R}/spec.json", "M7c2", "M7ef2"], capture_output=True, text=True,
                     env=dict(os.environ, OMP_NUM_THREADS="2")).stdout
open(f"{R}/cluster.txt", "w").write(out); print(out[out.find("per user"):] if "per user" in out else out[-1500:])
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
c = pd.read_csv(f"{R}/out/rescreen_sk.csv"); c = c[(c.status == "ok") & (c.dec == "sk") & (c.arm == "M7ef2")].iloc[0]
if c.typical >= 0.003 and int(c.typical_up) >= 9 and c.P04 >= -0.010:
    v = "CANDIDATE -> T2-D429r on fresh seeds + runtime + kit gate; the build is a template change (the user's typed line)"
elif c.typical <= 0:
    v = "KILLED (typical <= 0): the forgetting-EA line closes for the season"
else:
    v = "NOT ADOPTED: the forgetting-EA line closes for the season (no gamma dose)"
print(f"== VERDICT (pre-registered): {v}  [M7 shape typical {c.typical:+.4f} ({int(c.typical_up)} / 12), W50 {w50:+.4f}, P04 {c.P04:+.4f}, newT {c.newT:+.4f}]")
PYEOF
