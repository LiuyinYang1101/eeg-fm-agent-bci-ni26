#!/usr/bin/env bash
# T2-D430 read (pre-registered in docs/DECISIONS_T2.md before any D430 number): panel P7's I1 = M1, STRANGER-SAFE per-user copies (Recipe.pu_kl 1:
# each copy step adds KL(p_pooled || p_copy) on the other training users' windows) on ALL THREE CNN families of M7, fresh seeds 348-359, kit + seen1.
# Inputs: outputs/sofia_s1/ni26-t2p-d430-{sh-kl,ifn-kl,fbc-ctl,fbc-kl}-5090 (scripts/local/t2_d430_local.sh) + D429's ni26-t2p-d429-{sh,ifn}-ctl-5090.
#   SANITY: the kl arms' POOLED probabilities (pu files' prob_last) equal their controls' (same pooled recipe): max |diff| printed per family.
#   STAGE 1 (members alone at the shipped r, causal SKλ): kl vs ctl per family. KILL if the M7-weighted mean member new-user Δ (newT) < -0.005.
#   STAGE 2 (M7's shape, outputs/t2_d408/spec_m7.json arm M7): anchor M7c3 = fbc / sh / ifn -> their ctl members at the shipped r (1/3, 1/2, 1/2);
#     arms M7kl_r0 (kl copies at the shipped r), M7kl_r67 (every family at r 2/3), M7kl_r100 (r 1). CROSS-FITTED r (R2): r* picked on KIT typical,
#     judged on SEEN1 cells, and r* picked on SEEN1, judged on KIT cells; the judged halves combined = the read (typical, draws up, W50, newT).
#     BAR (R3): cross-fitted typical >= +0.015 with W50 > 0 ADOPTS (-> T2-D430r fresh seeds, kit gate; t2p_final must pass X_other first);
#     <= 0 KILLS; otherwise one more dose (pu_kl 3). W50 decides only if the user types USER_LINES line 84.
#   mem-guard bash scripts/local/t2_d430_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d430
mkdir -p "$R"
SEEDS=$(seq -s, 348 359)
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
#        arm      dir                 model         align  recipe stem                         shipped r  tag
for spec in "fbc_ctl:d430-fbc-ctl:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75tv8c:0.3333:3" "fbc_kl:d430-fbc-kl:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75_kl1tv8c:0.3333:3" \
            "sh_ctl:d429-sh-ctl:shallow:ea:r1_pulx_vp2_e180_vrtv8c:0.5:2" "sh_kl:d430-sh-kl:shallow:ea:r1_pulx_vp2_e180_vr_kl1tv8c:0.5:2" \
            "ifn_ctl:d429-ifn-ctl:ifnet:ea:r1_pulx_vp2_e180_vrtv8c:0.5:2" "ifn_kl:d430-ifn-kl:ifnet:ea:r1_pulx_vp2_e180_vr_kl1tv8c:0.5:2"; do
  IFS=: read -r arm dn model al rec r tag <<< "$spec"
  d="$S/ni26-t2p-$dn-5090/runs"; [ -d "$d" ] || { echo "missing $d" >&2; exit 2; }
  for p in kit seen1; do
    blend --runs "$d" --protocols $p --out "$R/route_$arm" --model "$model" --align "$al" --recipe "$rec" --r "$r" --tag "$tag" --seeds "$SEEDS"
    if [ "${arm#*_}" = kl ]; then
      blend --runs "$d" --protocols $p --out "$R/route_$arm" --model "$model" --align "$al" --recipe "$rec" --r 0.6667 --tag 67 --seeds "$SEEDS"
      blend --runs "$d" --protocols $p --out "$R/route_$arm" --model "$model" --align "$al" --recipe "$rec" --r 1.0 --tag 100 --seeds "$SEEDS"
    fi
  done
done
$PY - "$O" "$R" "$S" <<'PYEOF'
import json, os, sys
import numpy as np
O, R, S = sys.argv[1:4]
fam = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75", "fbc_tv8c", "3"), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr", "sh_tv8c", "2"),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr", "ifn", "2")}
ctl_dir = {"fbc": "d430-fbc-ctl", "sh": "d429-sh-ctl", "ifn": "d429-ifn-ctl"}
for f, (model, al, rec, _, _) in fam.items():           # SANITY: the pooled nets are the controls'
    a = np.load(f"{S}/ni26-t2p-{ctl_dir[f]}-5090/runs/kit__{model}__{al}__n00__{rec}tv8cpu__s348.npz")["prob_last"]
    b = np.load(f"{S}/ni26-t2p-d430-{f}-kl-5090/runs/kit__{model}__{al}__n00__{rec}_kl1tv8cpu__s348.npz")["prob_last"]
    print(f"== SANITY {f}: pooled prob_last max |kl - ctl| = {np.abs(a - b).max():.2e}")
m7 = json.load(open(f"{O}/t2_d408/spec_m7.json")); M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
for f, (model, al, rec, key, tag) in fam.items():
    assert M[key]["kit"].endswith(f"fb{tag}__s{{s}}.npz"), M[key]
    for v, stem, tags in (("ctl", rec + "tv8c", (tag,)), ("kl", rec + "_kl1tv8c", (tag, "67", "100"))):
        for t in tags:
            name = f"{f}_{v}" + ("" if t == tag else f"_r{t}")
            M[name] = {"offset": 348, **{p: f"{R}/route_{f}_{v}/{p}__{model}__{al}__n00__{stem}fb{t}__s{{s}}.npz" for p in ("kit", "seen1")}}
            assert all(os.path.exists(M[name][p].format(s=348 + i)) for p in ("kit", "seen1") for i in range(12)), name
def sw(suffix_kl):
    out = {}
    for k, v in A7.items():
        f = next((f for f, t in fam.items() if t[3] == k), None)
        out[(f"{f}_kl{suffix_kl}" if suffix_kl is not None else f"{f}_ctl") if f else k] = v
    return out
A = {"M7c3": sw(None), "M7kl_r0": sw(""), "M7kl_r67": sw("_r67"), "M7kl_r100": sw("_r100")}
json.dump({"n": 12, "anchor": "M7c3", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
for f in fam:
    json.dump({"n": 12, "anchor": "ctl", "members": {f"{f}_ctl": M[f"{f}_ctl"], f"{f}_kl": M[f"{f}_kl"]},
               "arms": {"ctl": {f"{f}_ctl": 1.0}, "kl": {f"{f}_kl": 1.0}}}, open(f"{R}/spec_member_{f}.json", "w"), indent=1)
json.dump({f"{R}/spec_member_{f}.json": ["ctl", "kl"] for f in fam}, open(f"{R}/sk_member.json", "w"))
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
    d['spec']=d.spec.str.split('/').str[-1]
    print('== ' + ('STAGE 1: MEMBERS ALONE at the shipped r (kl vs ctl)' if t else 'M7 SHAPE, uncross-fitted (descriptive; every arm vs M7c3)'))
    print(d[['spec','arm','anchor','typical','typical_up','P04','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
OMP_NUM_THREADS=2 $PY - "$R" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
R = sys.argv[1]
sys.path.insert(0, "scripts/local")
import t2_cluster_ci as C
b = pd.read_csv(f"{R}/out_member/rescreen_sk.csv"); b = b[(b.status == "ok") & (b.dec == "sk")]
wts = {"fbc": 0.31428571428571433, "sh": 0.23571428571428574, "ifn": 0.1}
nt = {f: float(b[b.spec.str.endswith(f"spec_member_{f}.json")].newT.iloc[0]) for f in wts}
wn = sum(wts[f] * nt[f] for f in wts) / sum(wts.values())
print(f"== STAGE 1: member newT fbc {nt['fbc']:+.4f}, sh {nt['sh']:+.4f}, ifn {nt['ifn']:+.4f}; M7-weighted {wn:+.4f} -> {'KILLED (< -0.005)' if wn < -0.005 else 'pass'}")
J = json.loads(Path(f"{R}/spec.json").read_text())
idx = pd.read_csv(C.EXP / "index.csv")
key = ["proto", "split", "i", "cell", "subj", "ses"]
base = C.cell_rows(J, "M7c3", 12, idx)
D = {}
for arm in ("M7kl_r0", "M7kl_r67", "M7kl_r100"):
    x = base.merge(C.cell_rows(J, arm, 12, idx), on=key, suffixes=("_a", "_b")); x["d"] = x.ba_b - x.ba_a; D[arm] = x
typ = ["P05", "P08", "P09", "P10"]
def typical(x):
    t = x[x.subj.isin(typ)]; return t.groupby("i").d.mean()
def w50(x):
    ret = x[(x.subj == "P10") | ((x.subj == "P04") & (x.proto == "seen1"))]
    new = x[x.subj.isin(["P05", "P08", "P09"]) | ((x.subj == "P04") & (x.proto == "kit"))]
    return 0.5 * ret.d.mean() + 0.5 * new.d.mean(), new.d.mean()
for arm, x in D.items():
    print(f"   {arm}: typical kit {typical(x[x.proto == 'kit']).mean():+.4f} | seen1 {typical(x[x.proto == 'seen1']).mean():+.4f} | both {typical(x).mean():+.4f}")
pk = max(D, key=lambda a: typical(D[a][D[a].proto == "kit"]).mean())
ps = max(D, key=lambda a: typical(D[a][D[a].proto == "seen1"]).mean())
X = pd.concat([D[pk][D[pk].proto == "seen1"], D[ps][D[ps].proto == "kit"]])
td = typical(X); w, newd = w50(X)
print(f"== STAGE 2 CROSS-FITTED: r* picked on kit = {pk} (judged on seen1), on seen1 = {ps} (judged on kit)")
print(f"   typical {td.mean():+.4f} ({int((td > 0).sum())} / 12 draws up); W50 {w:+.4f}; new-user {newd:+.4f}; P10 {X[X.subj == 'P10'].d.mean():+.4f}; P04 {X[X.subj == 'P04'].d.mean():+.4f}")
if wn < -0.005:
    v = "KILLED at STAGE 1 (M7-weighted member new-user Δ < -0.005)"
elif td.mean() >= 0.015 and w > 0:
    v = "ADOPT (R3) -> T2-D430r on fresh seeds + kit gate; t2p_final must pass X_other before any build"
elif td.mean() <= 0:
    v = "KILLED (R3: cross-fitted typical <= 0)"
else:
    v = "IN BETWEEN (R3) -> one more dose (pu_kl 3)"
print(f"== VERDICT (pre-registered): {v}  [cross-fitted typical {td.mean():+.4f}, W50 {w:+.4f}, new {newd:+.4f}]")
PYEOF
