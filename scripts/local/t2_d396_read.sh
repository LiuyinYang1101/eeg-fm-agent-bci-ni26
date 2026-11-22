#!/usr/bin/env bash
# T2-D396 read (pre-registered 2026-11-06 23:18; this script written before any D396 number): P1-style SUPERVISED pretraining of the CNN
# families (Scherer2015, t2p_cnn_p1.py) then the shipped fine-tune (--cnn-init), PAIRED with no-init controls on the same fresh seeds
# (FBCNet tilt 264-275, ShallowNet vr 228-239). Blends as shipped (fbc r 1/3 tag 3; sh r 0.5 tag 2), tv8c. On M6's shape (S3f):
#   P1both (THE DECISION ARM) = fbc_tl15f -> fbc_p1, sh_tv8c -> sh_p1   vs   CTLboth = the same with the paired controls (anchor);
#   singles P1f / P1s vs CTLf / CTLs; member alone m_* (descriptive + the member bar); PER USER (the T2-D103 / D105 P08-only pattern).
# Bar: member >= +0.02 typical (the better family, pretrained vs its control) AND P1both vs CTLboth typical >= +0.003 with >= 9 / 12,
#      P04 >= -0.010, new >= 0; a gain carried by P08 alone (P05 / P09 <= 0) is reported as the D105 pattern, not a pass.
#   mem-guard bash scripts/local/t2_d396_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d396
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d396-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d396-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
recof() { ls "$1" | grep -E "^kit__$2__$3__n00__.*tv8cpu__s$4\.npz$" | sed -E "s/^kit__$2__$3__n00__(.*)pu__s$4\.npz$/\1/" | sort -u; }
for arm in fbc_p1_s264 fbc_ctl_s264 sh_p1_s228 sh_ctl_s228; do
  RUNS=$(one $arm)
  case $arm in fbc*) m=fbcnet_bboth; al=none; s0=264; xr=();; *) m=shallow; al=ea; s0=228; xr=(--r 0.5 --tag 2);; esac
  rec=$(recof "$RUNS" $m $al $s0); [ "$(echo "$rec" | wc -l)" -eq 1 ] && [ -n "$rec" ] || { echo "recipe stem not unique for $arm: $rec" >&2; exit 2; }
  case $arm in *_p1_*) echo "$rec" | grep -q 'inip1' || { echo "$arm: no inip1 in $rec" >&2; exit 2; };; *) ! echo "$rec" | grep -q 'ini' || { echo "$arm: control has an init: $rec" >&2; exit 2; };; esac
  echo "$arm $rec" >> "$R/recipes.txt"
  for p in kit seen1; do blend --runs "$RUNS" --protocols $p --out "$R/route_$arm" --model $m --align $al --recipe "$rec" "${xr[@]}" --seeds $(seq -s, $s0 $((s0 + 11))); done
done
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d396"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
for arm, off in (("fbc_p1_s264", 264), ("fbc_ctl_s264", 264), ("sh_p1_s228", 228), ("sh_ctl_s228", 228)):
    stem = {}
    for p in ("kit", "seen1"):
        fs = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{R}/route_{arm}/{p}__*fb*__s*.npz")}); assert len(fs) == 1, (arm, p, fs)
        stem[p] = fs[0] + "__s{s}.npz"; assert all(os.path.exists(stem[p].format(s=off + i)) for i in range(12)), (arm, p)
    M[arm.rsplit("_s", 1)[0]] = {"offset": off, **stem}
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
A = {"CTLboth": sw(S3f, [("fbc_tl15f", "fbc_ctl"), ("sh_tv8c", "sh_ctl")]), "P1both": sw(S3f, [("fbc_tl15f", "fbc_p1"), ("sh_tv8c", "sh_p1")]),
     "CTLf": sw(S3f, [("fbc_tl15f", "fbc_ctl")]), "P1f": sw(S3f, [("fbc_tl15f", "fbc_p1")]), "CTLs": sw(S3f, [("sh_tv8c", "sh_ctl")]), "P1s": sw(S3f, [("sh_tv8c", "sh_p1")])}
B = {"m_fbc_ctl": {"fbc_ctl": 1.0}, "m_fbc_p1": {"fbc_p1": 1.0}, "m_sh_ctl": {"sh_ctl": 1.0}, "m_sh_p1": {"sh_p1": 1.0}}
json.dump({"n": 12, "anchor": "CTLboth", "members": M, "arms": A}, open(f"{R}/spec_pkg.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "m_fbc_ctl", "members": M, "arms": B}, open(f"{R}/spec_mem.json", "w"), indent=1)
json.dump({f"{R}/spec_pkg.json": list(A), f"{R}/spec_mem.json": list(B)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')].set_index('arm')
print(d[['anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(float_format=lambda v: f'{v:+.4f}'))
def lv(a):                                   # 23:3x fix: an anchor arm is not a rescreen row; its level = any row's level minus its Δ
    if a in d.index: return d.loc[a, 'typ_level']
    r = d[d['anchor'] == a]; return (r['typ_level'] - r['typical']).iloc[0] if len(r) else float('nan')
for fam in ('fbc', 'sh'):
    a, b = lv(f'm_{fam}_p1'), lv(f'm_{fam}_ctl'); print(f'member alone {fam}: p1 {a:.4f} vs ctl {b:.4f} -> {a - b:+.4f}')"  # 23:3x fix: the anchor arm is not a row (its level = level − Δ)
for pair in "CTLboth P1both" "CTLf P1f" "CTLs P1s"; do set -- $pair
  echo "== user-cluster CI: $2 vs $1 (per user)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_pkg.json" $1 $2 > "$R/cluster_$2.txt" 2>&1; grep -E 'P0[4589]|P10|typical:|returning' "$R/cluster_$2.txt" | tail -8
done
