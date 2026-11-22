#!/usr/bin/env bash
# T2-D315 read (pre-registered 2026-10-31 before any number; local CPU, zero GPU): GROUP DRO over the training users in the pooled CNNs
# (eta 0.003, both families), FRESH seeds (FBCNet rc75 216-227, ShallowNet vr 180-191; the anchor recipes re-trained on the same seeds,
# so the pair differs only in the loss). Route blends as T2-D310 (t2p_fpcalib; the per-user copies are plain CE in both arms), then
# scripts/local/t2_skl_pkg.py under the SHIPPED SKλ: A0 = {FBCNet, ShallowNet, ST-L} at 4:3:2 (MOR-free, the Nov 5 shape; ST-L = T2-D312's
# fresh k1 seeds 24+i); G = both families with gd3; descriptive Gf (FBCNet only), Gs (ShallowNet only).
# Bar: G vs A0 d141 (mix >= +0.005 with >= 9 / 12 up, kit test >= -0.005) AND new >= 0 AND no kill (kit val idx 0-5 < -0.01).
# Prediction (written now): G mix -0.003 to +0.005, new +0.000 to +0.006, seen -0.003 to +0.001; P(pass) ~ 0.08.
# Window prediction: the net fixes sit in NEW-user cells (new >= +0.003 while |seen| <= 0.002 if it works).
#   mem-guard bash scripts/local/t2_d315_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-d315-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d315-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
mkdir -p "$O/t2_d315"
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for r in r1_pulxx_vp2_e180_vr_rc75 r1_pulxx_vp2_e180_vr_rc75_gd3; do
    blend --runs "$R" --protocols $p --out "$O/t2_d315/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 216 227)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for r in r1_pulx_vp2_e180_vr r1_pulx_vp2_e180_vr_gd3; do
    blend --runs "$R" --protocols $p --out "$O/t2_d315/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 180 191)
  done
done
$PY -I - "$O/t2_d315" "$S" <<'PYEOF' || exit 2
import glob, json, sys
D, S = sys.argv[1], sys.argv[2]
stl = {p: glob.glob(f"{S}/ni26-t2p-d312-stl-{p}-a-*/runs")[0] for p in ("kit", "seen1")}
cpt = dict(kit="stlarge-d89b", seen1="pt-d111-with")
m = {"stl": {"offset": 24, **{p: f"{stl[p]}/{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cpt{cpt[p]}__s{{s}}.npz" for p in ("kit", "seen1")}}}
for fam, rec, off, stem, sfx in (("fbc", "r1_pulxx_vp2_e180_vr_rc75", 216, "fbcnet_bboth__none", "fb3"), ("sh", "r1_pulx_vp2_e180_vr", 180, "shallow__ea", "fb2")):
    for k, r in (("c", rec), ("g", rec + "_gd3")):
        m[f"{fam}_{k}"] = {"offset": off, **{p: f"{D}/route_{fam}_{r}/{p}__{stem}__n00__{r}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
c3 = lambda f, s: {f: 0.4, s: 0.3, "stl": 0.2}
arms = {"A0": c3("fbc_c", "sh_c"), "G": c3("fbc_g", "sh_g"), "Gf": c3("fbc_g", "sh_c"), "Gs": c3("fbc_c", "sh_g")}
json.dump(dict(n=12, anchor="A0", bar="d141", members=m, arms=arms), open(f"{D}/spec.json", "w"), indent=1)
print("spec: A0 = plain CNNs + ST-L (D312 k1) at 4:3:2; candidate G; descriptive Gf / Gs")
PYEOF
$PY -u scripts/local/t2_skl_pkg.py --spec "$O/t2_d315/spec.json" --candidates G | tee "$O/t2_d315/verdict.txt"
grep -h "\[gdro\]" "$S"/ni26-t2p-d315-*/arm*_shard*_*.log 2>/dev/null | sort | uniq -c | sort -rn | head -5
