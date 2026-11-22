#!/usr/bin/env bash
# T2-D310 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the two near-miss STACKS on FRESH seeds only (T2-D247r's:
# FBCNet rc75 192-203, ShallowNet vr 156-167; stl_rcll 12+i, mor i from outputs/t2_d247r/spec.json), decoded by the SHIPPED SKλ (λ 0.95, n0 3,
# iters 10). Anchor A0 = the Nov 5 package's shape: MOR-free {FBCNet plain, ShallowNet plain (T2-D296r's routes), stl_rcll} at 4:3:2.
#   S1 = MOR-free + pv10 copies + tv8c (the T2-W9 template; copies retrained = weights)
#   S2 = w4321 WITH the MOR + pv10 copies + tv8 (the T2-W10 template + a tv8 bank; copies retrained)
# descriptive: tv8 / tv8c / pv10 / MOR alone on A0. Bar d141 + new >= 0, kill kit val idx 0-5 < -0.01 (scripts/local/t2_skl_pkg.py).
#   mem-guard bash scripts/local/t2_d310_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-d310-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d310-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
mkdir -p "$O/t2_d310"
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for x in "" tv8 tv8c; do
    r=r1_pulxx_vp2_e180_vr_rc75_pv10$x
    blend --runs "$R" --protocols $p --out "$O/t2_d310/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 192 203)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for x in "" tv8 tv8c; do
    r=r1_pulx_vp2_e180_vr_pv10$x
    blend --runs "$R" --protocols $p --out "$O/t2_d310/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
  done
done
$PY -I - "$O/t2_d296r/spec.json" "$O/t2_d310/spec.json" "$O/t2_d310" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = dict(J["members"])
for x in ("", "tv8", "tv8c"):
    for fam, rec, off, stem, sfx in (("fbc", "r1_pulxx_vp2_e180_vr_rc75_pv10", 192, "fbcnet_bboth__none", "fb3"), ("sh", "r1_pulx_vp2_e180_vr_pv10", 156, "shallow__ea", "fb2")):
        r = rec + x
        m[f"{fam}_pv{x}"] = {"offset": off, **{p: f"{D}/route_{fam}_{r}/{p}__{stem}__n00__{r}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
c3 = lambda f, s: {f: 0.4, s: 0.3, "stl_rcll": 0.2}
arms = {"A0": c3("fbc_c", "sh_c"), "S1": c3("fbc_pvtv8c", "sh_pvtv8c"), "S2": {**c3("fbc_pvtv8", "sh_pvtv8"), "mor": 0.1},
        "tv8": c3("fbc_tv8", "sh_tv8"), "tv8c": c3("fbc_tv8c", "sh_tv8c"), "pv10": c3("fbc_pv", "sh_pv"), "mor": {**c3("fbc_c", "sh_c"), "mor": 0.1}}
json.dump(dict(n=12, anchor="A0", bar="d141", members=m, arms=arms), open(sys.argv[2], "w"), indent=1)
print("spec: A0 = MOR-free plain CNNs (T2-D296r routes) + stl_rcll at 4:3:2; candidates S1 / S2; descriptive tv8 / tv8c / pv10 / mor")
PYEOF
$PY -u scripts/local/t2_skl_pkg.py --spec "$O/t2_d310/spec.json" --candidates S1,S2 | tee "$O/t2_d310/verdict.txt"
