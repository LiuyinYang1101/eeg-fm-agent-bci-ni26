#!/usr/bin/env bash
# T2-D230r read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: T2-D230's candidate (FBCNet pooled windows re-coloured toward a random
# other training user at strength 0.5, rc5) on T2-D215's seeds (FBCNet 192-203), vs T2-D215's vr runs (c_vr; ShallowNet = D215's vr runs, seeds 156-167), at w4321.
# ADOPT iff mix > 0 AND new >= 0 (bar d141 reported beside, descriptive).
#   mem-guard bash scripts/local/t2_d230r_read.sh      (after scripts/local/t2_d215_read.sh, which writes the reference's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215/route_fbc_r1_pulxx_vp2_e180_vr/kit__*__s192.npz "$O"/t2_d215/route_sh_r1_pulx_vp2_e180_vr/seen1__*__s156.npz >/dev/null || { echo "run t2_d215_read.sh first"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d230r-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d230r-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  blend --runs "$R" --protocols $p --out "$O/t2_d230r/route_fbc_r1_pulxx_vp2_e180_vr_rc5" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc5 --seeds $(seq -s, 192 203)
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d230r/spec.json" --out "$O/t2_d230r/read.json" > "$O/t2_d230r/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d230r/read.json" <<'PYEOF' | tee "$O/t2_d230r/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1]))["arms"]["rc5"]
ok = r["mix"]["d"] > 0 and r["new"]["d"] >= 0
print("rc5 vs c_vr (D215's seeds): " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new"))
      + f" (mix {r['mix']['up']} / 12); mix > 0 {r['mix']['d'] > 0}; new >= 0 {r['new']['d'] >= 0}; bar d141 {'met' if r['met'] else 'not met'} (descriptive)")
print("VERDICT:", "ADOPT rc5 (FBCNet re-coloured at 0.5 joins the vp2 + vr line; upload 5's candidate is rebuilt)" if ok else "NOT ADOPTED (the replicate fails)")
PYEOF
