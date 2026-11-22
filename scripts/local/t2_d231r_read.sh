#!/usr/bin/env bash
# T2-D231r read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: T2-D231's candidate (FBCNet pooled windows re-coloured toward a random
# other training user at strength 0.75, rc75) on T2-D215's seeds (FBCNet 192-203), vs T2-D215's vr runs (c_vr; ShallowNet = D215's vr runs, seeds 156-167), at w4321;
# T2-D230r's rc5 on the same seeds beside. rc75 REPLACES rc5 iff mix > 0 AND new >= 0 AND its mix > rc5's on these seeds (T2-D230r, +0.0063).
#   mem-guard bash scripts/local/t2_d231r_read.sh      (after t2_d215_read.sh and t2_d230r_read.sh, which write the reference's and rc5's route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
ls "$O"/t2_d215/route_fbc_r1_pulxx_vp2_e180_vr/kit__*__s192.npz "$O"/t2_d230r/route_fbc_r1_pulxx_vp2_e180_vr_rc5/seen1__*__s192.npz >/dev/null || { echo "run t2_d215_read.sh and t2_d230r_read.sh first"; exit 2; }
one() { local d=( "$S"/ni26-t2p-d231r-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d231r-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  blend --runs "$R" --protocols $p --out "$O/t2_d231r/route_fbc_r1_pulxx_vp2_e180_vr_rc75" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds $(seq -s, 192 203)
done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d231r/spec.json" --out "$O/t2_d231r/read.json" > "$O/t2_d231r/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d231r/read.json" <<'PYEOF' | tee "$O/t2_d231r/verdict.txt"
import json, sys
A = json.load(open(sys.argv[1]))["arms"]
r, ref = A["rc75"], A["rc5"]["mix"]["d"]
ok = r["mix"]["d"] > 0 and r["new"]["d"] >= 0 and r["mix"]["d"] > ref
for a in ("rc75", "rc5"):
    q_ = A[a]
    print(f"{a} vs c_vr (D215's seeds): " + ", ".join(f"{q} {q_[q]['d']:+.4f}" for q in ("mix", "kit_test", "seen", "new"))
          + f" (mix {q_['mix']['up']} / 12); bar d141 {'met' if q_['met'] else 'not met'}" + (" (T2-D230r, descriptive)" if a == "rc5" else ""))
print(f"rc75: mix > 0 {r['mix']['d'] > 0}; new >= 0 {r['new']['d'] >= 0}; mix > rc5's {ref:+.4f} {r['mix']['d'] > ref}")
print("VERDICT:", "rc75 REPLACES rc5 (the re-tune's FBCNet is re-coloured at 0.75)" if ok else "rc5 STAYS (rc75 does not replicate above rc5)")
PYEOF
