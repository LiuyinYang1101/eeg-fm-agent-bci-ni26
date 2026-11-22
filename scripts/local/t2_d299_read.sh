#!/usr/bin/env bash
# T2-D299 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: same-class cross-user input mixing in the pooled CNNs
# (backlog 73), FBCNet rc75 (seeds 204-215) + ShallowNet vr (168-179) with xmix 0.5, copies = the lines'. Anchor c_rcll = the LOCAL plain
# CNN runs of T2-D296 (deterministic: = T2-D293's local control) + stl_rcll + mor, at w4321. Candidate arm: xm5 (both); fxm5 / sxm5 descriptive.
# Adopt-candidate iff bar d141 AND new >= 0 AND not killed (kit val delta over indices 0-5 < -0.01); adoption needs the T2-D299r replicate.
#   mem-guard bash scripts/local/t2_d299_read.sh      (after scripts/local/t2_d299_local.sh; needs t2_d296's routes)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-d299-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d299-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
[ -f "$O/t2_d296/spec.json" ] || { echo "t2_d296's spec / routes are missing"; exit 2; }
mkdir -p "$O/t2_d299"
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  blend --runs "$R" --protocols $p --out "$O/t2_d299/route_fbc_r1_pulxx_vp2_e180_vr_rc75_xm5" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75_xm5 --seeds $(seq -s, 204 215)
done
R=$(one sh) || exit 2
for p in kit seen1; do
  blend --runs "$R" --protocols $p --out "$O/t2_d299/route_sh_r1_pulx_vp2_e180_vr_xm5" --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_xm5 --r 0.5 --tag 2 --seeds $(seq -s, 168 179)
done
$PY -I - "$O/t2_d296/spec.json" "$O/t2_d299/spec.json" "$O/t2_d299" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("fbc_rc75", "sh_vr", "stl_rcll", "mor")}
for name, fam, rec, off, stem, sfx in (("fbc_xm5", "fbc", "r1_pulxx_vp2_e180_vr_rc75_xm5", 204, "fbcnet_bboth__none", "fb3"),
                                       ("sh_xm5", "sh", "r1_pulx_vp2_e180_vr_xm5", 168, "shallow__ea", "fb2")):
    m[name] = {"offset": off, **{p: f"{D}/route_{fam}_{rec}/{p}__{stem}__n00__{rec}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda f, s: {f: 0.4, s: 0.3, "stl_rcll": 0.2, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=m,
            arms={"c_rcll": w("fbc_rc75", "sh_vr"), "xm5": w("fbc_xm5", "sh_xm5"), "fxm5": w("fbc_xm5", "sh_vr"), "sxm5": w("fbc_rc75", "sh_xm5")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_rcll = T2-D296's local plain CNN files + stl_rcll + mor; arms xm5 (candidate), fxm5 / sxm5 (descriptive)")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d299/spec.json" --out "$O/t2_d299/read.json" > "$O/t2_d299/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d299/read.json" <<'PYEOF' | tee "$O/t2_d299/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = False
for a in ("xm5", "fxm5", "sxm5"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    if a == "xm5":
        ok = bool(r["met"]) and r["new"]["d"] >= 0 and kill >= -0.01
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}" + (" KILLED" if kill < -0.01 else "")
          + (f"; bar d141 {'met' if r['met'] else 'not met'}; new >= 0 {r['new']['d'] >= 0} -> " + ("CANDIDATE" if ok else "no")
             if a == "xm5" else " (descriptive)"))
print("VERDICT:", "candidate xm5 -> the T2-D299r replicate (T2-D247r's seeds) before any adoption" if ok else "NOT ADOPTED (xm5 does not pass)")
PYEOF
