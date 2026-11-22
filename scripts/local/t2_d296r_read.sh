#!/usr/bin/env bash
# T2-D296r read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: tv8 (T2-D296) and tv8c (T2-D302) on FRESH seeds (T2-D247r's:
# FBCNet rc75 192-203, ShallowNet vr 156-167; stl_rcll 12+i, mor i from outputs/t2_d247r/spec.json). One training per seed wrote the plain files
# (= the control), tv8 and tv8c; anchor c_rcll = this run's plain CNN files + stl_rcll + mor, w4321 (paired). Each of tv8 / tv8c REPLICATES iff
# its mix > 0 AND new >= 0 (the replicate rule); evidence for the user's tv8c shipping decision after the Nov 5 reads.
#   mem-guard bash scripts/local/t2_d296r_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( "$S"/ni26-t2p-d296r-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d296r-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
mkdir -p "$O/t2_d296r"
for p in kit seen1; do
  R=$(one fbc-$p) || exit 2
  for x in "" tv8 tv8c; do
    r=r1_pulxx_vp2_e180_vr_rc75$x
    blend --runs "$R" --protocols $p --out "$O/t2_d296r/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 192 203)
  done
done
R=$(one sh) || exit 2
for p in kit seen1; do
  for x in "" tv8 tv8c; do
    r=r1_pulx_vp2_e180_vr$x
    blend --runs "$R" --protocols $p --out "$O/t2_d296r/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
  done
done
$PY -I - "$O/t2_d247r/spec.json" "$O/t2_d296r/spec.json" "$O/t2_d296r" <<'PYEOF' || exit 2
import json, sys
J = json.load(open(sys.argv[1])); D = sys.argv[3]
m = {k: J["members"][k] for k in ("stl_rcll", "mor")}
for x, tag in (("", "c"), ("tv8", "tv8"), ("tv8c", "tv8c")):
    for fam, rec, off, stem, sfx in (("fbc", "r1_pulxx_vp2_e180_vr_rc75", 192, "fbcnet_bboth__none", "fb3"), ("sh", "r1_pulx_vp2_e180_vr", 156, "shallow__ea", "fb2")):
        r = rec + x
        m[f"{fam}_{tag}"] = {"offset": off, **{p: f"{D}/route_{fam}_{r}/{p}__{stem}__n00__{r}{sfx}__s{{s}}.npz" for p in ("kit", "seen1")}}
w = lambda t: {f"fbc_{t}": 0.4, f"sh_{t}": 0.3, "stl_rcll": 0.2, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=m, arms={"c_rcll": w("c"), "tv8": w("tv8"), "tv8c": w("tv8c")})
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("spec: c_rcll = this run's plain CNN files + T2-D247r's stl_rcll (12+i) + mor; arms tv8, tv8c (replicate rule)")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d296r/spec.json" --out "$O/t2_d296r/read.json" > "$O/t2_d296r/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d296r/read.json" <<'PYEOF' | tee "$O/t2_d296r/verdict.txt"
import json, sys
J = json.load(open(sys.argv[1]))
A, rows = J["arms"], J["rows"]
kv = {(r["arm"], r["i"]): r["val"] for r in rows if r["proto"] == "kit"}
ok = {}
for a in ("tv8", "tv8c"):
    r = A[a]
    kill = sum(kv[(a, i)] - kv[("c_rcll", i)] for i in range(6)) / 6
    ok[a] = r["mix"]["d"] > 0 and r["new"]["d"] >= 0
    print(f"{a} vs c_rcll: " + ", ".join(f"{q} {r[q]['d']:+.4f}" for q in ("mix", "kit_test", "kit_val", "seen", "new"))
          + f" (mix {r['mix']['up']} / 12); kit val idx 0-5 {kill:+.4f}; bar d141 {'met' if r['met'] else 'not met'}"
          + f" -> {'REPLICATES' if ok[a] else 'does not replicate'} (mix > 0 and new >= 0)")
print("VERDICT:", ", ".join(f"{a} {'replicates' if ok[a] else 'does not replicate'}" for a in ok))
PYEOF
