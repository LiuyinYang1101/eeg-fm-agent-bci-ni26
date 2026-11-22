#!/usr/bin/env bash
# T2-D244 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the two FM members (ST-L, MOR) enter the package blend at
# temperature T (p^(1/T) renormalised; t2p_pkgshape member field `temp`): t15 = 1.5, t2 = 2.0, vs c_rc75 {fbc_rc75 0.4, sh_vr 0.3, stl 0.2, mor 0.1}
# on the files on disk (FBCNet T2-D231 seeds 204-215, ShallowNet T2-D215r 168-179, ST-L / MOR 0-11). Adopt-CANDIDATE iff an arm meets bar d141
# AND new >= 0 (both -> the higher mix); a candidate is then read at once in T2-D244r (FBCNet T2-D231r 192-203, ShallowNet T2-D215 156-167, the same
# FM files): adopted iff mix > 0 AND new >= 0 there.
#   mem-guard bash scripts/local/t2_d244_read.sh      (the specs outputs/t2_d244/spec{,_r}.json are written at pre-registration)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge/t2_d244
sha256sum "$O/spec.json" "$O/spec_r.json" | cut -c1-12
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/spec.json" --out "$O/read.json" > "$O/read.txt" 2>&1; echo "rc $?"
best=$($PY -I - "$O/read.json" <<'PYEOF' | tee "$O/verdict.txt" | tail -1
import json, sys
r = json.load(open(sys.argv[1]))
ok = {}
for a in ("t15", "t2"):
    c = r["arms"][f"c_{a}"]
    ok[a] = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005 and c["new"]["d"] >= 0
    print(f"{a}: c_{a} vs c_rc75: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, "
          f"new {c['new']['d']:+.4f} -> {'CANDIDATE' if ok[a] else 'no'}")
c = [a for a in ok if ok[a]]
print(max(c, key=lambda a: r["arms"][f"c_{a}"]["mix"]["d"]) if c else "none")
PYEOF
)
cat "$O/verdict.txt" | head -2
if [ "$best" = none ]; then echo "VERDICT: NOT ADOPTED (the FM members stay at T 1)"; exit 0; fi
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/spec_r.json" --out "$O/read_r.json" > "$O/read_r.txt" 2>&1; echo "rc_r $?"
$PY -I - "$O/read_r.json" "$best" <<'PYEOF' | tee -a "$O/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); a = sys.argv[2]; c = r["arms"][f"c_{a}"]
ok = c["mix"]["d"] > 0 and c["new"]["d"] >= 0
print(f"T2-D244r {a}: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}")
print("VERDICT:", f"ADOPT {a} (template + rebuild rows follow)" if ok else f"NOT ADOPTED ({a} failed T2-D244r)")
PYEOF
