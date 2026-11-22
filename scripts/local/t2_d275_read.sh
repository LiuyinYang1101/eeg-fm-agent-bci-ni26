#!/usr/bin/env bash
# T2-D275 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md (the panel's T2 item 3, user-approved 2026-10-26), after
# `s1 fetch ni26` of ni26-t2p-d275-fbc-crop-<kit|seen1>-<jobid>: two EXTRA FBCNet members of the adopted FBCNet recipe (r1_pulxx_vp2_e180_vr_rc75,
# pooled + per-user copies) trained / predicting on window samples 0-240 (1-3 s after the cue) and 240-480 (3-5 s), seeds 204-215. Routes each
# crop member's files with t2p_fpcalib blend (r 1/3, tag 3: as fbc_rc75), writes outputs/t2_d275/spec.json from T2-D253's spec (fbc_rc75 204,
# sh_vr 168, stl_rcll 0, mor 0; offsets asserted) + fbc_cr13 / fbc_cr35 (204), then t2p_pkgshape: anchor c_rcll = w4321; arm c_crop = {fbc_rc75 0.3,
# fbc_cr13 0.05, fbc_cr35 0.05, sh_vr 0.3, stl_rcll 0.2, mor 0.1}; each crop member alone beside.
# Kill iff either crop member alone has mean kit test < 0.40 (seeds 204-215); adopt-CANDIDATE iff c_crop vs c_rcll meets bar d141 AND new >= 0;
# adoption needs the T2-D275r fresh-seed replicate (seeds 192-203, T2-D247r's set).
#   mem-guard bash scripts/local/t2_d275_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
R=$O/t2_d275
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d275-fbc-crop-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d275 $p: ${d[*]}"; exit 2; }
  for c in cr0-240 cr240-480; do
    blend --runs "${d[0]}runs" --protocols $p --out "$R/route_fbc_$c" --model fbcnet_bboth --recipe r1_pulxx_vp2_e180_vr_rc75$c --seeds $(seq -s, 204 215) > "$R/blend_${p}_$c.log" 2>&1 \
      || { echo "blend $p $c failed (see $R/blend_${p}_$c.log)"; exit 3; }
  done
done
$PY -I - "$O/t2_d253/spec.json" "$R" <<'PYEOF' || exit 5
import hashlib, json, os, sys
import numpy as np
s, out = json.load(open(sys.argv[1])), sys.argv[2]
offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs
for k, c in (("fbc_cr13", "cr0-240"), ("fbc_cr35", "cr240-480")):
    mem[k] = dict(offset=204, **{p: f"{out}/route_fbc_{c}/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75{c}fb3__s{{s}}.npz" for p in ("kit", "seen1")})
    for p in ("kit", "seen1"):
        for i in range(12):
            a, b = np.load(mem["fbc_rc75"][p].format(s=204 + i)), np.load(mem[k][p].format(s=204 + i))
            assert np.array_equal(a["test_idx"], b["test_idx"]) and float(np.abs(a["prob_last"] - b["prob_last"]).max()) > 1e-4, (k, p, i)
base = {"sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}
spec = dict(n=12, anchor="c_rcll", bar="d141", members=mem,
            arms={"c_rcll": dict(base, fbc_rc75=0.4), "c_crop": dict(base, fbc_rc75=0.3, fbc_cr13=0.05, fbc_cr35=0.05),
                  "fbc_rc75_alone": {"fbc_rc75": 1.0}, "fbc_cr13_alone": {"fbc_cr13": 1.0}, "fbc_cr35_alone": {"fbc_cr35": 1.0}})
sp = os.path.join(out, "spec.json")
open(sp, "w").write(json.dumps(spec, indent=1) + "\n")
print(f"spec {sp} sha256 {hashlib.sha256(open(sp, 'rb').read()).hexdigest()[:12]}")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$R/spec.json" --out "$R/read.json" > "$R/read.txt" 2>&1; echo "rc $?"
$PY -I - "$R/read.json" <<'PYEOF' | tee "$R/verdict.txt"
import json, sys
import numpy as np
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
kt = {a: float(np.mean([rows[("kit", i, f"{a}_alone")]["test"] for i in range(12)])) for a in ("fbc_rc75", "fbc_cr13", "fbc_cr35")}
kill = min(kt["fbc_cr13"], kt["fbc_cr35"]) < 0.40
c = r["arms"]["c_crop"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = not kill and met and c["new"]["d"] >= 0
print("members alone, kit test (mean of 12): " + ", ".join(f"{a} {v:.4f}" for a, v in kt.items()) + f" -> {'KILL' if kill else 'continue'} (< 0.40)")
print(f"c_crop vs c_rcll: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, "
      f"kit val {c['kit_val']['d']:+.4f}, seen1 val {c['seen1_val']['d']:+.4f}; bar d141 {'met' if met else 'not met'}")
print("VERDICT:", "KILLED" if kill else ("CANDIDATE -> the T2-D275r fresh-seed replicate (seeds 192-203) before any adoption" if ok else "NOT ADOPTED (FBCNet stays one full-window member)"))
PYEOF
