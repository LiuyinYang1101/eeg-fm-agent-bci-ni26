#!/usr/bin/env bash
# T2-D256 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d256-mor-<kit|seen1>-<jobid>:
# re-colouring 0.75 on the MOR member. Each job holds two arms on seeds 0-11: ctl = the package's MOR fine-tune (T2-D130 / D133 role LoRA r16, wz,
# 12 epochs) on today's code, rc = the same + --fm-rc 0.75 (tag rc0.75). Links the 48 result files into outputs/t2_d256/mor_{ctl,rc}/, writes
# outputs/t2_d256/spec.json from T2-D253's spec (members fbc_rc75 (204-215), sh_vr (168-179), stl_rcll, mor (the 10-15 files); offsets asserted;
# sha256 printed) + mor_ctl + mor_rc, then t2p_pkgshape with anchor c_ctl = {fbc_rc75 0.4, sh_vr 0.3, stl_rcll 0.2, mor_ctl 0.1}.
# Kill iff MOR alone (rc - ctl) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_rc vs c_ctl meets bar d141 (mix >= +0.005 with >= 9 / 12 up
# AND kit test >= -0.005) AND new >= 0; adoption needs T2-D256r. Descriptive: c_old (the 10-15 MOR files) vs c_ctl = the code-drift check.
#   mem-guard bash scripts/local/t2_d256_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
K="fmE12b64le0.001lh0.001wd0.05ll1ema0dp0"
mkdir -p "$O/t2_d256/mor_ctl" "$O/t2_d256/mor_rc"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d256-mor-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for mor $p: ${d[*]}"; exit 2; }
  for a in ctl rc; do
    [ $a = ctl ] && t="${K}li16roled0.1" || t="${K}rc0.75li16roled0.1"
    n=0
    for f in "${d[0]}"runs/${p}__fm-mor_p1v2-mean__wz__n00__${t}__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d256/mor_$a/$(basename "$f")" && n=$((n + 1)); done
    [ "$n" -eq 12 ] || { echo "mor $a $p: $n / 12 result files with $t"; exit 3; }
  done
done
[ "$(ls "$O/t2_d256/mor_ctl" | wc -l)" -eq 24 ] && [ "$(ls "$O/t2_d256/mor_rc" | wc -l)" -eq 24 ] || { echo "need 24 + 24 files"; exit 4; }
$PY -I - "$O/t2_d253/spec.json" "$O/t2_d256" <<'PYEOF' || exit 5
import glob, hashlib, json, os, sys
import numpy as np
s, out = json.load(open(sys.argv[1])), sys.argv[2]
offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
mem = {m: s["members"][m] for m in offs}
assert {m: d["offset"] for m, d in mem.items()} == offs, {m: d["offset"] for m, d in mem.items()}
for a in ("ctl", "rc"):
    pat = {}
    for p in ("kit", "seen1"):
        f = sorted(glob.glob(f"{out}/mor_{a}/{p}__*__s0.npz"))
        assert len(f) == 1, f
        pat[p] = f[0][:-len("0.npz")] + "{s}.npz"
        x, y = np.load(mem["mor"][p].format(s=0)), np.load(pat[p].format(s=0))
        assert np.array_equal(x["test_idx"], y["test_idx"]), p
        print(f"mor_{a} {p} seed 0: vs the 10-15 MOR files max |d prob| {float(np.abs(x['prob_last'] - y['prob_last']).max()):.3f}")
    mem[f"mor_{a}"] = dict(offset=0, **pat)
x, y = (np.load(mem[f"mor_{a}"]["kit"].format(s=0)) for a in ("ctl", "rc"))
d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
print(f"mor_rc vs mor_ctl kit seed 0: max |d prob| {d:.3f}")
assert d > 1e-4, "rc equals ctl"
base = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2}
spec = dict(n=12, anchor="c_ctl", bar="d141", members=mem,
            arms={"c_ctl": dict(base, mor_ctl=0.1), "c_rc": dict(base, mor_rc=0.1), "c_old": dict(base, mor=0.1),
                  "mor_ctl_alone": {"mor_ctl": 1.0}, "mor_rc_alone": {"mor_rc": 1.0}, "mor_old_alone": {"mor": 1.0}})
sp = os.path.join(out, "spec.json")
open(sp, "w").write(json.dumps(spec, indent=1) + "\n")
print(f"spec {sp} sha256 {hashlib.sha256(open(sp, 'rb').read()).hexdigest()[:12]}")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d256/spec.json" --out "$O/t2_d256/read.json" > "$O/t2_d256/read.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d256/read.json" <<'PYEOF' | tee "$O/t2_d256/verdict.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
k = sum(rows[("kit", i, "mor_rc_alone")]["test"] - rows[("kit", i, "mor_ctl_alone")]["test"] for i in range(6)) / 6
c, o = r["arms"]["c_rc"], r["arms"]["c_old"]
met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
ok = k > -0.01 and met and c["new"]["d"] >= 0
am = r["arms"]["mor_rc_alone"]["mix"]["d"] - r["arms"]["mor_ctl_alone"]["mix"]["d"]
print(f"code drift (descriptive): c_old vs c_ctl mix {o['mix']['d']:+.4f} ({o['mix']['up']} / 12), MOR alone old - ctl "
      f"{r['arms']['mor_old_alone']['mix']['d'] - r['arms']['mor_ctl_alone']['mix']['d']:+.4f}")
print(f"rc: kill check (MOR alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_rc vs c_ctl: mix {c['mix']['d']:+.4f} "
      f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, "
      f"seen1 val {c['seen1_val']['d']:+.4f}; MOR alone (rc - ctl) mix {am:+.4f}")
print("VERDICT:", "KILLED" if k <= -0.01 else ("CANDIDATE -> T2-D256r (fresh seeds 12-23, ctl + rc) before any adoption" if ok else "NOT ADOPTED (the MOR fine-tune stays)"))
PYEOF
