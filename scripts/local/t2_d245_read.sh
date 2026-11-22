#!/usr/bin/env bash
# T2-D245 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d245-stl-<mx8|ema>-<kit|seen1>-<jobid>:
# ST-L (the adopted T2-D247r recipe: rc 0.75 + LLRD 0.75) fine-tuned with mixup 0.8 (mx8) / scored with a weight EMA 0.995 (ema), seeds 0-11, vs the
# stl_rcll runs of T2-D247 (re-based 2026-10-25 03:2x, before any D245 number). Links the 2 x 24 result files into outputs/t2_d245/stl_<arm>/, checks each run
# logged its knobs (the result tag) and that the files differ from stl_rcll, then t2p_pkgshape on the adopted line {fbc_rc75 (T2-D231, seeds 204-215),
# sh_vr (T2-D215r, 168-179), stl_rcll, mor} with the ST-L member swapped (spec outputs/t2_d245/spec.json, sha256 e428e4b14e1f).
# Per arm: kill iff ST-L alone (arm - ctrl) kit test on seeds 0-5 <= -0.01; adopt-CANDIDATE iff c_<arm> vs c_rcll meets bar d141 AND new >= 0
# (both -> the higher mix); adoption needs T2-D245r on fresh seeds 12-23.
# Split 2026-10-25 04:3x, before any D245 number: the arms launch separately (mx8 within the GO, ema on the user's GO call), so the read takes the
# arms present as arguments; a one-arm read uses spec.json cut to that arm (its own sha256, printed), and every rule above is per arm.
#   mem-guard bash scripts/local/t2_d245_read.sh [mx8] [ema]     (default both; after t2_d215r_read.sh and t2_d231_read.sh, which write the CNN route files)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
declare -A K=([mx8]="ll0.75ema0dp0mx0.8cgrc0.75" [ema]="ll0.75ema0.995dp0mx0.4cgrc0.75")
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(mx8 ema)
for a in "${ARMS[@]}"; do [ -n "${K[$a]:-}" ] || { echo "unknown arm $a"; exit 2; }; done
T=$(IFS=_; echo "${ARMS[*]}"); SP=$O/t2_d245/spec.json; R=read; V=verdict
if [ "$T" != "mx8_ema" ]; then SP=$O/t2_d245/spec_$T.json; R=read_$T; V=verdict_$T
  $PY -I - "$O/t2_d245/spec.json" "$SP" "${ARMS[@]}" <<'PYEOF' || exit 2
import hashlib, json, sys
s, keep = json.load(open(sys.argv[1])), sys.argv[3:]
drop = {"mx8", "ema"} - set(keep)
s["members"] = {m: d for m, d in s["members"].items() if m not in {f"stl_{a}" for a in drop}}
s["arms"] = {a: w for a, w in s["arms"].items() if not any(a in (f"c_{x}", f"stl_{x}_alone") for x in drop)}
open(sys.argv[2], "w").write(json.dumps(s, indent=1) + "\n")
print(f"spec cut to {keep} (from spec.json): {sys.argv[2]} sha256 {hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()[:12]}")
PYEOF
fi
for a in "${ARMS[@]}"; do
  mkdir -p "$O/t2_d245/stl_$a"
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d245-stl-$a-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for stl $a $p: ${d[*]}"; exit 2; }
    n=0
    for f in "${d[0]}"runs/${p}__fm-st_large-*"${K[$a]}"cpt*__s*.npz; do [ -e "$f" ] && ln -sfn "$f" "$O/t2_d245/stl_$a/$(basename "$f")" && n=$((n + 1)); done
    [ "$n" -eq 12 ] || { echo "stl $a $p: $n / 12 result files with ${K[$a]}"; exit 3; }
  done
  [ "$(ls "$O/t2_d245/stl_$a" | wc -l)" -eq 24 ] || { echo "stl $a: need 24 files"; exit 4; }
done
$PY -I - "$SP" "${ARMS[@]}" <<'PYEOF' || exit 5
import json, sys
import numpy as np
m = json.load(open(sys.argv[1]))["members"]
for a in sys.argv[2:]:
    for p in ("kit", "seen1"):
        x, y = np.load(m["stl_rcll"][p].format(s=0)), np.load(m[f"stl_{a}"][p].format(s=0))
        assert np.array_equal(x["test_idx"], y["test_idx"]), (a, p)
        d = float(np.abs(x["prob_last"] - y["prob_last"]).max())
        print(f"{a} {p} seed 0: vs the adopted ST-L (stl_rcll) max |d prob| {d:.3f}")
        assert d > 1e-4, "the runs equal the reference"
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$SP" --out "$O/t2_d245/$R.json" > "$O/t2_d245/$R.txt" 2>&1; echo "rc $?"
$PY -I - "$O/t2_d245/$R.json" "${ARMS[@]}" <<'PYEOF' | tee "$O/t2_d245/$V.txt"
import json, sys
r = json.load(open(sys.argv[1])); rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ok = {}
for a in sys.argv[2:]:
    k = sum(rows[("kit", i, f"stl_{a}_alone")]["test"] - rows[("kit", i, "stl_rcll_alone")]["test"] for i in range(6)) / 6
    c = r["arms"][f"c_{a}"]
    met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
    ok[a] = k > -0.01 and met and c["new"]["d"] >= 0
    print(f"{a}: kill check (ST-L alone, kit test, seeds 0-5) {k:+.4f} -> {'KILL' if k <= -0.01 else 'continue'}; c_{a} vs c_rcll: mix {c['mix']['d']:+.4f} "
          f"({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f} -> "
          + ("KILLED" if k <= -0.01 else ("CANDIDATE" if ok[a] else "no")))
c = [a for a in ok if ok[a]]
best = max(c, key=lambda a: r["arms"][f"c_{a}"]["mix"]["d"]) if c else None
print("VERDICT:", f"candidate {best} -> launch T2-D245r (fresh seeds 12-23) before any adoption" if best else "NOT ADOPTED (the ST-L recipe stays)")
PYEOF
