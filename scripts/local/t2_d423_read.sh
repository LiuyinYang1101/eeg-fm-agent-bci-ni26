#!/usr/bin/env bash
# T2-D423 Stage B + C read (pre-registered in docs/DECISIONS_T2.md before any D423 number; written before the Stage B runs exist; arm S added 04:36).
# Arguments: the codec arms that PASSED Stage A, any of rbI (the iEEG-fit codec, job 1589048) and rbS (the scalp-fit codec, job 1589156).
# Inputs: the fetched sofia dirs ni26-t2p-d423-sh-ctl-<job>/ (ShallowNet-EA, M7's recipe, PLAIN export) and ni26-t2p-d423-sh-<arm>-<job>/ (the same on that
# arm's RE-EXPRESSED export), seeds 336-347, kit + seen1. Blends as shipped for ShallowNet (r 0.5, tag 2, tv8c; routing on the PLAIN fingerprints, as the
# package's router sees raw windows).
#   STAGE B (member alone, causal SKλ): each arm vs sh_ctl; member typical <= -0.010 KILLS that arm.
#   STAGE C reads ONE arm = the higher Stage B member typical among the arms not killed (M7's shape, outputs/t2_d408/spec_m7.json arm M7):
#     M7c = sh_tv8c -> sh_ctl (the ANCHOR) vs M7<arm> = sh_tv8c -> sh_<arm>; W50 + user-cluster CI.
#     BAR (R3): typical >= +0.015 with W50 > 0 ADOPTS (-> fresh-seed replicate, runtime, kit gate; the build is a template change = the user's line);
#     typical <= 0 KILLS; in between -> one more dose (RVQ 8 quantizers). Runtime is priced separately (the codec inside predict).
#   mem-guard bash scripts/local/t2_d423_read.sh rbI rbS
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { echo "usage: t2_d423_read.sh <arms that passed Stage A: rbI and / or rbS>" >&2; exit 2; }
for a in "$@"; do case "$a" in rbI|rbS) ;; *) echo "unknown arm $a" >&2; exit 2;; esac; done
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d423
mkdir -p "$R"
SEEDS=$(seq -s, 336 347)
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
# 2026-11-10 (before any Stage B number): D423_LOCAL=1 reads the LOCAL 5090 pair instead (the sofia queue put the rb arm ≈ 14 h out):
# control = ni26-t2p-d424-sh-ctl-5090 (the same arm file and seeds, local), arm = ni26-t2p-d423-sh-<arm>-5090 (scripts/local/t2_d423_local.sh).
one() {
  if [ "${D423_LOCAL:-}" = 1 ]; then
    local d; [ "$1" = sh-ctl ] && d="$S/ni26-t2p-d424-sh-ctl-5090/" || d="$S/ni26-t2p-d423-$1-5090/"
    [ -d "${d}runs" ] || { echo "need $d" >&2; exit 2; }; echo "${d}runs"; return
  fi
  local d=( "$S"/ni26-t2p-d423-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d423-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for arm in sh-ctl $(printf 'sh-%s ' "$@"); do
  for p in kit seen1; do
    blend --runs "$(one $arm)" --protocols $p --out "$R/route_$arm" --model shallow --align ea --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds "$SEEDS"
  done
done
$PY - "$O" "$R" "$@" <<'PYEOF'
import json, os, sys
O, R, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
m7 = json.load(open(f"{O}/t2_d408/spec_m7.json")); M, A7 = dict(m7["members"]), dict(m7["arms"]["M7"])
assert M["sh_tv8c"]["kit"].endswith("vrtv8cfb2__s{s}.npz"), M["sh_tv8c"]
def mem(arm):
    m = {"offset": 336, **{p: f"{R}/route_sh-{arm}/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{{s}}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=336 + i)) for p in ("kit", "seen1") for i in range(12)), arm
    return m
for a in ["ctl"] + arms:
    M[f"sh_{a}"] = mem(a)
Ms = {"n": 12, "anchor": "ctl", "members": {f"sh_{a}": M[f"sh_{a}"] for a in ["ctl"] + arms}, "arms": {a: {f"sh_{a}": 1.0} for a in ["ctl"] + arms}}
json.dump(Ms, open(f"{R}/spec_member.json", "w"), indent=1)
json.dump({f"{R}/spec_member.json": ["ctl"] + arms}, open(f"{R}/sk_member.json", "w"))
sw = lambda n: {(n if k == "sh_tv8c" else k): v for k, v in A7.items()}
A = {"M7c": sw("sh_ctl"), **{f"M7{a}": sw(f"sh_{a}") for a in arms}}
json.dump({"n": 12, "anchor": "M7c", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
for t in _member ""; do
  mkdir -p "$R/out$t"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R/out$t" --jobs 4 \
    --specs "$R/spec$t.json" --sk-list "$R/sk$t.json" > "$R/run$t.log" 2>&1
done
$PY -c "
import pandas as pd
for t in ('_member', ''):
    d=pd.read_csv('$R/out'+t+'/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
    print('== ' + ('STAGE B: MEMBER ALONE (each arm vs ctl)' if t else 'M7 SHAPE (every arm vs M7c; only the Stage-C arm is judged)'))
    print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
$PY - "$R" "$@" <<'PYEOF'
import os, re, subprocess, sys
import pandas as pd
R, arms = sys.argv[1], sys.argv[2:]
b = pd.read_csv(f"{R}/out_member/rescreen_sk.csv"); b = b[(b.status == "ok") & (b.dec == "sk")].set_index("arm")
alive = {a: float(b.loc[a, "typical"]) for a in arms if float(b.loc[a, "typical"]) > -0.010}
for a in arms:
    print(f"== STAGE B {a}: member typical {b.loc[a, 'typical']:+.4f} ({int(b.loc[a, 'typical_up'])} / 12) -> {'pass' if a in alive else 'KILLED (<= -0.010)'}")
if not alive:
    print("== VERDICT (pre-registered): D423 KILLED at STAGE B (every arm <= -0.010)"); sys.exit(0)
arm = max(alive, key=alive.get)
out = subprocess.run([sys.executable, "-I", "scripts/local/t2_cluster_ci.py", f"{R}/spec.json", "M7c", f"M7{arm}"], capture_output=True, text=True,
                     env=dict(os.environ, OMP_NUM_THREADS="2")).stdout
open(f"{R}/cluster_{arm}.txt", "w").write(out); print(out[out.find("per user"):] if "per user" in out else out[-1500:])
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", out).group(1))
c = pd.read_csv(f"{R}/out/rescreen_sk.csv"); c = c[(c.status == "ok") & (c.dec == "sk") & (c.arm == f"M7{arm}")].iloc[0]
if c.typical >= 0.015 and w50 > 0:
    v = "ADOPT (R3) -> fresh-seed replicate + runtime + kit gate; the build is a template change (the user's typed line)"
elif c.typical <= 0:
    v = "KILLED (R3: Stage C typical <= 0)"
else:
    v = "IN BETWEEN -> one more dose (RVQ 8 quantizers)"
print(f"== VERDICT (pre-registered): STAGE C arm {arm}: {v}  [M7 shape typical {c.typical:+.4f} ({int(c.typical_up)} / 12), W50 {w50:+.4f}, P04 {c.P04:+.4f}, newT {c.newT:+.4f}]")
PYEOF
