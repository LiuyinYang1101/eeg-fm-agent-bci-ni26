#!/usr/bin/env bash
# T2-D407 read (pre-registered 2026-11-07 10:08; this script written before any D407 number): the in-house FM project's C + P1 as T2's MOR member.
# Four MOR members, each T2-D256's rc 0.75 MOR line (D386's arm line) on FRESH seeds 36-47: p1v2 (the package's MOR, pinned = the control),
# C150k (C alone, descriptive), CP1a (C + P1 arm 1, 8 sources), CP1b (C + P1 arm 2, +IS / UL).
#   spec_a (member alone, anchor m_p1v2, descriptive): m_C150k / m_CP1a / m_CP1b.
#   spec_b (package, anchor S3f = M6's shape, SKλ): SW_<x> = vp12 -> the MOR member at its 0.07 (runtime-neutral). THE DECISION ARMS:
#          SW_CP1a, SW_CP1b; SW_p1v2 (D386's reference on the new seeds) / SW_C150k descriptive; AD_<x> = + the member at 0.05 (over 40 ms: info).
# Bar (a decision arm): SW_x vs S3f typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; W50 reported (t2_cluster_ci.py), never judged.
# Two decision arms -> a pass is SELECTED: its own fresh replicate (seeds 48-59) first. Top-2 relevance line: typical >= +0.012 over M6 (W50 ≈ +0.029).
# X (env, space-separated) = the members present; default all four. A partial read (p1v2 + C150k) is descriptive only.
#   s1 fetch ni26 ni26-t2p-d407-<x>-<kit|seen1>-<jobid>  (each), then: mem-guard bash scripts/local/t2_d407_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d407
X="${X:-p1v2 C150k CP1a CP1b}"
mkdir -p "$R"
$PY - "$O" $X <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; X = tuple(sys.argv[2:]); R = f"{O}/t2_d407"; S = f"{O}/sofia_s1"
assert "p1v2" in X and set(X) <= {"p1v2", "C150k", "CP1a", "CP1b"}, X
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert "vp12" in S3f, sorted(S3f)
for x in X:
    m = {"offset": 36}
    for p in ("kit", "seen1"):
        ds = glob.glob(f"{S}/ni26-t2p-d407-{x}-{p}-[0-9]*"); assert len(ds) == 1, (x, p, ds)
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{ds[0]}/runs/{p}__fm-mor_p1v2-mean__*__s*.npz")}); assert len(st) == 1, (x, p, st)
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=36 + i)) for i in range(12)), (x, p)
    M[f"mor_{x}"] = m
import numpy as np
stems = {x: M[f"mor_{x}"]["kit"].rsplit("/", 1)[-1] for x in X}
# 14:3x fix (before any decision number): the bench's stem tags a contract-only checkpoint by its DIRECTORY (cptp-d407-ckpts), so C150k /
# CP1a / CP1b share one file name in different run dirs; check instead that every run of a member recorded ITS OWN --fm-ckpt (res.fm_args).
ck = {}
for x in X:
    for p in ("kit", "seen1"):
        got = {(json.loads(str(np.load(M[f"mor_{x}"][p].format(s=36 + i), allow_pickle=True)["res"])).get("fm_args", {}).get("fm_ckpt") or "pinned").rsplit("/", 1)[-1]
               for i in range(12)}
        assert len(got) == 1 and got == ({"pinned"} if x == "p1v2" else {f"{x}.pt"}), (x, p, got)
        ck[(x, p)] = got.pop()
assert len({ck[(x, "kit")] for x in X}) == len(X), ck
def sw(a, o, n):
    a = dict(a); assert o in a and n not in a, (o, n); a[n] = a.pop(o); return a
def add(a, n, frac=0.05):
    tot = sum(a.values()); a = {k: v * (tot - frac) / tot for k, v in a.items()}; a[n] = frac; return a
A = {"m_p1v2": {"mor_p1v2": 1.0}, **{f"m_{x}": {f"mor_{x}": 1.0} for x in X if x != "p1v2"}}
B = {"S3f": S3f, **{f"SW_{x}": sw(S3f, "vp12", f"mor_{x}") for x in X}, **{f"AD_{x}": add(S3f, f"mor_{x}") for x in X}}
for a in list(A.values()) + list(B.values()):
    assert abs(sum(a.values()) - 1) < 1e-9, a
json.dump({"n": 12, "anchor": "m_p1v2", "members": M, "arms": A}, open(f"{R}/spec_a.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": B}, open(f"{R}/spec_b.json", "w"), indent=1)
json.dump({f"{R}/spec_a.json": list(A), f"{R}/spec_b.json": list(B)}, open(f"{R}/sk.json", "w"))
print("[D407] members:", stems)
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
for x in ('CP1a', 'CP1b'):
    r = d[d.arm == f'SW_{x}']
    if not len(r): continue
    r = r.iloc[0]; ok = r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010 and r.newT >= 0
    print(f'[D407] DECISION SW_{x} vs S3f: typical {r.typical:+.4f} ({int(r.typical_up)} / 12), P04 {r.P04:+.4f}, new {r.newT:+.4f} -> '
          f\"{'BAR MET (selected: fresh replicate 48-59 first)' if ok else 'bar NOT met'}; top-2 line (>= +0.012): {'REACHED' if r.typical >= 0.012 else 'not reached'}\")"
for x in $X; do
  echo "== user-cluster CI + W50: SW_$x vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_b.json" S3f SW_$x > "$R/cluster_SW_$x.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster_SW_$x.txt"
done
