#!/usr/bin/env bash
# T2-D382 read (pre-registered 2026-11-06 11:41; this script written before any D382 number): M6's CNN lines at 360 vs 180 epochs, PAIRED on
# fresh seeds (FBCNet 228-239, ShallowNet 192-203, IFNet 180-191; sofia 1569851-56). Blends as shipped (fbc r 1/3 tag 3; sh / ifn r 0.5 tag 2), tv8c.
#   L360 (THE DECISION ARM, anchor L180c) = S3f with fbc_tl15f / sh_tv8c / ifn -> the e360 runs; L180c = the same with the paired e180 controls.
#   Also vs S3f: L360, L180c; singles (one family at 360, the others at their paired e180): L360f / L360s / L360i (anchor L180c). Curves descriptive.
# Bar: L360 vs L180c typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0; the user-cluster CI reported.
#   mem-guard bash scripts/local/t2_d382_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d382
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d382-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d382-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
for L in 360 180; do
  F=$(one fbc_e${L}_s228); H=$(one sh_e${L}_s192); I=$(one ifn_e${L}_s180)
  for p in kit seen1; do
    blend --runs "$F" --protocols $p --out "$R/route_fbc_e$L" --model fbcnet_bboth --recipe r1_pulxx_vp2_e${L}_vr_rc75_tl15tv8c --seeds $(seq -s, 228 239)
    blend --runs "$H" --protocols $p --out "$R/route_sh_e$L" --model shallow --align ea --recipe r1_pulx_vp2_e${L}_vrtv8c --r 0.5 --tag 2 --seeds $(seq -s, 192 203)
    blend --runs "$I" --protocols $p --out "$R/route_ifn_e$L" --model ifnet --align ea --recipe r1_pulx_vp2_e${L}_vrtv8c --r 0.5 --tag 2 --seeds $(seq -s, 180 191)
  done
done
$PY - "$O" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
O = sys.argv[1]; R = f"{O}/t2_d382"; S = f"{O}/sofia_s1"
d378 = json.load(open(f"{O}/t2_d378/spec.json")); M = dict(d378["members"]); S3f = d378["arms"]["S3f"]
assert {"fbc_tl15f", "sh_tv8c", "ifn"} <= set(S3f), sorted(S3f)
def mem(off, pat):
    m = {"offset": off, **{p: pat.format(p=p) + "__s{s}.npz" for p in ("kit", "seen1")}}
    assert all(os.path.exists(m[p].format(s=off + i)) for p in ("kit", "seen1") for i in range(12)), pat
    return m
for L in (360, 180):
    M[f"fbc_e{L}"] = mem(228, f"{R}/route_fbc_e{L}/{{p}}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e{L}_vr_rc75_tl15tv8cfb3")
    M[f"sh_e{L}"] = mem(192, f"{R}/route_sh_e{L}/{{p}}__shallow__ea__n00__r1_pulx_vp2_e{L}_vrtv8cfb2")
    M[f"ifn_e{L}"] = mem(180, f"{R}/route_ifn_e{L}/{{p}}__ifnet__ea__n00__r1_pulx_vp2_e{L}_vrtv8cfb2")
# member-alone curves (descriptive): the bench's per-epoch val / test lines
Rx = re.compile(r"\]\s+ep (\d+) loss ([\d.]+) val ([\d.na]+) test ([\d.na]+)")
for f in ("fbc", "sh", "ifn"):
    for L in (360, 180):
        cur = {}
        for lg in glob.glob(f"{S}/ni26-t2p-d382-{f}_e{L}_s*/*.log"):
            for line in open(lg):
                m = Rx.search(line)
                if m:
                    cur.setdefault(int(m.group(1)), []).append([float(m.group(2))] + [float(x) if x != "nan" else np.nan for x in m.group(3, 4)])
        es = [e for e in (1, 60, 120, 180, 240, 300, 360) if e in cur]
        print(f"curves {f} e{L} (epoch: loss / val / test):", {e: np.nanmean(np.array(cur[e]), 0).round(4).tolist() for e in es})
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
fam = (("fbc_tl15f", "fbc"), ("sh_tv8c", "sh"), ("ifn", "ifn"))
L180c = sw(S3f, [(o, f"{n}_e180") for o, n in fam]); L360 = sw(S3f, [(o, f"{n}_e360") for o, n in fam])
A = {"L180c": L180c, "L360": L360}
for o, n in fam:
    A[f"L360{n[0]}"] = sw(L180c, [(f"{n}_e180", f"{n}_e360")])
B = {"S3f": S3f, "L360": L360, "L180c": L180c}
json.dump({"n": 12, "anchor": "L180c", "members": M, "arms": A}, open(f"{R}/spec_a.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "S3f", "members": M, "arms": B}, open(f"{R}/spec_b.json", "w"), indent=1)
json.dump({f"{R}/spec_a.json": list(A), f"{R}/spec_b.json": list(B)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: L360 vs L180c (the decision contrast)"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_a.json" L180c L360 > "$R/cluster_L360.txt" 2>&1; tail -8 "$R/cluster_L360.txt"
echo "== user-cluster CI: L360 vs S3f"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_b.json" S3f L360 > "$R/cluster_L360_S3f.txt" 2>&1; tail -8 "$R/cluster_L360_S3f.txt"
