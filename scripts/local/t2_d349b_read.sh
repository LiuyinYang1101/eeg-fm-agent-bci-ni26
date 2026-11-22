#!/usr/bin/env bash
# T2-D349b read (pre-registered 2026-11-05 22:56; this script written 2026-11-06 before any D349b number): per arm (a head-freeze, b token-drop +
# head wd + head lr, c LoRA-in 16 at lr 5e-4) the per-epoch curves (train loss / val / test, mean over seeds) and a 4-restart ensemble (draw i:
# seeds i, i+3, i+6, i+9 mod 12, T2-D349's construction) vs mean4 on the k4_35 shape, under SKλ; then the best arm's member as a 5th slot member.
#   mem-guard bash scripts/local/t2_d349b_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d349b
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
O = sys.argv[1]; S = f"{O}/sofia_s1"
D = {"a": ("ni26-t2p-d349b-a-kit-1566581", "ni26-t2p-d349b-a-seen1-1568516"), "b": ("ni26-t2p-d349b-b-kit-1568517", "ni26-t2p-d349b-b-seen1-1568518"),
     "c": ("ni26-t2p-d349b-c-kit-1568519", "ni26-t2p-d349b-c-seen1-1566586")}
s = json.load(open(f"{O}/t2_d349/spec.json")); M = {k: v for k, v in s["members"].items() if k in ("fbc_tv8c", "sh_tv8c") or k.startswith("mean")}
R_ = re.compile(r"\]\s+ep (\d+) loss ([\d.]+) val ([\d.na]+) test ([\d.na]+)")
for arm, (kd, sd) in D.items():
    src = {}
    for p, d in (("kit", kd), ("seen1", sd)):
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-grid__*__s*.npz")}); assert len(st) == 1, (arm, p, st)
        src[p] = st[0] + "__s{s}.npz"
        assert all(os.path.exists(src[p].format(s=i)) for i in range(12)), (arm, p)
        cur = {}
        for f in glob.glob(f"{S}/{d}/*.log"):
            for line in open(f):
                m = R_.search(line)
                if m:
                    e = int(m.group(1)); cur.setdefault(e, []).append([float(m.group(2))] + [float(x) if x != "nan" else np.nan for x in m.group(3, 4)])
        c = {e: np.nanmean(np.array(v), 0).round(4).tolist() for e, v in sorted(cur.items())}
        print(f"curves {arm} {p} (epoch: loss / val / test, mean over seeds):", {e: c[e] for e in (1, 5, 10, 15, 20, 25, 30) if e in c})
    for j in range(4):
        dj = f"{O}/t2_d349b/restarts/{arm}{j}"; os.makedirs(dj, exist_ok=True)
        M[f"{arm}{j}"] = {"offset": 0}
        for p in ("kit", "seen1"):
            for i in range(12):
                ln = f"{dj}/{p}__s{i}.npz"
                if os.path.lexists(ln): os.remove(ln)
                os.symlink(src[p].format(s=(i + 3 * j) % 12), ln)
            M[f"{arm}{j}"][p] = f"{dj}/{p}__s{{s}}.npz"
fw = 0.35; c = {"fbc_tv8c": (1 - fw) * 4 / 7, "sh_tv8c": (1 - fw) * 3 / 7}
arms = {"mean4": dict(c, **{f"mean{j}": fw / 4 for j in range(4)})}
for arm in D:
    arms[f"{arm}4"] = dict(c, **{f"{arm}{j}": fw / 4 for j in range(4)})
    arms[f"mean4+{arm}"] = dict(c, **{f"mean{j}": fw / 5 for j in range(4)}, **{f"{arm}0": fw / 5})   # the arm's member as a 5th slot member
json.dump({"n": 12, "anchor": "mean4", "members": M, "arms": arms}, open(f"{O}/t2_d349b/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d349b/spec.json": list(arms)}, open(f"{O}/t2_d349b/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
