#!/usr/bin/env bash
# T2-D361 read (pre-registered 2026-11-05; this script written 2026-11-06 before any D361 package number): ST-L fine-tune length 60 / 100 vs 30.
# (i) curves per length; (ii) member alone; (iii) nm35 with r0 -> e60 / e100 (anchor nm35); (iv) a 4-restart slot (seeds i, i+3, i+6, i+9;
# T2-D349's construction) vs mean4 at 30 on the k4_35 shape (anchor mean4). SKλ, 12 draws.
#   mem-guard bash scripts/local/t2_d361_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d361
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
O = sys.argv[1]; S = f"{O}/sofia_s1"
D = {"e60": ("ni26-t2p-d361-e60-kit-1566587", "ni26-t2p-d361-e60-seen1-1566588"), "e100": ("ni26-t2p-d361-e100-kit-1566589", "ni26-t2p-d361-e100-seen1-1566590")}
base = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(base["members"]); nm = base["arms"]["nm35"]
k4 = json.load(open(f"{O}/t2_d349/spec.json")); M.update({k: v for k, v in k4["members"].items() if k.startswith("mean")})
Rx = re.compile(r"\]\s+ep (\d+) loss ([\d.]+) val ([\d.na]+) test ([\d.na]+)")
for L, (kd, sd) in D.items():
    src = {}
    for p, d in (("kit", kd), ("seen1", sd)):
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (L, p, st)
        src[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(src[p].format(s=i)) for i in range(12)), (L, p)
        cur = {}
        for f in glob.glob(f"{S}/{d}/*.log"):
            for line in open(f):
                m = Rx.search(line)
                if m:
                    cur.setdefault(int(m.group(1)), []).append([float(m.group(2))] + [float(x) if x != "nan" else np.nan for x in m.group(3, 4)])
        es = [e for e in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) if e in cur]
        print(f"curves {L} {p} (epoch: loss / val / test):", {e: np.nanmean(np.array(cur[e]), 0).round(4).tolist() for e in es})
    M[L] = {"offset": 0, **src}
    for j in range(4):
        dj = f"{O}/t2_d361/restarts/{L}{j}"; os.makedirs(dj, exist_ok=True); M[f"{L}r{j}"] = {"offset": 0}
        for p in ("kit", "seen1"):
            for i in range(12):
                ln = f"{dj}/{p}__s{i}.npz"
                if os.path.lexists(ln): os.remove(ln)
                os.symlink(src[p].format(s=(i + 3 * j) % 12), ln)
            M[f"{L}r{j}"][p] = f"{dj}/{p}__s{{s}}.npz"
sw = lambda L: (lambda a: (a.__setitem__(L, a.pop("r0")), a)[1])(dict(nm))  # noqa: E731
A3 = {"nm35": nm, "nm35_e60": sw("e60"), "nm35_e100": sw("e100"), "m30": {"r0": 1.0}, "m60": {"e60": 1.0}, "m100": {"e100": 1.0}}
fw = 0.35; c = {"fbc_tv8c": (1 - fw) * 4 / 7, "sh_tv8c": (1 - fw) * 3 / 7}
A4 = {"mean4": dict(c, **{f"mean{j}": fw / 4 for j in range(4)}), "e60x4": dict(c, **{f"e60r{j}": fw / 4 for j in range(4)}), "e100x4": dict(c, **{f"e100r{j}": fw / 4 for j in range(4)})}
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": A3}, open(f"{O}/t2_d361/spec_iii.json", "w"), indent=1)
json.dump({"n": 12, "anchor": "mean4", "members": M, "arms": A4}, open(f"{O}/t2_d361/spec_iv.json", "w"), indent=1)
json.dump({f"{O}/t2_d361/spec_iii.json": list(A3), f"{O}/t2_d361/spec_iv.json": list(A4)}, open(f"{O}/t2_d361/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec_*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
