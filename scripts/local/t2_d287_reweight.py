#!/usr/bin/env python
"""T2-D287 (zero GPU; pre-registered in docs/DECISIONS_T2.md): the closed seen-vs-new trades re-read at the sealed session split.
Every outputs/t2_dNNN/read.json written by t2p_pkgshape (rows T2-D180 ... D284): each package arm vs its row's anchor (member-alone
arms excluded), per index i: seen_i, new_i = (kit test_i + seen1 val_i) / 2, mix_w = (1 - w) seen + w new at w = 0.40 / 0.57.
A trade = seen and new deltas of opposite sign, each |d| >= 0.001.
REOPEN candidate iff a trade arm meets d141 with mix_w at BOTH w (mix_w >= +0.005, >= 9 / 12 up, kit test >= -0.005);
one end only -> split-dependent; met d141 at 1/2 but fails at an end -> fragile.
  python -I scripts/local/t2_d287_reweight.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
WS = (0.40, 0.50, 0.57)


def d141(mix, kt):
    return float(np.mean(mix)) >= 0.005 and int((mix > 0).sum()) >= 9 and float(np.mean(kt)) >= -0.005


def main():
    out, n_read = [], 0
    for d in sorted(O.glob("t2_d[0-9]*")):
        m = re.fullmatch(r"t2_d(\d+)(r?)", d.name)
        if not m or not (180 <= int(m[1]) <= 284) or not (d / "read.json").is_file():
            continue
        r = json.load(open(d / "read.json"))
        if not isinstance(r, dict) or "rows" not in r or "spec" not in r or "anchor" not in r.get("spec", {}):
            continue
        df = pd.DataFrame(r["rows"])
        if not {"proto", "i", "arm"} <= set(df.columns) or set(df.proto) != {"kit", "seen1"}:
            continue
        n_read += 1
        anc = r["spec"]["anchor"]
        n = int(r["spec"]["n"])

        def wide(proto, col):
            return df[df.proto == proto].pivot_table(index="i", columns="arm", values=col)
        kt, ss, sv = wide("kit", "test"), wide("seen1", "seen"), wide("seen1", "val")
        for a in r["spec"]["arms"]:
            if a == anc or a.startswith("a_") or a.endswith("_alone") or a not in kt.columns:
                continue
            dk, ds, dv = (kt[a] - kt[anc]).to_numpy(), (ss[a] - ss[anc]).to_numpy(), (sv[a] - sv[anc]).to_numpy()
            new = (dk + dv) / 2
            row = dict(row=d.name, arm=a, n=n, seen=float(ds.mean()), new=float(new.mean()), kit_test=float(dk.mean()))
            row["trade"] = bool(np.sign(row["seen"]) != np.sign(row["new"]) and abs(row["seen"]) >= 0.001 and abs(row["new"]) >= 0.001)
            for w in WS:
                mw = (1 - w) * ds + w * new
                row[f"mix{w:.2f}"] = float(mw.mean())
                row[f"up{w:.2f}"] = int((mw > 0).sum())
                row[f"met{w:.2f}"] = d141(mw, dk) if n == 12 else False
            out.append(row)
    df = pd.DataFrame(out)
    tr = df[df.trade]
    print(f"{n_read} package-shape reads, {len(df)} package arms, {len(tr)} seen-vs-new trades (|d| >= 0.001 each)")
    cols = ["row", "arm", "seen", "new", "kit_test", "mix0.40", "up0.40", "mix0.50", "up0.50", "mix0.57", "up0.57"]
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.float_format", "{:+.4f}".format):
        print(tr.sort_values("mix0.57", ascending=False)[cols].to_string(index=False))
    reopen = tr[tr["met0.40"] & tr["met0.57"] & ~tr["met0.50"]]
    split = tr[(tr["met0.40"] ^ tr["met0.57"]) & ~tr["met0.50"]]
    fragile = df[df["met0.50"] & ~(df["met0.40"] & df["met0.57"])]
    print("REOPEN candidates (d141 at both 0.40 and 0.57, not at 1/2): " + (", ".join(f"{a.row}:{a.arm}" for a in reopen.itertuples()) or "none"))
    print("split-dependent (d141 at one end only): "
          + (", ".join(f"{x['row']}:{x['arm']} (met at {'0.40' if x['met0.40'] else '0.57'})" for _, x in split.iterrows()) or "none"))
    print("fragile (d141 at 1/2, not at both ends): " + (", ".join(f"{x['row']}:{x['arm']} (0.40 {x['met0.40']}, 0.57 {x['met0.57']})" for _, x in fragile.iterrows()) or "none"))
    (O / "t2_d287").mkdir(exist_ok=True)
    df.to_csv(O / "t2_d287" / "reweight.csv", index=False)
    json.dump(dict(reopen=reopen.to_dict("records"), split=split.to_dict("records"), fragile=fragile.to_dict("records")),
              open(O / "t2_d287" / "verdict.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
