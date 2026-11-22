#!/usr/bin/env python
"""Read T2's official Codabench result files (zero compute; the user downloads them: <dir> holds prediction_result.zip + scoring_result.zip).
  read   <dir> [--replay X]          : cell-mean BA (the ranking metric), duration, worker platform; vs our kit replay when given
  probes --u1 <dir> --u2 <dir>       : uploads 1 and 2 (Nov 5 / 6) under the APPROVED 3-way rule (PLAN §6b item 8, panel amendment, user
                                       2026-10-23 15:53): u2 - u1 >= +0.004 -> the vp2 set; <= -0.004 -> the non-vp set; else 3 vp2 + 2 non-vp draws.
                                       Prints the chosen set and the ZIPs of uploads 3 / 4 / 5 (user 2026-10-23 16:57); the mixed case's ZIPs as
                                       approved by the user 2026-10-23 23:19 (vp2 ZIPs 1-3 + non-vp ZIPs 1-2; upload 3 = vp2 ZIP 2, 4 = vp15, 5 = vp2 + vr).
  guard5 --u5 <dir> --cmp <dir> <dir> : upload 5's loss guard: the last five keep the re-tunes unless u5 < mean(cmp) - 0.010; cmp = the comparable
                                       earlier uploads (vp2 set: uploads 2 and 3; non-vp set: 1 and 3; mixed: uploads 2 and 3, the vp2-recipe ones).
  (the old `sealed` mode encoded the superseded form D / item 8 rules; removed 2026-10-23)
The Codabench stdout line `[ni26] cuda graph main: used | fallback` is not in these files; it is on the submission's log page (a user check).
  python scripts/local/t2_official_read.py read ~/Downloads/results/track2 --replay 0.5677
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path


def load(d):
    d = Path(d).expanduser()
    sc = json.loads(zipfile.ZipFile(d / "scoring_result.zip").read("scores.json"))
    pz = zipfile.ZipFile(d / "prediction_result.zip")
    meta = pz.read("metadata").decode()
    plat = {}
    if "results.parquet" in pz.namelist():
        import pandas as pd
        r = pd.read_parquet(io.BytesIO(pz.read("results.parquet"))).iloc[0]
        plat = {k: r[k] for k in r.index if str(k).startswith(("platform", "system", "version-cuda", "version-torch", "version-numpy"))}
        plat["objective_n_cells"] = r.get("objective_n_cells")
    return sc, meta, plat


def cmd_read(a):
    sc, meta, plat = load(a.dir)
    print(f"cell-mean BA {sc['balanced_accuracy_cell_mean']:.5f}  (BA {sc['balanced_accuracy']:.5f}, acc {sc['accuracy']:.5f}); predict duration {sc['duration']:.1f} s")
    if a.replay is not None:
        print(f"vs our kit replay {a.replay:.5f}: {sc['balanced_accuracy_cell_mean'] - a.replay:+.5f}")
    print("ingestion:", " ".join(meta.split()))
    for k, v in plat.items():
        print(f"  {k}: {v}")


# the approved sets (docs/SUBMISSIONS.md; sha256 prefixes): ZIP 1 = the line (= the probe), ZIPs 2-5 = draws 1-4
SETS = {"non-vp": ["5acacd71", "f8238563", "5dfa7e82", "c1d30428", "f9002dc7"],
        "vp2": ["7aa7ba44", "3af46d0f", "a4af42e5", "99feb728", "601da65f"]}
LINE15 = {"non-vp": "191c922d", "vp2": "ae5cbeb4"}                   # T2-D211: upload 4 (the 15-seed runtime guard)
U5 = {"non-vp": None, "vp2": "5816509e (T2-D232 u5rc: the vp2 line + virtual references + FBCNet re-colouring rc75)"}   # T2-D215 / D230 apply to the vp2 line only (T2-D232 replaced 8c8e63eb 2026-10-24)


def rule3(d):
    d = round(d, 9)                                              # a difference of exactly 0.004 is not lost to float error
    return "vp2" if d >= 0.004 else ("non-vp" if d <= -0.004 else "mixed")


def plan(chosen):
    """uploads 3-5 and the last five for one 3-way outcome (PLAN §6b item 8)."""
    if chosen == "mixed":                                        # APPROVED by the user 2026-10-23 23:19 (PLAN §6b item 8)
        return dict(u3=SETS["vp2"][1] + " (vp2 ZIP 2)", u4=LINE15["vp2"] + " (vp15)", u5=U5["vp2"],
                    last5="vp2 ZIPs 1-3 " + " ".join(SETS["vp2"][:3]) + " + non-vp ZIPs 1-2 " + " ".join(SETS["non-vp"][:2])
                    + " (if upload 5's guard keeps the re-tune, it applies to the three vp2 draws; guard5 --cmp = uploads 2 and 3)")
    z = SETS[chosen]
    return dict(u3=z[1] + f" ({chosen} ZIP 2)", u4=LINE15[chosen] + f" ({chosen} 15-seed line)",
                u5=U5[chosen] or (z[3] + f" ({chosen} ZIP 4, descriptive: no adopted re-tune applies)"), last5=" ".join(z))


def cmd_probes(a):
    s1, s2 = (load(d)[0]["balanced_accuracy_cell_mean"] for d in (a.u1, a.u2))
    d = s2 - s1
    c = rule3(d)
    print(f"upload 1 (non-vp) {s1:.5f}, upload 2 (vp2-180) {s2:.5f}: u2 - u1 = {d:+.5f} -> the 3-way rule picks: {c.upper()}"
          + ("  (|d| < 0.004)" if c == "mixed" else ""))
    for k, v in plan(c).items():
        print(f"  {k}: {v}")
    print("  upload 4's rule: the last five are built 15-seed iff it completes on the worker with `[ni26] cuda graph main: used` (its log page: a user check)")


def cmd_guard5(a):
    s5 = load(a.u5)[0]["balanced_accuracy_cell_mean"]
    cs = [load(d)[0]["balanced_accuracy_cell_mean"] for d in a.cmp]
    m = sum(cs) / len(cs)
    keep = round(s5 - (m - 0.010), 9) >= 0
    print(f"upload 5 {s5:.5f} vs the comparable earlier uploads {', '.join(f'{x:.5f}' for x in cs)} (mean {m:.5f}): u5 - mean = {s5 - m:+.5f}")
    print(f"  -> the last five {'KEEP the re-tunes' if keep else 'DROP the re-tunes (u5 < mean - 0.010)'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read"); r.add_argument("dir"); r.add_argument("--replay", type=float)
    s = sp.add_parser("probes"); s.add_argument("--u1", required=True); s.add_argument("--u2", required=True)
    g = sp.add_parser("guard5"); g.add_argument("--u5", required=True); g.add_argument("--cmp", nargs="+", required=True)
    a = ap.parse_args()
    {"read": cmd_read, "probes": cmd_probes, "guard5": cmd_guard5}[a.cmd](a)


if __name__ == "__main__":
    main()
