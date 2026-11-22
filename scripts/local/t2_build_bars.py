#!/usr/bin/env python
"""T2 build verification, step 2 (repo copy of T2-D276's scratchpad bars; used from T2-D284 on): the D140 bars per ZIP of row <ROW> plus a config
diff against a reference ZIP (descriptive). Bars: golden >= 0.995, worker_guard ok, kit Simulated + warm-up replays of the build and the repack,
clean_differ 0, robustness all_pass, `[ni26] cuda graph main: used` in every replay log, <= 18.1 ms / window (repack), build ZIP = the repack's input.
  python -I scripts/local/t2_build_bars.py --row d284 --ref <reference zip> name ...
"""
import argparse
import glob
import json
import os
import re
import zipfile

S = os.path.expanduser("~/workspace/outputs/neural_interface_challenge/sofia_s1")


def cfg(z):
    return json.loads(zipfile.ZipFile(z).read("config.json"))


def sub(z):
    return zipfile.ZipFile(z).read("submission.py")


def diff(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{p}.{k}") for k in sorted(set(a) | set(b))), [])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(a) <= 40 and any(isinstance(x, (dict, list)) for x in a):
        return sum((diff(x, y, f"{p}[{i}]") for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--ref", required=True, help="reference ZIP for the config diff (descriptive)")
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    allok = True
    for n in a.names:
        P = glob.glob(f"{S}/ni26-t2p-pkg-{a.row}-{n}-[0-9]*/")
        R = glob.glob(f"{S}/ni26-t2p-repack-{a.row}-{n}-[0-9]*/")
        if len(P) != 1 or len(R) != 1 or not glob.glob(f"{R[0]}t2p_{a.row}_{n}_r.zip"):
            print(f"{n}: not fetched yet ({P}, {R})")
            allok = False
            continue
        P, R = P[0], R[0]
        f = json.load(open(f"{P}t2p_{a.row}_{n}.final.json"))
        r = json.load(open(f"{R}t2p_{a.row}_{n}_r.repack.json"))
        reps = {k: os.path.exists(q) and "objective_balanced_accuracy_cell_mean" in json.load(open(q))
                for k, q in (("pkg_sim", f"{P}t2p_{a.row}_{n}.kit_sim.json"), ("pkg_warm", f"{P}t2p_{a.row}_{n}.kit_warmup.json"),
                             ("rep_sim", f"{R}t2p_{a.row}_{n}_r.kit_sim.json"), ("rep_warm", f"{R}t2p_{a.row}_{n}_r.kit_warmup.json"))}
        logs = glob.glob(f"{P}*.log") + glob.glob(f"{R}*.log")
        graph = sum(len(re.findall(r"\[ni26\] cuda graph main: used", open(lg, errors="replace").read())) for lg in logs)
        gb = f.get("golden_agreement_by") or {}
        bars = dict(golden=f["golden_agreement"] >= 0.995, guard=bool(f["worker_guard"]["ok"]), replays=all(reps.values()),
                    clean=r["clean_differ"] == 0, robust=bool(r["all_pass"]), graph=graph >= len(logs) and graph > 0,
                    ms=r["new_ms_per_window"] <= 18.1, zip_ok=r["old_zip_sha256"] == f["zip_sha256"])
        allok &= all(bars.values())
        z = glob.glob(f"{R}t2p_{a.row}_{n}_r.zip")[0]
        c = cfg(z)
        print(f"{n}: zip {r['zip_sha256'][:16]} ({r['zip_mb']} MB; build {f['zip_sha256'][:8]}); seeds {c['seeds']} / FM {[m['seed'] for m in c['fm_members']]}; "
              f"golden {f['golden_agreement']:.4f} on {f['golden_windows']} windows ({gb}); guard {f['worker_guard']['ok']}; replays {reps}; "
              f"clean_differ {r['clean_differ']}; robust {r['all_pass']}; graph used {graph} / {len(logs)} logs; "
              f"ms {f['package_seconds_per_window'] * 1e3:.2f} build / {r['new_ms_per_window']} repack")
        print(f"  BARS: {'ALL PASS' if all(bars.values()) else 'MISSED ' + str([k for k, v in bars.items() if not v])} {bars}")
        d = diff(c, cfg(a.ref))
        print(f"  config vs {os.path.basename(a.ref)}: {len(d)} differing paths: {d[:30]}; submission.py identical: {sub(z) == sub(a.ref)}")
    print("ALL BARS PASS" if allok else "NOT ALL PASS / FETCHED")


if __name__ == "__main__":
    main()
