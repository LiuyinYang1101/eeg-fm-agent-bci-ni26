#!/usr/bin/env python
"""T2-D288 member check (pre-registered in docs/DECISIONS_T2.md): each SK10 repack vs its source ZIP of docs/T2_SEALED_MANIFEST.md, no unpacking.
Every weights*.pt / fm_*.pt: same name, size and CRC-32; config.json equal except `tta` (= sinkhorn, 10, 3); submission.py sha256 = T2-W7's.
Plus the repack.json bars (T2-D286's): clean_differ 0, all_pass, ms <= 18.1, and the log's graph / SK-recombine lines.
  python -I scripts/local/t2_d288_members.py [--row d288]
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

S = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1"
MAN = Path(__file__).resolve().parents[2] / "docs" / "T2_SEALED_MANIFEST.md"
SK = {"kind": "sinkhorn", "iters": 10, "n0": 3}
W7_SUB = "19bccb7ffe43"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="d288")
    a = ap.parse_args()
    rows = [ln for ln in MAN.read_text().splitlines() if re.match(r"\| (Nov 5|L_[VN] next)", ln)]
    assert len(rows) == 15, len(rows)
    out, n_ok = [], 0
    for ln in rows:
        c = [x.strip().strip("`") for x in ln.split("|")]
        src, src_sha = S / c[8].removeprefix("sofia_s1/"), c[6]
        tag = re.sub(r"^t2p_d\d{3}_", "", Path(src).stem).removesuffix("_r")
        d = sorted(S.glob(f"ni26-t2p-repack-{a.row}-{tag}sk-*"))
        if len(d) != 1 or not list(d[0].glob("t2p_*.zip")):
            print(f"{tag}: no fetched repack ({len(d)} dirs)")
            out.append(dict(tag=tag, ok=False, why="not fetched"))
            continue
        z = next(d[0].glob("t2p_*.zip"))
        why = []
        A, B = zipfile.ZipFile(src), zipfile.ZipFile(z)
        ia, ib = {i.filename: i for i in A.infolist()}, {i.filename: i for i in B.infolist()}
        if set(ia) != set(ib):
            why.append(f"file sets differ {sorted(set(ia) ^ set(ib))}")
        for f in sorted(set(ia) & set(ib)):
            if f.endswith(".pt") and (ia[f].CRC, ia[f].file_size) != (ib[f].CRC, ib[f].file_size):
                why.append(f"{f} differs")
        ca, cb = json.loads(A.read("config.json")), json.loads(B.read("config.json"))
        if cb.get("tta") != SK:
            why.append(f"tta {cb.get('tta')}")
        if {k: v for k, v in ca.items() if k != "tta"} != {k: v for k, v in cb.items() if k != "tta"}:
            why.append("config differs beyond tta: " + ",".join(k for k in set(ca) | set(cb) if k != "tta" and ca.get(k) != cb.get(k)))
        sub = hashlib.sha256(B.read("submission.py")).hexdigest()
        if not sub.startswith(W7_SUB):
            why.append(f"submission.py {sub[:12]}")
        r = json.load(open(next(d[0].glob("t2p_*.repack.json"))))
        if r["clean_differ"] != 0 or not r["all_pass"] or r["new_ms_per_window"] > 18.1:
            why.append(f"bars differ {r['clean_differ']} all_pass {r['all_pass']} ms {r['new_ms_per_window']}")
        log = S / "logs" / f"{d[0].name}.out"
        lt = log.read_text(errors="replace") if log.is_file() else ""
        n_graph = lt.count("[ni26] cuda graph main: used")
        if n_graph < 2:
            why.append(f"graph lines {n_graph} (log {'present' if lt else 'missing'})")
        sha = hashlib.sha256(z.read_bytes()).hexdigest()
        ok = not why
        n_ok += ok
        out.append(dict(tag=tag, src=str(src), src_sha=src_sha, zip=str(z), sha256=sha, mb=round(z.stat().st_size / 1e6), ok=ok, why=why,
                        ms=r["new_ms_per_window"], kit_test=r["kit_test_cellba"], clean_differ=r["clean_differ"], all_pass=r["all_pass"],
                        graph_lines=n_graph))
        print(f"{tag:10s} {'PASS' if ok else 'FAIL'} {sha[:16]} ms {r['new_ms_per_window']} kit test {r['kit_test_cellba']:.4f} "
              f"graph {n_graph} " + ("; ".join(why) if why else "members identical, config = source + tta"))
    print(f"{n_ok} / 15 pass")
    (Path.home() / f"workspace/outputs/neural_interface_challenge/t2_{a.row}").mkdir(exist_ok=True)
    (Path.home() / f"workspace/outputs/neural_interface_challenge/t2_{a.row}/members.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
