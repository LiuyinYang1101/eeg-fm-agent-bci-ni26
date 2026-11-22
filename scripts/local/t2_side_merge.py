#!/usr/bin/env python
"""T2-D549 (D543 for the build): one side.pt with the side nets of two t2p_side_final runs, the BASE's nets first (W18's side.pt, seeds
51700-04) then the ADDED run's (local job 13, seeds 51705-09). Asserted: the same kind / n0 / mu / sd / scale / recipe / export_sha; disjoint
seeds; the base's entries byte-identical in the output (t2p_repack --side-from re-checks it on sofia). Writes <out>/side.pt (+ its sha256).
    mem-guard python scripts/local/t2_side_merge.py --base <dir with side.pt> --add <dir with side.pt> --out <dir>
"""
import argparse
import hashlib
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bo = torch.load(Path(a.base) / "side.pt", map_location="cpu", weights_only=True)
    ba = torch.load(Path(a.add) / "side.pt", map_location="cpu", weights_only=True)
    for k in ("kind", "n0", "scale", "recipe", "export_sha"):
        assert bo.get(k) == ba.get(k), (k, bo.get(k), ba.get(k))
    for k in ("mu", "sd"):
        assert torch.equal(torch.as_tensor(bo[k]), torch.as_tensor(ba[k])), k
    assert not set(bo["seeds"]) & set(ba["seeds"]), (bo["seeds"], ba["seeds"])
    fams = [k for k in ("cnn2", "cnn3") if k in bo]
    assert fams == [k for k in ("cnn2", "cnn3") if k in ba], fams
    out = dict(bo, seeds=list(bo["seeds"]) + list(ba["seeds"]), **{f: list(bo[f]) + list(ba[f]) for f in fams})
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    torch.save(out, od / "side.pt")
    chk = torch.load(od / "side.pt", map_location="cpu", weights_only=True)
    for f in fams:
        assert len(chk[f]) == len(bo[f]) + len(ba[f])
        for i, e in enumerate(bo[f]):
            for part in ("net", "head"):
                assert all(torch.equal(e[part][k], chk[f][i][part][k]) for k in e[part]), (f, i, part)
    sha = hashlib.sha256((od / "side.pt").read_bytes()).hexdigest()
    (od / "side.pt.sha256").write_text(f"{sha}  side.pt\n")
    print(f"side merge: {[len(bo[f]) for f in fams]} + {[len(ba[f]) for f in fams]} nets per family, seeds {out['seeds']} -> {od / 'side.pt'} "
          f"(sha256 {sha[:16]})", flush=True)


if __name__ == "__main__":
    main()
