#!/usr/bin/env python
"""T2-D475e: t2p_fm_stack_time.variant() (CPU, no export, no model): k copies of the first k ST members are appended as
fm_{n+j}.pt symlinks to fm_j.pt, every other file is a symlink to the base's, the base config is untouched, and a package
with per-member weights is refused (the copies assume the equal-weight mean).
  python tests/test_fm_stack_time.py  ->  "OK"
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm_stack_time as S  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "pkg"
        base.mkdir()
        fm = [dict(family="st", file=f"fm_{j}.pt", seed=j) for j in range(5)]
        cfg = dict(fm_weight=0.35, fm_members=fm)
        (base / "config.json").write_text(json.dumps(cfg))
        for f in ["submission.py", "weights.pt"] + [m["file"] for m in fm]:
            (base / f).write_text(f)
        n = S.variant(base, 3, Path(td) / "dup3")
        assert n == 8
        c3 = json.loads((Path(td) / "dup3/config.json").read_text())
        assert [m["file"] for m in c3["fm_members"]] == [f"fm_{j}.pt" for j in range(8)]
        assert [m["seed"] for m in c3["fm_members"]] == [0, 1, 2, 3, 4, 0, 1, 2]
        for j in range(3):
            assert (Path(td) / f"dup3/fm_{5 + j}.pt").read_text() == f"fm_{j}.pt", "a copy must point at its source member"
        assert (Path(td) / "dup3/weights.pt").is_symlink() and (Path(td) / "dup3/submission.py").read_text() == "submission.py"
        assert json.loads((base / "config.json").read_text()) == cfg, "the base config must stay as it was"
        # T2-D475g: cnn_keep keeps the first n seeds of each CNN family (pooled + each user's copies), written; the rest linked
        import torch
        torch.save([{"w": torch.full((1,), float(j))} for j in range(15)], base / "weights.pt")
        torch.save([{"w": torch.full((1,), 100.0 + j)} for j in range(15)], base / "weights2.pt")
        torch.save([[{"w": torch.full((1,), 10.0 * u + j)} for j in range(15)] for u in range(3)], base / "weights_pu.pt")
        cfg2 = dict(cfg, cnn2=dict(file="weights2.pt"), user_blend=dict(file="weights_pu.pt", n_seeds=15))
        (base / "config.json").write_text(json.dumps(cfg2))
        n = S.variant(base, 2, Path(td) / "dup2_cnn10", 10)
        d2 = Path(td) / "dup2_cnn10"
        assert n == 7
        w = torch.load(d2 / "weights.pt", weights_only=True)
        assert len(w) == 10 and [float(x["w"]) for x in w] == list(range(10)) and not (d2 / "weights.pt").is_symlink()
        assert len(torch.load(d2 / "weights2.pt", weights_only=True)) == 10
        pu = torch.load(d2 / "weights_pu.pt", weights_only=True)
        assert [len(u) for u in pu] == [10, 10, 10] and float(pu[2][9]["w"]) == 29.0
        assert json.loads((d2 / "config.json").read_text())["user_blend"]["n_seeds"] == 10
        assert (d2 / "fm_5.pt").read_text() == "fm_0.pt" and (d2 / "submission.py").is_symlink()
        assert len(torch.load(base / "weights.pt", weights_only=True)) == 15, "the base must stay as it was"
        (base / "config.json").write_text(json.dumps(dict(cfg, fm_members=[dict(m, weight=0.07) for m in fm])))
        try:
            S.variant(base, 2, Path(td) / "dupw")
            raise SystemExit("per-member weights must be refused")
        except AssertionError:
            pass
    print("OK")


if __name__ == "__main__":
    main()
