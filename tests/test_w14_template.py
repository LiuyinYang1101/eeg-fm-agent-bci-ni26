#!/usr/bin/env python
"""T2-D482: template W14 (PREPARED, NOT APPROVED) = W13 + a per-member input reference for FM members. CPU, no package:
  1. the template's car_windows / lap_windows are src/ni26/stream.py's bitwise (random windows with a NaN and a flat channel; batch and single);
  2. StreamModel._fm_x: ref None -> the window itself (the same object: W13's path), "car" / t2p_final's car: true -> stream.car_windows,
     "lap" -> stream.lap_windows with stream.lap_neighbours' table;
  3. source: W14 = W13 except the header note, the two verbatim functions, the fm_ref / fm_lapnb block, _fm_x, and the two FM input lines.
  python tests/test_w14_template.py  ->  "OK"
"""
import difflib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import stream  # noqa: E402

bu = types.ModuleType("benchmark_utils"); bs = types.ModuleType("benchmark_utils.base_solver")
bs.CompetSolver = object; bu.base_solver = bs
sys.modules.update({"benchmark_utils": bu, "benchmark_utils.base_solver": bs})
SUB = ROOT / "tracks/t2_bci/submission"
CH = ['C2', 'CP2', 'P2', 'Pz', 'CPz', 'Cz', 'C1', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'TP7', 'CP5', 'CP3', 'C3', 'PO8', 'P6', 'P4', 'TP8', 'CP6',
      'CP4', 'C6', 'C4', 'FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2', 'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz', 'C5']


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    w14 = load(SUB / "t2p_submission_w14.py", "w14")
    rng = np.random.default_rng(3)
    X = (rng.standard_normal((6, len(CH), 480)) * 12).astype(np.float32)
    X[2, 5] = 0.0
    X[3, 9, 17] = np.nan
    nb = stream.lap_neighbours(CH, 4)
    assert np.array_equal(w14.car_windows(X), stream.car_windows(X), equal_nan=True), "car_windows must be stream's bitwise"
    assert np.array_equal(w14.lap_windows(X, nb), stream.lap_windows(X, CH), equal_nan=True), "lap_windows must be stream's bitwise"
    for k in range(len(X)):
        assert np.array_equal(w14.car_windows(X[k]), stream.car_windows(X)[k], equal_nan=True)
        assert np.array_equal(w14.lap_windows(X[k], nb), stream.lap_windows(X, CH)[k], equal_nan=True)
    m = w14.StreamModel.__new__(w14.StreamModel)
    cfg_members = [dict(), dict(car=True), dict(ref="car"), dict(ref="lap", lap_nb=nb.tolist())]
    m.fm_ref = [c.get("ref") or ("car" if c.get("car") else None) for c in cfg_members]
    m.fm_lapnb = [np.asarray(c["lap_nb"], dtype=np.int64) if r == "lap" else None for c, r in zip(cfg_members, m.fm_ref)]
    x = X[0]
    assert m._fm_x(0, x) is x, "no ref: the delivered window itself (W13's path)"
    assert np.array_equal(m._fm_x(1, x), stream.car_windows(x)) and np.array_equal(m._fm_x(2, x), stream.car_windows(x))
    assert np.array_equal(m._fm_x(3, x), stream.lap_windows(x, CH))
    src = inspect_init = (SUB / "t2p_submission_w14.py").read_text()
    assert 'self.fm_ref = [m.get("ref") or ("car" if m.get("car") else None) for m in cfg.get("fm_members", [])]' in src
    w13 = (SUB / "t2p_submission_w13.py").read_text().splitlines()
    w14l = inspect_init.splitlines()
    removed = [l[2:] for l in difflib.ndiff(w13, w14l) if l.startswith("- ")]
    assert removed == ["            [al(xi) for al in self.fm_aligners]",
                       "                for net, al in zip(self.fm_nets, self.fm_aligners):",
                       "                    zf = torch.from_numpy(al(xi))[None].to(dev)"], removed
    print("OK")


if __name__ == "__main__":
    main()
