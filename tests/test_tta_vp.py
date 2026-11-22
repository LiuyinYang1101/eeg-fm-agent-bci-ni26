#!/usr/bin/env python
"""Backlog 71: test-time averaging over the vp bank for the pooled CNNs (ni26.train.predict_proba_vp; t2p_bench --tta-vp), CPU:
  1. an identity bank (mats I, gains 1, shifts 0) gives predict_proba exactly;
  2. on a random bank, predict_proba_vp(K) = the mean over the identity and draws 0..K-2, each draw transformed EXACTLY as augment
     transforms a training window with that draw (augment replayed on a one-draw bank), for both CNN families;
  3. K outside [2, bank size + 1] or no bank is refused;
  4. t2p_bench --list-outputs gates the <recipe>tv<K> twin per CNN task with --tta-vp, and nothing extra without it.
  python tests/test_tta_vp.py  ->  "OK"
"""
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ni26.train as T  # noqa: E402


def bank(C, K, rng):
    mats = np.stack([np.eye(C) + 0.05 * rng.standard_normal((C, C)) for _ in range(K)])
    return dict(mats=torch.as_tensor(mats, dtype=torch.float32), gains=torch.as_tensor(rng.uniform(0.8, 1.2, (K, C)), dtype=torch.float32),
                shifts=torch.as_tensor(rng.integers(-6, 7, K), dtype=torch.float32))


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    C, L, fs = 41, 480, 120.0
    X = torch.as_tensor(rng.standard_normal((20, C, L)), dtype=torch.float32)
    rec = replace(T.RECIPES["r1"], vp=2)
    for name in ("fbcnet_bboth", "shallow"):
        net = T.build_model(name, C, 3, L, fs).eval()
        T.VP_BANK.clear()
        T.VP_BANK.update(mats=torch.eye(C)[None].repeat(4, 1, 1), gains=torch.ones(4, C), shifts=torch.zeros(4))
        assert np.allclose(T.predict_proba_vp(net, X, 4), T.predict_proba(net, X), atol=1e-6), name
        b = bank(C, 5, rng)
        T.VP_BANK.clear(); T.VP_BANK.update(b)
        got = T.predict_proba_vp(net, X, 4)
        want = [T.predict_proba(net, X)]
        for k in range(3):                       # augment on a one-draw bank = draw k applied to every window
            T.VP_BANK.clear(); T.VP_BANK.update({n: v[k:k + 1] for n, v in b.items()})
            want.append(T.predict_proba(net, T.augment(X.clone(), rec)))
        assert np.allclose(got, np.mean(want, 0), atol=1e-5), (name, float(np.abs(got - np.mean(want, 0)).max()))
        assert not np.allclose(got, want[0], atol=1e-4), "the bank must change the prediction"
        T.VP_BANK.clear(); T.VP_BANK.update(b)             # T2-D302 (a): + shift-only views of draws K-1.. (no remap, no gain)
        gs = T.predict_proba_vp(net, X, 3, shift_only=3)
        ws = want[:3] + [T.predict_proba(net, torch.roll(X, shifts=int(b["shifts"][k]), dims=2)) for k in (2, 3)]
        assert np.allclose(gs, np.mean(ws, 0), atol=1e-5), (name, float(np.abs(gs - np.mean(ws, 0)).max()))
        assert np.allclose(T.predict_proba_vp(net, X, 4, shift_only=1), got, atol=1e-6), "shift_only 1 = no extra view"
        T.VP_BANK.clear(); T.VP_BANK.update(b)
        for bad in (1, 7):
            try:
                T.predict_proba_vp(net, X, bad)
                raise SystemExit(f"K {bad} must be refused")
            except AssertionError:
                pass
        print(f"{name}: identity bank = predict_proba; K 4 = mean(identity, augment's draws 0-2) (max err "
              f"{float(np.abs(got - np.mean(want, 0)).max()):.1e}); K 1 / 7 refused")
    T.VP_BANK.clear()
    try:
        T.predict_proba_vp(net, X, 4)
        raise SystemExit("no bank must be refused")
    except AssertionError:
        pass

    exp = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
    line = "--protocols kit,seen1 --models fbcnet_bboth,shallow --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds 204,205 --peruser"
    with tempfile.TemporaryDirectory() as td:
        lists = {}
        for k, extra in (("plain", ""), ("tv", " --tta-vp 8"), ("tvsc", " --tta-vp 8 --tta-shift 8 --tta-copies")):
            f = Path(td) / f"{k}.txt"
            subprocess.run([sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_bench.py"), "--export", str(exp), "--out", td,
                            *(line + extra).split(), "--list-outputs", str(f)], check=True, capture_output=True)
            lists[k] = f.read_text().split()
        extra = sorted(set(lists["tv"]) - set(lists["plain"]))
        assert set(lists["plain"]) <= set(lists["tv"]) and len(extra) == len(lists["plain"]), (len(extra), len(lists["plain"]))
        assert all("_rc75tv8__s" in e for e in extra), extra[:2]
    print(f"t2p_bench --list-outputs: --tta-vp 8 adds one <recipe>tv8 twin per CNN task ({len(extra)} for {len(lists['plain'])}), none without it")
    more = sorted(set(lists["tvsc"]) - set(lists["tv"]))
    assert len(more) == 2 * len(lists["plain"]) and {m.split("__n00__")[1].split("__s")[0][-5:] for m in more} == {"tv8s8", "5tv8c"}, more[:4]
    print(f"  + --tta-shift 8 --tta-copies: the tv8s8 and tv8c twins per CNN task ({len(more)})")
    print("OK")


if __name__ == "__main__":
    main()
