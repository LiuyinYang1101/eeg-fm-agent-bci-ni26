"""T2-D478: the stream-balanced sampler covers every window once per epoch, the loss is blind to a per-stream logit offset, and sbal = 0
leaves the shipped recipes unchanged (field default, peruser_for)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26 import train as T  # noqa: E402


def test_batches_cover_once():
    sid = torch.as_tensor(np.repeat(np.arange(5), [37, 16, 50, 3, 21]))
    g = torch.Generator().manual_seed(0)
    B = T.sbal_batches(sid, 64, 16, g)
    allix = torch.cat(B).sort().values
    assert torch.equal(allix, torch.arange(len(sid)))
    assert len(B) == T.sbal_steps(sid, 64, 16)
    assert all(len(torch.unique(sid[b])) <= 4 for b in B)


def test_loss_blind_to_stream_offset():
    torch.manual_seed(0)
    z = torch.randn(32, 3)
    y = torch.randint(0, 3, (32,))
    sid = torch.as_tensor(np.repeat([0, 1], 16))
    off = torch.zeros(32, 3)
    off[:16, 0] = 2.0                      # stream 0 shifted towards class 0
    off[16:, 2] = -1.5
    a = T.sbal_ce(z, y, sid, 0.0)
    b = T.sbal_ce(z + off, y, sid, 0.0)
    assert abs(float(a) - float(b)) < 0.05, (float(a), float(b))   # exactly blind only in the large-batch limit; close here


def test_off_is_default():
    for k in ("r1_pulxx_vp2_e180_vr_rc75", "r1_pulx_vp2_e180_vr"):
        r = T.RECIPES[k]
        assert r.sbal == 0 and T.peruser_for(r).sbal == 0
    assert T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr_sb16"]).sbal == 16
