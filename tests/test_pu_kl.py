"""T2-D430: stranger-safe per-user copies (Recipe.pu_kl -> finetune's kl_other)."""
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26 import train as T  # noqa: E402


def _net():
    from braindecode.models import ShallowFBCSPNet
    torch.manual_seed(0)
    return ShallowFBCSPNet(n_chans=4, n_outputs=3, n_times=480).eval()


def _kl(net, copy, X):
    with torch.no_grad():
        p = F.softmax(net(X), 1)
        return float(F.kl_div(F.log_softmax(copy(X), 1), p, reduction="batchmean"))


def test_off_is_bitwise_and_on_stays_pooled_on_strangers():
    g = torch.Generator().manual_seed(1)
    X, y = torch.randn(48, 4, 480, generator=g), torch.randint(0, 3, (48,), generator=g)
    Xo = torch.randn(96, 4, 480, generator=g) * 1.5
    net = _net()
    rec = replace(T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr"]), epochs=6)
    assert rec.kl_other == 0.0
    a, b = T.finetune(net, X, y, rec, 3), T.finetune(net, X, y, rec, 3, X_other=Xo)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    rk = replace(T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr_kl1"]), epochs=6)
    assert rk.kl_other == 1.0
    c = T.finetune(net, X, y, rk, 3, X_other=Xo)
    assert _kl(net, c, Xo) < _kl(net, a, Xo)


def test_kl_needs_strangers():
    net = _net()
    rk = replace(T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr_kl1"]), epochs=1)
    try:
        T.finetune(net, torch.randn(8, 4, 480), torch.zeros(8, dtype=torch.long), rk, 0)
    except AssertionError:
        return
    raise AssertionError("kl_other without X_other must fail")
