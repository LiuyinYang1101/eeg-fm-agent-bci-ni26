#!/usr/bin/env python
"""T2-D560 GPU check (needs CUDA; not in the CPU self-test): the W20 template's CUDA-graph runner (StreamModel._run, its own capture /
replay / check logic) on a stub whose 'FM member' is a small net with a Linear `out` head under bf16 autocast, with D560's buffer hook
(StreamModel._fm_feat_copy): after each window's run, the buffer == the eager features of THAT window (max |d| <= 1e-2, bf16), for
windows before and after the capture, and the buffer changes from window to window. Run: python tests/gpu_t2p_w20_graph_hook.py -> OK"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
spec = importlib.util.spec_from_file_location("w20", ROOT / "tracks/t2_bci/submission/t2p_submission_w20.py")
W20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W20)
assert torch.cuda.is_available(), "needs CUDA"
dev = torch.device("cuda")
torch.manual_seed(0)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 96))
        self.out = torch.nn.Linear(96, 3)

    def forward(self, x):
        return self.out(self.body(x).mean(1))


net = Net().to(dev).eval()
m = W20.StreamModel.__new__(W20.StreamModel)
torch.nn.Module.__init__(m)
m.gshared = {"g": {}}
m._fm_feat_buf = [None]
net.out.register_forward_pre_hook(lambda mod, inp: m._fm_feat_copy(0, inp[0]))


def fn(zs, cache=True):                                  # = _gpu_eager's signature (the capture passes cache=False)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
        return [torch.softmax(net(zs[0]).float(), 1)[0]]


xs = [torch.randn(1, 20, 64, device=dev) for _ in range(6)]
prev, worst = None, 0.0
for t, x in enumerate(xs):
    with torch.no_grad():
        out = m._run(fn, [x], "main")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = net.body(x).mean(1).float().reshape(-1)
    b = m._fm_feat_buf[0].clone()
    d = float((b - ref).abs().max())
    worst = max(worst, d)
    moved = prev is None or float((b - prev).abs().max()) > 1e-3
    print(f"window {t}: graph state {sorted(k for k in m.gshared['g'])} off={m.gshared.get('off')} buffer vs eager |d| {d:.2e} moved {moved}", flush=True)
    assert moved, ("the buffer did not change between windows", t)
    prev = b
assert worst <= 1e-2, worst
assert m.gshared["g"].get("main") is not None and not m.gshared.get("off"), ("the graph was not used", m.gshared)
print("OK")
