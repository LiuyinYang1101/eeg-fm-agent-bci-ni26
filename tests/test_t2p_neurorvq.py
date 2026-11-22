"""T2-D313: the t2p_fm NeuroRVQ family (research only). CPU: strict pretrained load (sha-checked), the 4 s / 120 Hz front end
(800 samples at 200 Hz, clip +-500 uV), the (B, C, 4, 800) grid, LLRD layer map, FMClf forward + backward.
  CUDA_VISIBLE_DEVICES="" mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python tests/test_t2p_neurorvq.py"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tracks" / "t2_bci" / "scripts"), str(ROOT / "tracks" / "t1_image" / "scripts")]
import t2p_fm  # noqa: E402

CH = ['C2', 'CP2', 'P2', 'Pz', 'CPz', 'Cz', 'C1', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'TP7', 'CP5', 'CP3', 'C3', 'PO8', 'P6', 'P4', 'TP8',
      'CP6', 'CP4', 'C6', 'C4', 'FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2', 'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz', 'C5']

enc = t2p_fm.make_encoder("neurorvq", CH, print, calib="gain")
assert enc.n_layers == 12 and enc.dim == 800, (enc.n_layers, enc.dim)
x = torch.randn(2, len(CH), 480)                              # aligned unit-scale 4 s window at 120 Hz
xp = enc.prep(x)
assert xp.shape == (2, len(CH), 800) and float(xp.abs().max()) <= 500.0 + 1e-4, (xp.shape, float(xp.abs().max()))
g = enc(xp, enc.n_layers)
assert g.shape == (2, len(CH), 4, 800), g.shape
assert enc.layer_of("nr.net.blocks.0.attn.qkv.weight") == 1 and enc.layer_of("nr.net.blocks.11.mlp.fc2.weight") == 12
assert enc.layer_of("nr.net.pos_embed") == 0 and enc.layer_of("nr.net.patch_embed.conv1_1.weight") == 0
clf = t2p_fm.FMClf(enc, enc.n_layers, "mean", 3)
out = clf(x)
assert out.shape == (2, 3), out.shape
out.sum().backward()
n_grad = sum(p.grad is not None for p in clf.parameters() if p.requires_grad)
n_req = sum(p.requires_grad for p in clf.parameters())
assert n_grad == n_req, (n_grad, n_req)
print(json.dumps(dict(ok=True, prov=enc.provenance, grid=list(g.shape), params_M=round(sum(p.numel() for p in enc.parameters()) / 1e6, 2))))

# T2-D313b: weight-parametrised LoRA r16 (alpha 32, delta dropout 0.1) on every Linear in the NeuroRVQ blocks (t1.inject_lora):
# NeuroRVQ's attention calls F.linear(x, self.qkv.weight, q/v bias), so forward-hook LoRA (inject_lora_input) never reaches qkv
enc2 = t2p_fm.make_encoder("neurorvq", CH, print, calib="gain")
t2p_fm.t1.inject_lora(enc2, enc2.n_layers, 16, 32.0, 0.1, print)
tr = [(n, p.numel()) for n, p in enc2.named_parameters() if p.requires_grad]
lora = [n for n, _ in tr if ".parametrizations.weight.0." in n]
assert len(lora) == 2 * 4 * 12, len(lora)                         # A + B on qkv / proj / fc1 / fc2 of 12 blocks
assert all(n in lora or "norm" in n for n, _ in tr), [n for n, _ in tr if n not in lora and "norm" not in n][:5]
assert sorted({enc2.layer_of(n) for n in lora}) == list(range(1, 13))
clf2 = t2p_fm.FMClf(enc2, enc2.n_layers, "mean", 3)
clf2(x).sum().backward()
no = [n for n, p in clf2.named_parameters() if p.requires_grad and p.grad is None]
assert not no, no[:5]
print(json.dumps(dict(ok_lora=True, n_lora_tensors=len(lora), trainable_M=round(sum(k for _, k in tr) / 1e6, 3))))
