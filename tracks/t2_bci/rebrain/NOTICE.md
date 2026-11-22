# Vendored ReBrain code (T2-D423)

Source: github.com/Xianglong-Bao/ReBrain at commit 41881e56dceaf9d070d9ea9264ea6b8f7bd7c01e (2026-07-22), licensed Apache-2.0 (LICENSE_REBRAIN, copied unchanged).
ReBrain's README states that it builds on EnCodec and BrainCodec (Apache-2.0).

## Files copied, and what changed in them
- `encodec.py`: copied from `models/encodec.py`. Only the import `from utils.conv import` became `from .conv import`.
- `discriminators.py` (from `models/`), and `conv.py`, `loss.py`, `balancer.py` (from `utils/`): copied unchanged.

## New code (not ReBrain's)
- `__init__.py`: `build_codec()` holds the values of `configs/ReBrain_train_ieeg_gan.yaml`. `reexpress()` / `quantize()` are ReBrain's `_common_step` forward.
  - The ResidualVQ is built with `channel_last=True`, and the (b, d, n) encoder features are transposed around it. The config says `channel_last=False`, but vector-quantize-pytorch 1.31.0's ResidualVQ fails on that ("Sizes of tensors must match": its residual buffer assumes channel-last). The transpose is exactly what the flag does inside each VQ layer.
  - vector-quantize-pytorch is pinned to **1.31.0**, the release current at ReBrain's commit date (PyPI: 1.31.0 on 2026-07-20, 1.31.1 on 2026-08-02).
- `tracks/t2_bci/scripts/t2_d423_fit.py`: ReBrain's Lightning `training_step` ported to a plain loop. Differences:
  - deepspeed `FusedAdam` → `torch.optim.AdamW` with the same arguments.
  - transformers' cosine-with-warmup schedule is re-implemented.
  - The BIDS / EDF loader → random 4 s windows of the prepared ds003029 runs (36 channels per window, as ReBrain's `n_channels=36`).
  - The discriminator step uses the detached reconstruction (what Lightning's `toggle_model` intends).
  - The generator step re-runs the discriminator after the discriminator update. ReBrain reuses the pre-update pass; the discriminator lr is 1e-7, so this is negligible.
- `t2_d423_prep.py` / `t2_d423_reexpress.py`: data preparation and re-expression of the PROTEUS export.
