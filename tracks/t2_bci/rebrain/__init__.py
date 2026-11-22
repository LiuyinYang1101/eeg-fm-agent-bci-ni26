"""T2-D423: vendored model / loss / balancer code of ReBrain (github.com/Xianglong-Bao/ReBrain, commit 41881e56dceaf9d070d9ea9264ea6b8f7bd7c01e,
Apache-2.0, see LICENSE_REBRAIN; ReBrain itself builds on EnCodec and BrainCodec, Apache-2.0). See NOTICE.md for what was changed.

build_codec() returns the (Encodec, ResidualVQ) pair of configs/ReBrain_train_ieeg_gan.yaml; the ResidualVQ comes from the
vector-quantize-pytorch package (MIT), imported lazily so that inference-only code paths can swap in a plain-torch lookup.
"""
import torch

from .encodec import Encodec, SEANetDecoder, SEANetEncoder

SR = 512          # the codec's sampling rate (ReBrain's config)
SEG_S = 4         # seconds per chunk
CFG = dict(n_filters=16, max_filters=256, dimension=64, kernel_size=3, last_kernel_size=3, causal=False,
           ratios=[2, 2, 2, 2, 2, 2], lstm=0, norm="none", true_skip=True)
RVQ = dict(dim=64, num_quantizers=4, codebook_size=256, decay=0.99, kmeans_init=True, kmeans_iters=800, rotation_trick=False,
           learnable_codebook=False, use_cosine_sim=False, channel_last=True, commitment_weight=1000, threshold_ema_dead_code=2)
# channel_last: ReBrain's config says False (features (b, d, n)); vector-quantize-pytorch 1.31.0's ResidualVQ breaks on that, so the quantizer is
# built channel-last and reexpress() transposes (b, d, n) <-> (b, n, d) itself, which is exactly what the flag does inside each layer.


def build_codec(num_quantizers=None):
    from vector_quantize_pytorch import ResidualVQ
    model = Encodec(encoder=SEANetEncoder(**CFG), decoder=SEANetDecoder(**CFG), sample_rate=SR, segment=SEG_S, overlap=0)
    rvq = dict(RVQ, num_quantizers=num_quantizers or RVQ["num_quantizers"])
    return model, ResidualVQ(**rvq)


def quantize(quantizer, h):
    """h: encoder features (b, d, n) -> (quantized (b, d, n), indices, commit loss); see the channel_last note above."""
    q, idx, commit = quantizer(h.transpose(1, 2))
    return q.transpose(1, 2), idx, commit


def reexpress(model, quantizer, x):
    """x: (n, L) single-channel segments at SR (µV) -> the codec's reconstruction (n, L); ReBrain's forward (_common_step)."""
    q, _, commit = quantize(quantizer, model.encode(x[:, None, :])[0])
    return model.decode(q)[:, 0, :]


@torch.no_grad()
def reexpress_eval(model, quantizer, x):
    model.eval(); quantizer.eval()
    return reexpress(model, quantizer, x)
