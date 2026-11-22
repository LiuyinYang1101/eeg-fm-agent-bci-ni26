#!/usr/bin/env python
"""T2-D544 stage 1b (registered 2026-11-16 20:03:46): weight interpolation of the pinned ST-L root and a t2p_preft expert, written in
use_cpt's contract so the usual fine-tune reads it (t2p_bench --fm-ckpt <out>/checkpoint-cpt.pth).
  W = alpha * root + (1 - alpha) * expert, every floating tensor of the encoder (enc.vit); non-floating tensors must be equal.
The root's vit state is the one make_encoder('st_large') builds from the pinned file (the same path t2p_preft starts from with
PREFT_INIT=root); keys and shapes are asserted equal to the expert's. cpt_summary.json = the expert's, with ckpt_sha256 = the new file's
and an 'interp' record (alpha, both shas). CPU only.
    python t2p_interp_ckpt.py --expert <run dir of the stage-1 job> --names-from <export_proteus_v1> --alpha 0.5 --out <dir>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2p_fm  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--expert", required=True)
ap.add_argument("--names-from", required=True)
ap.add_argument("--alpha", type=float, default=0.5, help="the root's weight")
ap.add_argument("--out", required=True)
A = ap.parse_args()
assert 0.0 < A.alpha < 1.0, A.alpha
ex = Path(A.expert)
summ = json.loads((ex / "cpt_summary.json").read_text())
assert summ.get("status") == "ok" and summ.get("fm") == "st_large", (summ.get("status"), summ.get("fm"))
ch = list(json.loads((Path(A.names_from) / "meta.json").read_text())["ch_names"])
enc = t2p_fm.make_encoder("st_large", ch, print, 0.0, "gain")
pinned = t2p_fm.t1.checkpoint_sha("st_large")[1]
assert summ.get("base_sha256") == pinned, "the expert must descend from the pinned root (PREFT_INIT=root)"
R = {k: v.detach().cpu().clone() for k, v in enc.vit.state_dict().items()}
E = torch.load(ex / "checkpoint-cpt.pth", map_location="cpu", weights_only=False)["model"]
assert set(R) == set(E), (sorted(set(R) ^ set(E))[:6])
W, n_f = {}, 0
for k, e in E.items():
    r = R[k]
    assert tuple(r.shape) == tuple(e.shape), (k, tuple(r.shape), tuple(e.shape))
    if e.is_floating_point():
        W[k] = (A.alpha * r.double() + (1.0 - A.alpha) * e.double()).to(e.dtype)
        n_f += 1
    else:
        assert torch.equal(r, e), k
        W[k] = e.clone()
dmax = max(float((W[k].double() - E[k].double()).abs().max()) for k in W if E[k].is_floating_point())
out = Path(A.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({"model": W}, out / "checkpoint-cpt.pth")
s_new = sha(out / "checkpoint-cpt.pth")
summ = dict(summ, ckpt_sha256=s_new, interp=dict(decision="T2-D544", alpha_root=A.alpha, root_sha256=pinned,
                                                 expert=str(ex / "checkpoint-cpt.pth"), expert_sha256=summ.get("ckpt_sha256"),
                                                 n_float_tensors=n_f, max_abs_move_from_expert=dmax))
(out / "cpt_summary.json").write_text(json.dumps(summ, indent=1))
print(f"T2-D544 interp: {n_f} floating tensors, alpha_root {A.alpha}, max |W - expert| {dmax:.4g}; wrote {out / 'checkpoint-cpt.pth'} sha256 {s_new[:12]}",
      flush=True)
