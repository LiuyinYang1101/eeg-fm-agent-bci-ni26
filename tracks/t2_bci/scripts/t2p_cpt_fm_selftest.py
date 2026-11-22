#!/usr/bin/env python
"""T2-D492 self-test of t2p_cpt_fm.py (GPU via gpu-submit, ~2 min): per fm (reve, labram) a 20-step CPT on the local PROTEUS export
(kit_tv, no extras) into <tmp>/<fm>, then the bench's path: t2p_fm.use_cpt(fm, ckpt) + make_encoder -> every backbone tensor equals the
CPT run's saved state, differs from the pinned file somewhere (the CPT moved it), and a forward on 4-s windows gives the (B, C, 4, D) grid.
    gpu-submit python tracks/t2_bci/scripts/t2p_cpt_fm_selftest.py --tmp ~/workspace/outputs/neural_interface_challenge/t2_d492/selftest
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    for fm in ("reve", "labram"):
        o = Path(a.tmp) / fm
        cmd = [sys.executable, "-u", str(HERE / "t2p_cpt_fm.py"), "--fm", fm, "--proteus", str(EXP), "--out", str(o), "--steps", "20",
               "--batch", "16", "--accum", "2", "--eval-every", "10", "--lr", "1e-4"]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
        assert r.returncode == 0, f"{fm}: CPT failed"
        import t2p_fm
        t1 = t2p_fm.t1
        names = list(__import__("json").loads((EXP / "meta.json").read_text())["ch_names"])
        pinned = t2p_fm.make_encoder(fm, names, print, 0.0, "gain")
        ck = o / ("model.safetensors" if fm == "reve" else "checkpoint-cpt.pth")
        t2p_fm.use_cpt(fm, ck, print)
        enc = t2p_fm.make_encoder(fm, names, print, 0.0, "gain")
        bb, bp = (enc.reve, pinned.reve) if fm == "reve" else (enc.lab, pinned.lab)
        if fm == "reve":
            from safetensors.torch import load_file
            saved = load_file(str(ck))
        else:
            saved = t1.labram_state(ck)
        sd, moved = bb.state_dict(), 0
        for k, v in sd.items():
            if fm == "reve" and k.startswith("final_layer."):
                continue
            s_ = saved[k][:, :v.shape[1]] if (fm == "labram" and k == "temporal_embedding") else saved[k]
            assert torch.equal(v.cpu(), s_.cpu()), f"{fm}: {k} differs from the CPT state"
            moved += int(not torch.equal(v.cpu(), bp.state_dict()[k].cpu()))
        assert moved > 0, f"{fm}: the loaded encoder equals the pinned one"
        x = torch.as_tensor(np.load(EXP / "X.npy", mmap_mode="r")[:4].astype(np.float32))
        x = x / float(np.median(x.std(-1)))
        g = enc(enc.prep(x), enc.n_layers)
        assert g.shape[:3] == (4, len(names), 4), g.shape
        print(f"SELFTEST {fm} OK: {moved} tensors moved by CPT, grid {tuple(g.shape)}; cpt_info {t1.CPT_INFO.get(fm, {}).get('sha256', '')[:12]}",
              flush=True)
    print("SELFTEST ALL OK", flush=True)


if __name__ == "__main__":
    main()
