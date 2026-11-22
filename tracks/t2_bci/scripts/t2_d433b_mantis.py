#!/usr/bin/env python
"""T2-D433 part (b) (amended row, before any (b) number): FROZEN MantisV2 (a general time-series FM, MIT; local weights
~/workspace/datasets/pretrained_tsfm/mantis_v2, 4.2 M parameters) per-channel embeddings + L2 logistic regression, as a member.
  dump   (project env)  causally whitened windows (t2_d433_rocket.whiten) -> DIR/white.npy
  embed  (eeg311: the `mantis` package lives there)  each channel of each window, linearly resized 480 -> 512, -> the last layer's CLS (256-d);
         DIR/mantis_emb.npy (n, 41, 256). No project import.
  fit    (project env)  41 x 256 features, StandardScaler + LR, C in {1e-5, 1e-4, 1e-3} by leave-one-user-out on the TRAINING users; bench-format
         files DIR/runs/{proto}__mantis__s{0..11}.npz (deterministic: one draw written 12 times, as T2-D328).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CS = (1e-5, 1e-4, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dump", "embed", "fit"))
    ap.add_argument("--export")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    if a.step == "embed":
        import torch
        import torch.nn.functional as F
        from mantis.architecture.version2 import MantisV2
        from safetensors.torch import load_file
        W = Path.home() / "workspace/datasets/pretrained_tsfm/mantis_v2"
        cfg = {k: v for k, v in json.loads((W / "config.json").read_text()).items() if k != "device"}
        net = MantisV2(**cfg, device="cpu")
        net.load_state_dict(load_file(str(W / "model.safetensors")), strict=True)
        net.eval()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        X = np.load(D / "white.npy", mmap_mode="r")
        n, C, T = X.shape
        out = np.empty((n, C, 256), np.float32)
        tu = net.transf_unit
        with torch.no_grad():
            for s in range(0, n, 16):
                x = torch.from_numpy(np.ascontiguousarray(X[s:s + 16])).reshape(-1, 1, T)
                x = F.interpolate(x, size=512, mode="linear", align_corners=False)
                z = net.tokgen_unit(x)
                h = torch.cat([z, tu.cls_token.expand(len(z), 1, -1)], dim=1)[:, None]
                for layer in tu.transformer.layers:
                    h = layer(h)
                out[s:s + 16] = h[:, 0, -1].reshape(-1, C, 256).numpy()
                if s % 320 == 0:
                    print(f"[d433b] embed {s} / {n}", flush=True)
        np.save(D / "mantis_emb.npy", out)
        print("[d433b] embed done", out.shape, flush=True)
        return
    sys.path.insert(0, str(HERE))
    import t2p_bench as B
    X, idx, meta = B.load_export(a.export)
    idx = idx.reset_index(drop=True)
    if a.step == "dump":
        from t2_d433_rocket import whiten
        np.save(D / "white.npy", whiten(X, idx))
        print("[d433b] dump done", flush=True)
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F_ = np.load(D / "mantis_emb.npy").reshape(len(idx), -1)
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].str[-3:].to_numpy()
    os.makedirs(D / "runs", exist_ok=True)
    for proto in ("kit", "seen1"):
        (fname, trm, vam, tem), = B.folds(idx, proto)
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        score = {}
        for c in CS:
            accs = []
            for u in sorted(set(subj[tr])):
                fit, hold = tr[subj[tr] != u], tr[subj[tr] == u]
                if len(set(y[fit])) < 3 or len(hold) == 0:
                    continue
                p = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(F_[fit], y[fit]).predict(F_[hold])
                accs.append(np.mean([np.mean(p[y[hold] == k] == k) for k in np.unique(y[hold])]))
            score[c] = float(np.mean(accs))
        cbest = max(score, key=score.get)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=cbest, max_iter=4000)).fit(F_[tr], y[tr])
        pte, pva = m.predict_proba(F_[te]).astype(np.float32), m.predict_proba(F_[va]).astype(np.float32)
        res = dict(protocol=proto, fold=fname, model="mantis", C=cbest, louo=score)
        for s in range(12):
            np.savez_compressed(D / f"runs/{proto}__mantis__s{s}.npz", test_idx=te, prob_last=pte, val_idx=va, prob_val_last=pva, prob_best=pte,
                                res=json.dumps(res))
        print(f"[d433b] {proto}: C {cbest} (LOUO {score}); window acc test {np.mean(pte.argmax(1) == y[te]):.3f} val {np.mean(pva.argmax(1) == y[va]):.3f}",
              flush=True)
    print("[d433b] fit done", flush=True)


if __name__ == "__main__":
    main()
