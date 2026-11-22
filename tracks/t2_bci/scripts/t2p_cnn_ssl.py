#!/usr/bin/env python
"""T2-D367 (outlier B6-3): self-supervised pretraining (SimCLR) of a pooled CNN member on IN-DOMAIN unlabelled windows, per protocol.

Pool (never a window of the protocol's test users / streams; no labels anywhere):
  kit   = the export's train + val windows (t2p_bench.folds 'kit') + the session crops of the kit train + val users;
  seen1 = t2p_cpt.pool_keep('seen1') on the export and on the crops (P01-P03 P05-P08 every window + P04 / P10 session 1).
Input = exactly the bench's for this member: align 'none' = x / scale, scale = the median std of the protocol's TRAINING windows
(t2p_bench's formula); crops longer than the contract's 480 samples are cut at a random offset per draw.
Views = two independent draws of ni26.train.augment under the shipped recipe's bank (r1_pulxx_vp2_e180_vr_rc75 with rc off: the 10 mm vp
bank + virtual references + its gains / shifts; t2p_bench.vp_bank, the same seed as every bench job).
Model = ni26.train.build_model(model); features = final_layer's input (flattened); projection = Linear-ReLU-Linear (d -> 256 -> 128);
loss = NT-Xent (tau); AdamW, cosine; fixed budget (--epochs, --bs) set before any number. Saves the net's state_dict (final_layer
included but never used: t2p_bench --cnn-init loads every other tensor) + a summary JSON.
  python t2p_cnn_ssl.py --export E --crops DIR --protocol kit|seen1 --out OUT.pt [--model fbcnet_bboth] [--epochs 40] [--bs 256]
"""
import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
from t2p_cpt import pool_keep  # noqa: E402
import ni26.train as T  # noqa: E402

RECIPE = "r1_pulxx_vp2_e180_vr_rc75"


def pools(X, idx, crops, protocol):
    """(export rows, crop rows, scale) of a protocol's SSL pool; asserts no test-user / test-stream window enters."""
    (_, trm, vam, tem), = B.folds(idx, protocol)
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))     # t2p_bench's per-task scale
    ci = pd.read_csv(crops / "index.csv", dtype={"subject": str, "session": str})
    csub, cses = ci["subject"].str.split("/").str[-1].to_numpy(), ci["session"].astype(str).to_numpy()
    if protocol == "kit":
        rows = np.flatnonzero(trm | vam)
        users = set(idx["subject"].to_numpy()[rows])
        crow = np.flatnonzero(np.isin(csub, sorted(users)))
    else:
        rows = np.flatnonzero(pool_keep(idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy(), protocol))
        crow = np.flatnonzero(pool_keep(csub, cses, protocol))
    te_streams = set(idx["stream"].to_numpy()[tem])
    assert not tem[rows].any(), "a protocol test window in the SSL pool"
    assert not ({f"{s}|{e}" for s, e in zip(csub[crow], cses[crow])} & te_streams), "a crop of a protocol test stream in the SSL pool"
    if protocol == "kit":
        assert not (set(csub[crow]) & set(idx["subject"].to_numpy()[tem])), "a crop of a kit test user in the SSL pool"
    return rows, crow, scale


def nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--crops", required=True, help="the proteus_sessions_v1 dir (index.csv, X.npy, meta.json)")
    ap.add_argument("--protocol", required=True, choices=["kit", "seen1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="fbcnet_bboth")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests only (0 = the full budget)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    X, idx, meta = B.load_export(a.export)
    crops = Path(a.crops)
    cmeta = json.loads((crops / "meta.json").read_text())
    assert list(cmeta["ch_names"]) == list(meta["ch_names"]) and float(cmeta["sfreq"]) == float(meta["sfreq"]), "crops must be on the export's montage / rate"
    XC = np.load(crops / "X.npy", mmap_mode="r")
    rows, crow, scale = pools(X, idx, crops, a.protocol)
    L = int(meta["n_times"])
    assert XC.shape[1] == X.shape[1] and XC.shape[2] >= L, (XC.shape, X.shape)
    xe = torch.as_tensor(np.asarray(X[rows], np.float32)[:, :, :L] / scale)
    xc = torch.as_tensor(np.asarray(XC[crow], np.float32) / scale)
    print(f"[cnn_ssl] {a.protocol}: {len(rows)} export windows + {len(crow)} crops (length {XC.shape[2]}), scale {scale:.4g}, "
          f"model {a.model}, device {a.device}", flush=True)
    dev = torch.device(a.device)
    rec = replace(T.RECIPES[RECIPE], rc=0.0)
    T.VP_BANK.clear()
    T.VP_BANK.update({k: v.to(dev) for k, v in B.vp_bank(list(meta["ch_names"]), K=rec.vp_k, sigma=0.005 * rec.vp, gain=rec.vp_g, shift=rec.vp_s,
                                                       vref=rec.vp_r, cap=rec.vp_cap).items()})
    net = T.build_model(a.model, X.shape[1], 3, L, float(meta["sfreq"])).to(dev)
    feat = {}
    hook = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.__setitem__("f", inp[0]))
    with torch.no_grad():
        net.eval()
        net(xe[:2].to(dev))
    d = feat["f"].flatten(1).shape[1]
    proj = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)).to(dev)
    params = [p for n, p in net.named_parameters() if not n.startswith("final_layer.")] + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n = len(xe) + len(xc)
    spe = int(math.ceil(n / a.bs))
    total = a.epochs * spe if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    gen = torch.Generator().manual_seed(a.seed)
    curve, step, t0 = [], 0, time.time()
    net.train()
    for ep in range(a.epochs):
        perm = torch.randperm(n, generator=gen)
        tl, tn = 0.0, 0
        for i in range(0, n, a.bs):
            b = perm[i:i + a.bs]
            be, bc = b[b < len(xe)], b[b >= len(xe)] - len(xe)
            parts = [xe[be]]
            if len(bc):
                off = torch.randint(0, xc.shape[2] - L + 1, (len(bc),), generator=gen)
                parts.append(torch.stack([xc[j, :, o:o + L] for j, o in zip(bc.tolist(), off.tolist())]))
            xb = torch.cat(parts).to(dev)
            if len(xb) < 4:
                continue
            v1, v2 = T.augment(xb, rec), T.augment(xb, rec)
            net(v1)
            z1 = proj(feat["f"].flatten(1))
            net(v2)
            z2 = proj(feat["f"].flatten(1))
            loss = nt_xent(z1.float(), z2.float(), a.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tl, tn, step = tl + float(loss) * len(xb), tn + len(xb), step + 1
            if a.max_steps and step >= a.max_steps:
                break
        curve.append(round(tl / max(tn, 1), 4))
        print(f"[cnn_ssl] ep {ep + 1} loss {curve[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    hook.remove()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out)
    summ = dict(protocol=a.protocol, model=a.model, n_export=int(len(rows)), n_crops=int(len(crow)), scale=scale, epochs=a.epochs, bs=a.bs,
                lr=a.lr, wd=a.wd, tau=a.tau, seed=a.seed, steps=step, loss_curve=curve, recipe_views=RECIPE + " (rc off)",
                ln_2bs=math.log(2 * a.bs - 1), seconds=round(time.time() - t0))
    Path(str(out) + ".json").write_text(json.dumps(summ, indent=1))
    print(f"[cnn_ssl] saved {out} ({step} steps; loss {curve[0] if curve else float('nan')} -> {curve[-1] if curve else float('nan')}; "
          f"chance ln(2bs-1) = {math.log(2 * a.bs - 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
