#!/usr/bin/env python
"""T2-D396 (P3 item 4a): SUPERVISED pretraining of a pooled CNN family on an external LABELLED corpus, then t2p_bench --cnn-init fine-tunes
it on PROTEUS with the shipped recipe (pretrain -> fine-tune; T2-D219 mixed the same corpus JOINTLY into training instead).
Data: t2p_bench.load_extra(--ext) (its 3-command label map, refused without one) mapped onto PROTEUS's montage and aligned per stream exactly
as the bench maps --extra for the CNNs (t2p_bench.aligned_extra: interp_matrix onto the export's ch_names, then align_stream with --align);
no PROTEUS window is used. Training: ni26.train.fit_eval with --recipe (the shipped pooled recipe: vp bank, rc, tilt where set) for its
own epochs; per-user re-colouring (rc) uses the corpus's own subjects. Saves the net's state_dict (t2p_bench --cnn-init loads every tensor
but final_layer.*) + a summary JSON.
  python t2p_cnn_p1.py --export PROTEUS --ext EXT --model fbcnet_bboth --align none --recipe R --out OUT.pt [--seed 0] [--device cuda]
T2-D398 multi-source mode: --pools DIR1,DIR2,... (labelled cpt_pools at PROTEUS's 120 Hz / 480-sample contract, 'event' labels): each pool's
windows (events >= 1 % of the pool, at most --cap per pool, uniform over its subjects, rng 0) mapped onto PROTEUS's montage and aligned per
stream exactly as t2p_bench.kd_aligned; ONE linear head per source on final_layer's input; the trunk trained with per-source-balanced
batches, plain CE, AdamW (lr 1e-3, wd 1e-2), a fixed --mepochs; saves the trunk's state_dict (final_layer never loaded by --cnn-init).
  python t2p_cnn_p1.py --export PROTEUS --pools P1,P2 --model M --align none --out OUT.pt [--cap 10000] [--mepochs 40]
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_bench as B  # noqa: E402
import ni26.train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="the PROTEUS export (its montage + rate are the target)")
    ap.add_argument("--ext", default="", help="an external LABELLED export with a 3-command label map (t2p_bench.load_extra)")
    ap.add_argument("--pools", default="", help="T2-D398: comma-separated labelled cpt_pools dirs (multi-source mode, one head per source)")
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--mepochs", type=int, default=40)
    ap.add_argument("--model", required=True)
    ap.add_argument("--align", required=True, choices=("none", "ea", "sz"))
    ap.add_argument("--recipe", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="smoke only (0 = the recipe's)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    _, _, meta = B.load_export(a.export)
    names = list(meta["ch_names"])
    if a.pools:
        return multi(a, names, float(meta["sfreq"]), t0)
    assert a.ext and a.recipe, "--ext and --recipe (single-source mode)"
    ext = B.load_extra(a.ext)
    Xk, yk = B.aligned_extra(ext, a.align, 0.05, dst_names=names)
    sub = ext["idx"].iloc[ext["keep"]]
    users = sorted(sub["subject"].unique())
    utr = sub["subject"].map({u: k for k, u in enumerate(users)}).to_numpy()
    assert Xk.shape[1:] == (len(names), 480) and set(np.unique(yk)) == {0, 1, 2}, (Xk.shape, np.unique(yk))
    rec = T.RECIPES[a.recipe] if not a.epochs else replace(T.RECIPES[a.recipe], epochs=a.epochs)
    T.VP_BANK.clear()
    T.VP_BANK.update({k: v.to(a.device) for k, v in B.vp_bank(names, K=rec.vp_k, sigma=0.005 * rec.vp, gain=rec.vp_g, shift=rec.vp_s,
                                                               vref=rec.vp_r, cap=rec.vp_cap).items()})
    n = min(256, len(Xk))
    cells = (sub["subject"] + "|" + sub["session"].astype(str)).to_numpy()
    data = dict(Xtr=Xk, ytr=yk, Xva=Xk[:0], yva=yk[:0], cva=cells[:0], Xte=Xk[:n], yte=yk[:n], cte=cells[:n], ext=[], utr=utr,
                str=sub["session"].astype(str).to_numpy())
    r = T.fit_eval(a.model, data, rec, a.seed, device=a.device, sfreq=float(meta["sfreq"]), log_every=max(1, rec.epochs // 6), return_state=True)
    sd = {k: v.detach().cpu() for k, v in r["state_last"].items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.out)
    summ = dict(ext=a.ext, ext_name=ext["name"], ext_sha=ext.get("sha"), n_windows=int(len(Xk)), n_users=len(users), classes=np.bincount(yk).tolist(),
                model=a.model, align=a.align, recipe=a.recipe, epochs=rec.epochs, seed=a.seed, seconds=round(time.time() - t0, 1),
                train_subset_acc_last=r.get("test_cell_bacc_last"))
    Path(a.out + ".json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"[cnn_p1] {a.model} on {ext['name']}: {len(Xk)} windows, {len(users)} users, classes {np.bincount(yk).tolist()} -> {a.out} "
          f"({time.time() - t0:.0f}s)", flush=True)


def multi(a, names, sfreq, t0):
    """T2-D398: multi-source supervised pretraining with one head per source (see the module docstring)."""
    import pandas as pd
    rng = np.random.default_rng(0)
    sel, info = [], []
    for d in [Path(x) for x in a.pools.split(",") if x]:
        meta = json.loads((d / "meta.json").read_text())
        idx = pd.read_csv(d / "index.csv", dtype=str, keep_default_na=False)
        X = np.load(d / "X.npy", mmap_mode="r")
        assert X.shape[-1] >= 480 and abs(float(meta["sfreq"]) - sfreq) < 1e-6, (d.name, X.shape, meta["sfreq"])
        ev = idx["event"].to_numpy()
        vc = pd.Series(ev).value_counts()
        cls = sorted(c for c, n in vc.items() if c and n >= 0.01 * len(ev))
        ok = np.flatnonzero(np.isin(ev, cls))
        subs = pd.unique(idx["subject"].to_numpy()[ok])
        per = max(1, a.cap // len(subs))
        rows = np.sort(np.concatenate([rng.choice(r, min(per, len(r)), replace=False) for r in (ok[idx["subject"].to_numpy()[ok] == u] for u in subs)]))
        sel.append(dict(name=d.name, X=X, idx=idx, names=list(meta["ch_names"]), rows=rows))
        info.append(dict(pool=d.name, classes=cls, n=int(len(rows)), subjects=int(len(subs))))
        print(f"[cnn_p1m] {d.name}: {len(rows)} windows, {len(subs)} subjects, classes {cls}", flush=True)
    XA, keys = B.kd_aligned(sel, a.align, 0.05, names)
    src = np.concatenate([np.full(len(p["rows"]), k) for k, p in enumerate(sel)])
    yv = np.concatenate([np.array([info[k]["classes"].index(e) for e in p["idx"]["event"].to_numpy()[p["rows"]]]) for k, p in enumerate(sel)])
    dev = a.device
    torch.manual_seed(a.seed)
    net = T.build_model(a.model, len(names), 3, 480, sfreq).to(dev)
    feat = {}
    h = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.__setitem__("f", inp[0]))
    Xt = torch.as_tensor(XA)
    net.eval()
    with torch.no_grad():
        net(Xt[:2].to(dev))
    dim = feat["f"].flatten(1).shape[1]
    heads = torch.nn.ModuleList([torch.nn.Linear(dim, len(i_["classes"])) for i_ in info]).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(heads.parameters()), lr=1e-3, weight_decay=1e-2)
    by_src = [np.flatnonzero(src == k) for k in range(len(sel))]
    steps = int(np.ceil(len(XA) / 128))
    for ep in range(a.mepochs):
        net.train(); hit, n = np.zeros(len(sel)), np.zeros(len(sel))
        for _ in range(steps):
            k = int(rng.integers(len(sel)))
            b = rng.choice(by_src[k], min(128, len(by_src[k])), replace=False)
            xb, yb = Xt[b].to(dev), torch.as_tensor(yv[b]).to(dev)
            net(xb)
            lo = heads[k](feat["f"].flatten(1))
            loss = torch.nn.functional.cross_entropy(lo, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            hit[k] += float((lo.argmax(1) == yb).sum()); n[k] += len(b)
        if ep % 10 == 9 or ep == a.mepochs - 1:
            print(f"[cnn_p1m] epoch {ep + 1}: train acc per source " + ", ".join(f"{i_['pool']} {hit[j] / max(n[j], 1):.3f}" for j, i_ in enumerate(info)), flush=True)
    h.remove()
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.out)
    Path(a.out + ".json").write_text(json.dumps(dict(pools=info, model=a.model, align=a.align, mepochs=a.mepochs, cap=a.cap, seed=a.seed,
                                                     n_windows=int(len(XA)), seconds=round(time.time() - t0, 1)), indent=1))
    print(f"[cnn_p1m] {a.model}: {len(XA)} windows from {len(sel)} sources -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
