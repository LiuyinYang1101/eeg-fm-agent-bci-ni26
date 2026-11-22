#!/usr/bin/env python
"""D87: an earlier in-house BCI project's Two-Stream Fusion (TSF) decoder in its CAUSAL form on the PROTEUS stream contract.

TSF: backbone outputs ⊕ per-channel log band powers -> LayerNorm ->
2-layer LSTM (hidden 128, dropout 0.1) -> Linear. "TSF-on" = unidirectional, so window t sees windows 1..t only; here the
state resets at every session = Codabench's stream reset. Trained on whole training SESSIONS in kit acquisition order
(the test-time sequence structure, PRIOR_WORK lesson 1) using OUT-OF-FOLD backbone outputs (bench protocols l2o / kit2:
the backbone that scores subject j never saw j nor the outer test subject).

Arms (D87): A0 backbone per-window; A1 A0 + causal output prior correction; A2 TSF-on (logits ⊕ band, LSTM);
A3 LSTM on logits only; A4 per-window MLP on logits ⊕ band; A5 = A2 with every training and test session's windows
randomly permuted (diagnostic only: what is left when the cue-order statistics are removed = session-level adaptation).

Usage: python t2p_tsf.py --export <dir with X.npy + index.csv> --runs <bench run dirs> --model fbcnet --align wz \
          --seeds 0,1,2 --protocol loso|kit [--csv out.csv]
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def band_features(X, fs, cache=None):
    """(N, C, T) -> (N, C * 5) log mean Welch power per band (window-local; 1 s segments, 50 % overlap), channel-major."""
    if cache and Path(cache).exists():
        return np.load(cache)
    from scipy.signal import welch
    out = []
    for s in range(0, len(X), 256):
        f, P = welch(np.asarray(X[s:s + 256], dtype=np.float64), fs=fs, nperseg=int(fs), axis=-1)
        out.append(np.stack([np.log(P[..., (f >= lo) & (f < hi)].mean(-1) + 1e-4) for lo, hi in BANDS], -1)
                   .reshape(P.shape[0], -1))
    F_ = np.concatenate(out).astype(np.float32)
    if cache:
        np.save(cache, F_)
    return F_


class SeqDecoder(nn.Module):
    """TSF registered spec: LayerNorm -> LSTM (uni) -> Dropout -> Linear; `seq=False` = the per-window MLP twin (A4)."""

    def __init__(self, d_in, n_out=3, hidden=128, layers=2, dropout=0.1, seq=True):
        super().__init__()
        self.seq = seq
        self.norm = nn.LayerNorm(d_in)
        if seq:
            self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        else:
            self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):                                  # x (B, T, D)
        h = self.norm(x)
        h = self.rnn(h)[0] if self.seq else self.mlp(h)
        return self.out(self.drop(h))


def train_decoder(seqs, d_in, seq, seed, epochs=40, bs=8, lr=1e-3, wd=1e-4, device="cpu"):
    """seqs: list of (feat (T, D), y (T,)) training sessions. Plain CE (PRIOR_WORK lesson 5)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SeqDecoder(d_in, seq=seq).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(seqs)), max(1, int(np.ceil(len(seqs) / bs)))):
            L = max(len(seqs[i][1]) for i in b)
            xb = torch.zeros(len(b), L, d_in, device=device)
            yb = torch.full((len(b), L), -100, dtype=torch.long, device=device)
            for r, i in enumerate(b):
                f, y = seqs[i]
                xb[r, :len(y)] = torch.as_tensor(f, device=device)
                yb[r, :len(y)] = torch.as_tensor(y, device=device)
            loss = F.cross_entropy(net(xb).reshape(-1, 3), yb.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net.eval()


@torch.no_grad()
def predict_seq(net, f, device="cpu"):
    """One session from a fresh state (= reset per stream); a unidirectional LSTM's output at t depends on 1..t only."""
    return F.softmax(net(torch.as_tensor(f, device=device)[None]), -1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--model", default="fbcnet")
    ap.add_argument("--align", default="wz")
    ap.add_argument("--rec", default="r1", help="the recipe field of the bench file names (5th '__' field)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--protocol", default="loso", choices=("loso", "kit"))
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4,A5")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    ex = Path(args.export)
    idx = pd.read_csv(ex / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    idx["stream"] = idx["subject"] + "|" + idx["session"]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    meta = json.loads((ex / "meta.json").read_text())
    X = np.load(ex / "X.npy", mmap_mode="r")
    band = band_features(X, float(meta["sfreq"]), cache=ex / "band5_log.npy")
    ordr, strm = idx["order"].to_numpy(), idx["stream"].to_numpy()

    # bench outputs: (fold, seed) -> full-length prob array (NaN where the fold did not score)
    probs = {}
    for d in args.runs.split(","):
        for f in glob.glob(str(Path(d) / f"*__{args.model}__{args.align}__n00__{args.rec}__s*.npz")):
            fold, seed = Path(f).name.split("__")[0], int(Path(f).stem.split("__s")[-1])
            z = np.load(f)
            P = np.full((len(idx), 3), np.nan, np.float32)
            P[z["test_idx"]] = z["prob_last"]
            probs[(fold, seed)] = P
    subs = sorted(idx["subject"].unique())
    if args.protocol == "loso":
        outer = [(k, f"loso_{k}", {j: f"l2o_{k}_{j}" for j in subs if j != k}) for k in subs]
    else:
        trv = sorted(idx.loc[idx["split"].isin(["train", "val"]), "subject"].unique())
        outer = [("kit", "kit2_outer", {j: f"kit2_{j}" for j in trv})]
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    res = defaultdict(dict)      # arm -> seed -> full-length prob array over the scored windows
    for seed in seeds:
        missing = [f for _, fo, inner in outer for f in [fo, *inner.values()] if (f, seed) not in probs]
        if missing:
            print(f"[t2p_tsf] seed {seed}: {len(missing)} folds missing (e.g. {missing[:3]}) -> skipped", flush=True)
            continue
        out = {a: np.full((len(idx), 3), np.nan) for a in arms}
        for k, fo, inner in outer:
            te_rows = np.flatnonzero(~np.isnan(probs[(fo, seed)][:, 0]))
            P_te = probs[(fo, seed)]
            if "A0" in arms:
                out["A0"][te_rows] = P_te[te_rows]
            if "A1" in arms:
                out["A1"][te_rows] = causal_prior_correction(P_te[te_rows], (strm[te_rows], ordr[te_rows]))
            feats = {"A2": lambda P, r: np.concatenate([np.log(P[r] + 1e-6), band[r]], 1),
                     "A5": lambda P, r: np.concatenate([np.log(P[r] + 1e-6), band[r]], 1),
                     "A3": lambda P, r: np.log(P[r] + 1e-6),
                     "A4": lambda P, r: np.concatenate([np.log(P[r] + 1e-6), band[r]], 1)}
            for a in [a for a in arms if a in feats]:
                seqs = []
                for j, fj in inner.items():
                    Pj = probs[(fj, seed)]
                    for st, g in idx[idx["subject"] == j].groupby("stream"):
                        r = g.sort_values("order").index.to_numpy()
                        if a == "A5":       # diagnostic: order destroyed within the session (sequence statistics removed)
                            r = np.random.default_rng(seed + 7).permutation(r)
                        seqs.append((feats[a](Pj, r), y[r]))
                net = train_decoder(seqs, seqs[0][0].shape[1], seq=(a != "A4"), seed=seed)
                for st in np.unique(strm[te_rows]):
                    r = idx.loc[te_rows][idx.loc[te_rows, "stream"] == st].sort_values("order").index.to_numpy()
                    if a == "A5":
                        r = np.random.default_rng(seed + 11).permutation(r)
                    out[a][r] = predict_seq(net, feats[a](P_te, r))
            print(f"[t2p_tsf] seed {seed} outer {k} done", flush=True)
        for a in arms:
            res[a][seed] = out[a]
    rows, per_sub = [], {}
    for a in arms:
        if not res[a]:
            continue
        sc, pb, ac, ens = [], [], [], None
        for seed, P in res[a].items():
            r = np.flatnonzero(~np.isnan(P[:, 0]))
            pr = P[r].argmax(1)
            sc.append(metrics.cell_balanced_accuracy(y[r], pr, cells[r])[0])
            pb.append(metrics.pooled_balanced_accuracy(y[r], pr))
            ac.append(float(np.mean(y[r] == pr)))
            ens = P[r] if ens is None else ens + P[r]
        e_cb, per = metrics.cell_balanced_accuracy(y[r], ens.argmax(1), cells[r])
        ps = defaultdict(list)
        for c, v in per.items():
            ps[c.split("|")[0]].append(v)
        per_sub[a] = {s: float(np.mean(v)) for s, v in ps.items()}
        rows.append(dict(arm=a, n_seeds=len(sc), cellBA=np.mean(sc), sd=np.std(sc, ddof=1) if len(sc) > 1 else np.nan,
                         pooledBA=np.mean(pb), acc=np.mean(ac), ens_cellBA=e_cb,
                         ens_pooledBA=metrics.pooled_balanced_accuracy(y[r], ens.argmax(1)),
                         ens_acc=float(np.mean(y[r] == ens.argmax(1))), seeds=" ".join(f"{v:.3f}" for v in sc), n_cells=len(per)))
    df = pd.DataFrame(rows)
    if "A0" in per_sub:
        df["d_A0"] = df["cellBA"] - df.loc[df["arm"] == "A0", "cellBA"].iloc[0]
        df["users_up"] = [f"{sum(per_sub[a][s] > per_sub['A0'][s] for s in per_sub['A0'])}/{len(per_sub['A0'])}" for a in df["arm"]]
    pd.set_option("display.width", 250)
    print(f"== {args.protocol} | backbone {args.model} {args.align} {args.rec}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
