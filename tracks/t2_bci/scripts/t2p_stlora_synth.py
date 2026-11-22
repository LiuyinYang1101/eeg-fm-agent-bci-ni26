#!/usr/bin/env python
"""T2-D188: a package with SYNTHETIC per-user ST-L deltas, for the single-pass routed-LoRA profile (no training; timing + parity only).

  synth : copy a T2 package dir, regenerate its submission.py from the current template (t2p_mor_pkg.template_source), and give its ST
          member random r-rank deltas for every training user (K = the user_blend users) on every Linear inside the ST blocks
          (attn.qkv / attn.proj / mlp.fc1 / mlp.fc2); bank (K, N) fp16 -> weights_stlora.pt; config fm_members[j]["st_lora"].
  parity: load the original and the synthetic package through the template (as the kit does) on the first --n windows of --streams
          public streams, B = 1, fresh deepcopy + reset_state per stream, and report:
          (a) the synthetic package with its bank zeroed vs the original: probabilities equal (max |dp|, argmax agreement);
          (b) the synthetic package, graph path vs NI26_EAGER-style eager path (gpu_graph off): max |dp|, argmax agreement.
  python t2p_stlora_synth.py synth  --pkg <dir> --out <dir> [--r 16 --scale 0.5 --amp 0.01 --seed 0]
  python t2p_stlora_synth.py parity --orig <dir> --synth <dir> --export <export> [--n 150 --streams 3] [--out res.json]
"""
import argparse
import copy
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2] / "src"))
BLK = re.compile(r"^(enc\.vit\.blocks\.(\d+)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2))\.weight$")
ORDER = {"attn.qkv": 0, "attn.proj": 1, "mlp.fc1": 2, "mlp.fc2": 3}


def synth(a):
    import t2p_mor_pkg
    src, out = Path(a.pkg), Path(a.out)
    assert not out.exists(), out
    shutil.copytree(src, out)
    cfg = json.loads((out / "config.json").read_text())
    fams = [m.get("family", "st") for m in cfg["fm_members"]]
    j = fams.index("st")
    (out / "submission.py").write_text(t2p_mor_pkg.template_source(st=True, mor="mor" in fams))
    sd = torch.load(out / cfg["fm_members"][j]["file"], map_location="cpu", weights_only=True)
    layout = []
    for k, v in sd.items():
        mm = BLK.match(k)
        if mm:
            layout.append((int(mm.group(2)), ORDER[mm.group(3)], mm.group(1), int(v.shape[1]), int(v.shape[0])))
    layout = [(n, i, o) for _, _, n, i, o in sorted(layout)]
    assert layout, "no ST block Linear found"
    K = len(cfg["user_blend"]["users"])
    N = sum(a.r * i + o * a.r for _, i, o in layout)
    g = torch.Generator().manual_seed(a.seed)
    bank = (a.amp * torch.randn(K, N, generator=g)).half()
    torch.save(bank, out / "weights_stlora.pt")
    cfg["fm_members"][j]["st_lora"] = dict(file="weights_stlora.pt", r=a.r, scale=a.scale, layout=layout, synthetic=True, amp=a.amp, seed=a.seed)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[synth] {out}: ST member {j}, {len(layout)} adapted Linear, r {a.r}, K {K}, N {N} ({bank.numel() * 2 / 1e6:.1f} MB fp16)", flush=True)


def load(pkg, dev, meta, X, tag):
    from ni26 import kit
    kit.use_track("bci_decoding")
    spec = importlib.util.spec_from_file_location(f"t2p_par_{tag}", Path(pkg) / "submission.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, mod.Solver.__new__(mod.Solver).load_model(dict(submission_dir=Path(pkg), device=dev, ch_names=list(meta["ch_names"]),
                                                               sfreq=meta["sfreq"], n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3))


def run(m, X, groups, n, dev):
    P = []
    for rows in groups:
        mm = copy.deepcopy(m); mm.reset_state()
        with torch.inference_mode():
            for r in rows[:n]:
                mm.predict(torch.as_tensor(np.asarray(X[int(r)], np.float32)[None], device=dev))
                P.append(np.asarray(mm.last_p, np.float64))
    return np.stack(P)


def parity(a):
    import pandas as pd
    dev = "cuda"
    meta = json.loads((Path(a.export) / "meta.json").read_text())
    X = np.load(Path(a.export) / "X.npy", mmap_mode="r")
    idx = pd.read_csv(Path(a.export) / "index.csv", dtype={"subject": str, "session": str})
    groups = [g.sort_values("order")["i"].to_numpy() for _, g in list(idx.groupby("stream", sort=False))[: a.streams]]
    _, mo = load(a.orig, dev, meta, X, "orig")
    Po = run(mo, X, groups, a.n, dev); del mo; torch.cuda.empty_cache()
    _, ms = load(a.synth, dev, meta, X, "synth")
    Ps = run(ms, X, groups, a.n, dev)
    routed = [n for n in ms.fm_nets if hasattr(n, "delta")]
    assert len(routed) == 1, len(routed)
    bank = routed[0].bank.clone()
    routed[0].bank.zero_()
    ms.gshared = {"g": {}}                                   # recapture with the zeroed bank
    Pz = run(ms, X, groups, a.n, dev)
    routed[0].bank.copy_(bank)
    ms.gshared = {"g": {}, "off": True}                      # the eager fallback path (graphs off)
    Pe = run(ms, X, groups, a.n, dev)
    res = dict(windows=int(len(Po)),
               zero_vs_orig=dict(max_dp=float(np.abs(Pz - Po).max()), argmax_agree=float((Pz.argmax(1) == Po.argmax(1)).mean())),
               synth_vs_orig=dict(max_dp=float(np.abs(Ps - Po).max()), argmax_agree=float((Ps.argmax(1) == Po.argmax(1)).mean())),
               graph_vs_eager=dict(max_dp=float(np.abs(Ps - Pe).max()), argmax_agree=float((Ps.argmax(1) == Pe.argmax(1)).mean())))
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth")
    s.add_argument("--pkg", required=True); s.add_argument("--out", required=True)
    s.add_argument("--r", type=int, default=16); s.add_argument("--scale", type=float, default=0.5)
    s.add_argument("--amp", type=float, default=0.01); s.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("parity")
    p.add_argument("--orig", required=True); p.add_argument("--synth", required=True); p.add_argument("--export", required=True)
    p.add_argument("--n", type=int, default=150); p.add_argument("--streams", type=int, default=3); p.add_argument("--out", default="")
    a = ap.parse_args()
    {"synth": synth, "parity": parity}[a.cmd](a)


if __name__ == "__main__":
    main()
