#!/usr/bin/env python
"""Track 2: per-window runtime of a T2 package shape on THIS GPU, streamed B = 1 like Codabench (lead request 2026-10-15:
the worker is an RTX PRO 4500 Blackwell, not an A100). Starts from an existing zip (its ST-L member and config), rebuilds
submission.py from the CURRENT template (as t2p_repack), then scales the members to the requested shape with copies
(timing only: weights are copies / random, predictions are not scored):

  python t2p_timing.py --zip fda57e12.zip --export <export_proteus_v1> --fbc 20 --fbc-model fbcnet_bboth --sh 20 --st 3 \
      --users 10 [--windows 400] [--out timing.json]

Reports ms / window for each member family alone and the whole package end to end (window-to-prediction, incl. aligners,
the fingerprint routing and the prior correction), after 20 warm-up windows.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from ni26 import kit  # noqa: E402
from ni26 import train as T  # noqa: E402

TEMPLATE = REPO / "tracks" / "t2_bci" / "submission" / "t2p_submission.py"
STEEG_PT = REPO / "src" / "ni26" / "fm" / "steeg_pt.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--fbc", type=int, default=20)
    ap.add_argument("--fbc-model", default="fbcnet_bboth")
    ap.add_argument("--sh", type=int, default=20)
    ap.add_argument("--st", type=int, default=3)
    ap.add_argument("--users", type=int, default=10, help="per-user copies for the user blend (0 = off)")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-stack", action="store_true", help="time the per-member loop (no stacked forward)")
    ap.add_argument("--fp-every", type=int, default=1, help="fingerprint update every N windows (user blend)")
    args = ap.parse_args()
    dev = "cuda"
    d = Path(tempfile.mkdtemp(prefix="t2p_timing_"))
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(d)
    cfg = json.loads((d / "config.json").read_text())
    src = TEMPLATE.read_text()
    st_src = STEEG_PT.read_text()
    marker = next(ln for ln in src.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
    src = src.replace(marker, f"# ---- inlined: steeg_pt.py (sha256 {hashlib.sha256(st_src.encode()).hexdigest()[:12]})\n" + st_src)
    (d / "submission.py").write_text(src)
    kit.use_track("bci_decoding")
    spec = importlib.util.spec_from_file_location("t2p_timing_sub", d / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    E = Path(args.export)
    meta_e = json.loads((E / "meta.json").read_text())
    meta = dict(device=dev, ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]),
                n_times=480, n_classes=3, submission_dir=d)
    m = mod.Solver.__new__(mod.Solver).load_model(meta)
    # scale the members to the requested shape (timing only)
    fcfg = dict(cfg, braindecode_class=T.MODEL_SPECS[args.fbc_model][0],
                model_kwargs={k: ([list(b) for b in v] if k == "n_bands" else v) for k, v in T.MODEL_SPECS[args.fbc_model][1].items()})
    m.cfg = fcfg
    m.nets = torch.nn.ModuleList([mod.build_net(fcfg).eval() for _ in range(args.fbc)]).to(dev)
    if len(m.nets2):
        m.nets2 = torch.nn.ModuleList([copy.deepcopy(m.nets2[i % len(m.nets2)]) for i in range(args.sh)]).to(dev)
    if len(m.fm_nets):
        m.fm_nets = torch.nn.ModuleList([copy.deepcopy(m.fm_nets[i % len(m.fm_nets)]) for i in range(args.st)]).to(dev)
        m.fm_aligners = [copy.deepcopy(m.fm_aligners[0]) for _ in range(args.st)]
    if args.users:
        sys.path.insert(0, str(HERE))
        import t2p_route as RT
        X = np.load(E / "X.npy", mmap_mode="r")
        nf = RT.fingerprint(np.asarray(X[:4], np.float64), float(meta_e["sfreq"]), list(meta_e["ch_names"])).shape[1]
        rng = np.random.default_rng(0)
        m.pu_nets = torch.nn.ModuleList([torch.nn.ModuleList([mod.build_net(fcfg).eval() for _ in range(args.fbc)])
                                         for _ in range(args.users)]).to(dev)
        m.ub_r, m.ub_mean, m.ub_scale = 1 / 3, np.zeros(nf), np.ones(nf)
        m.ub_W, m.ub_mu = rng.normal(size=(nf, args.users - 1)), rng.normal(size=(args.users, args.users - 1))
    m = m.to(dev).eval()
    if not args.no_stack:
        m.stack_members()
    if args.users:
        m.ub_every = args.fp_every
    res_stack = dict(stacked=[x is not None for x in m.stk], stacked_pu=sum(x is not None for x in m.stk_pu))
    idx = pd.read_csv(E / "index.csv")
    rows = idx[idx["split"] == "test"].sort_values(["stream", "order"])["i"].to_numpy()[:args.windows + 20]
    X = np.load(E / "X.npy", mmap_mode="r")
    xs = [torch.as_tensor(np.asarray(X[int(r)])[None], device=dev) for r in rows]
    F = torch.nn.functional

    def timeit(fn, n):
        for x in xs[:20]:
            fn(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for x in xs[20:20 + n]:
            fn(x)
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1e3

    res = dict(gpu=torch.cuda.get_device_name(0), shape=dict(fbc=args.fbc, fbc_model=args.fbc_model, sh=len(m.nets2), st=len(m.fm_nets),
                                                              users=args.users, fp_every=args.fp_every), windows=args.windows)
    res.update(res_stack)
    xi0 = lambda x: x.detach().float().cpu().numpy()[0][m.idx]  # noqa: E731
    with torch.inference_mode():
        def fam(al, nets):                               # ONE aligner call per window, then every net of the family
            def f(x):
                z = torch.from_numpy(al(xi0(x)))[None].to(dev)
                for n in nets:
                    n(z)
            return f
        def famst(al, nets, stk):
            def f(x):
                z = torch.from_numpy(al(xi0(x)))[None].to(dev)
                m._fam(nets, stk, z)
            return f
        res["fbcnet_all_ms"] = timeit(famst(m.aligner, m.nets, m.stk[0]), args.windows)
        if len(m.nets2):
            res["shallow_ea_all_ms"] = timeit(famst(m.aligner2, m.nets2, m.stk[1]), args.windows)
            res["ea_aligner_ms"] = timeit(lambda x: m.aligner2(xi0(x)), args.windows)
        if len(m.fm_nets):
            def st(x):
                for net, al in zip(m.fm_nets, m.fm_aligners):
                    zf = torch.from_numpy(al(xi0(x)))[None].to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        F.softmax(net(zf).float(), dim=1)
            res["st_all_ms"] = timeit(st, args.windows)
            res["st_per_member_ms"] = res["st_all_ms"] / len(m.fm_nets)
        if args.users:
            m.reset_state()
            res["fingerprint_route_ms"] = timeit(lambda x: m._user(xi0(x)), args.windows)
            res["peruser_nets_ms"] = timeit(famst(m.aligner, m.pu_nets[0], m.stk_pu[0]), args.windows)

        def full(x):
            m.predict(x)
        m.reset_state()
        res["package_ms"] = timeit(full, args.windows)
    for k, v in res.items():
        if k.endswith("_ms"):
            res[k] = round(v, 2)
    print(json.dumps(res, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    shutil.rmtree(d)


if __name__ == "__main__":
    main()
