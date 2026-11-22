#!/usr/bin/env python
"""Track 2 PROTEUS: train a CNN-ensemble package on fixed epochs, package it, and prove the package = the research path.

  python t2p_final.py --export <dir> --out <pkg dir> --model shallow --align ea --epochs 40 --seeds 0,1,2 \
      --train kit|all [--extra <ext export>] [--tta prior:1:3] [--n0 0 --shrink 0.05]

--train kit : the warm-up mirror package (trains on the kit TRAIN + VAL participants only, so the Codabench warm-up
              replay on P04 / P09 is an honest held-out number);
--train all : every PROTEUS participant (+ --extra), the sealed-phase package.
Writes <out>/{submission.py, config.json, weights.pt}, <out>.zip (files at the root) and <out>.final.json with:
  golden : vectorised research path (ni26.stream.align_stream + nets + t2p_report prior correction) on EVERY public
           stream (all 14; agreement also reported on the kit test and on P04 / P10's later sessions) vs the PACKAGED StreamModel fed one window at a time from deepcopy + reset_state per stream,
           as benchmark_utils.streaming does -> identical argmax required, max |dp| reported;
  score  : the golden predictions' cell-BA on the kit test split (the number the kit replay must reproduce).
Kit runs (Simulated + the warm-up replay through benchopt) are done by the slurm wrapper t2p_final.slurm.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from ni26 import kit, metrics, stream  # noqa: E402
import t2p_bench as B  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402

TEMPLATE = REPO / "tracks" / "t2_bci" / "submission" / "t2p_submission.py"
_tpl = lambda p: Path(p) if Path(p).is_absolute() else REPO / p     # noqa: E731  (T2-D352: a relative --template = in this repo / snapshot)
STEEG_PT = REPO / "src" / "ni26" / "fm" / "steeg_pt.py"
GUARD = REPO / "src" / "ni26" / "worker_guard"
GUARD_DISTS = GUARD / "dists_442edcd.txt"   # the worker image of kit 442edcd (make_dists.py)

_GUARDED = r"""
import copy, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch
out, dev = Path(sys.argv[1]), sys.argv[2]   # the kit passes a Path
spec = importlib.util.spec_from_file_location("submission", f"{out}/submission.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg = json.load(open(f"{out}/config.json"))
C, T = len(cfg["ch_names"]), int(cfg["n_times"])
m = mod.Solver.__new__(mod.Solver).load_model(dict(submission_dir=out, device=dev, ch_names=cfg["ch_names"],
    sfreq=cfg["sfreq"], n_chans=C, n_times=T, n_classes=3))
mm = copy.deepcopy(m); mm.reset_state()
g = torch.Generator().manual_seed(0)
t0 = time.time()
pr = [int(mm.predict((20 * torch.randn(1, C, T, generator=g)).to(dev))[0]) for _ in range(8)]
print("GUARD_OK", json.dumps(dict(preds=pr, s_per_window=(time.time() - t0) / 8)))
"""


def _versions():
    """the training stack, recorded in config.json (the template prints it beside the worker's; review T2 P1a)."""
    import importlib.metadata as md
    out = {}
    for d in ("torch", "braindecode", "numpy", "scipy", "mne"):
        try:
            out[d] = md.version(d)
        except Exception:
            out[d] = None
    return out


def guarded_check(out, dev):
    """Load the package and stream 8 windows in a FRESH interpreter under the worker-image import guard
    (src/ni26/worker_guard): an import the Codabench image cannot satisfy (timm, ni26, ...) fails here, not there."""
    root = kit.kit_root()
    env = dict(os.environ, NI26_WORKER_GUARD=str(GUARD_DISTS), NI26_WORKER_GUARD_ROOTS=f"{out}:{root}",
               PYTHONPATH=os.pathsep.join([str(GUARD), str(root / "tracks" / "bci_decoding")]
                                          + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    r = subprocess.run([sys.executable, "-c", _GUARDED, str(out), dev], env=env, capture_output=True, text=True)
    refused = sorted({ln.split("'import ")[1].split("'")[0] for ln in r.stderr.splitlines() if "worker-guard: refused" in ln})
    ok = r.returncode == 0 and "GUARD_OK" in r.stdout
    res = dict(ok=ok, refused_imports=refused, dists=GUARD_DISTS.name)
    if ok:
        res.update(json.loads(r.stdout.split("GUARD_OK", 1)[1]))
    else:
        print(r.stdout[-2000:], r.stderr[-4000:], sep="\n", flush=True)
    return res


def user_index(subjects):
    """T2-D232: the training windows' user index as t2p_bench builds it (k = rank of the subject among the sorted training subjects)."""
    users = sorted(set(subjects))
    return np.array([users.index(u) for u in subjects], dtype=np.int64)


def cnn2_recipe(recipe, recipe2, epochs2):
    """T2-D232: the --model2 pooled recipe (= --recipe unless --recipe2); T.VP_BANK is installed once from --recipe, so the virtual-participant
    fields must agree."""
    from dataclasses import replace
    from ni26 import train as T
    r1, r2 = T.RECIPES[recipe], T.RECIPES[recipe2 or recipe]
    assert all(getattr(r1, k) == getattr(r2, k) for k in ("vp", "vp_k", "vp_g", "vp_s", "vp_r", "vp_cap")), (recipe, recipe2, "vp bank differs")
    return replace(r2, epochs=epochs2)


def rc_ext_bank(path, ch_names, recipes):
    """backlog 65 (a) / T2-D295: install t2p_covbank.py's external covariance bank as the re-colouring targets (T.RC_EXT_BANK) iff one of the
    pooled recipes sets rc_ext, exactly as t2p_bench --rc-ext-bank; training-only (the package carries no bank); returns the bank's size (0 = none)."""
    from ni26 import train as T
    if not any(r.rc_ext for r in recipes):
        return 0
    assert path, "an rc_ext recipe needs --rc-ext-bank"
    bk = np.load(path)
    assert list(bk["ch_names"]) == list(ch_names), "the bank must be on the export's montage"
    T.RC_EXT_BANK["C"] = bk["C"]
    return len(bk["C"])


def fm_seed_knobs(spec, seeds):
    """T2-D333: '' -> {}; else the per-seed knobs {int seed: {vp: int, lp_epochs: int}} for seeds in the build (others refused)."""
    if not spec:
        return {}
    raw = json.loads(spec)
    out = {int(s): dict(k) for s, k in raw.items()}
    assert set(out) <= set(seeds), (sorted(out), seeds)
    for s, k in out.items():
        assert set(k) <= {"vp", "lp_epochs", "align", "lr_head", "epochs", "ckpt", "tilt", "car"}, (s, k)  # T2-D475e car  # T2-D360 align; T2-D380 lr_head; T2-D384 epochs; T2-D392 ckpt; T2-D400 tilt
        assert all(isinstance(v, int) and v >= 0 for kk, v in k.items() if kk not in ("align", "lr_head", "ckpt", "tilt")), (s, k)
        assert "ckpt" not in k or (isinstance(k["ckpt"], str) and k["ckpt"].endswith((".pth", ".pt"))), (s, k)   # T2-D392: this seed's CPT checkpoint
        assert "lr_head" not in k or (isinstance(k["lr_head"], (int, float)) and k["lr_head"] > 0), (s, k)
        assert "epochs" not in k or k["epochs"] > 0, (s, k)
        assert "tilt" not in k or (isinstance(k["tilt"], (int, float)) and 0 < k["tilt"] < 1), (s, k)   # T2-D400: the spectral tilt (unset = off)                       # T2-D384: this seed's fine-tune length (unset = --fm-epochs)
        assert k.get("align", "sz") in ("sz", "ea", "none"), (s, k)
        assert k.get("car", 0) in (0, 1), (s, k)     # T2-D475e: 1 = this seed trains on common-average-referenced windows (stream.car_windows)
    return out


def fm_knobs(fm_llrd, fm_rc, subjects_tr):
    """T2-D248: the ST fine-tune's optional knobs (T2-D247: re-colouring + layer-wise lr decay) as fit_eval_fm kwargs, and the training
    windows' user index the rc branch needs; both unset = ({}, {}) (the earlier build byte for byte)."""
    kw = {}
    if fm_llrd != 1.0:
        kw["llrd"] = float(fm_llrd)
    if fm_rc > 0:
        kw["rc"] = float(fm_rc)
    return kw, (dict(utr=user_index(subjects_tr)) if fm_rc > 0 else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="eegnet")
    ap.add_argument("--align", default="ea")
    ap.add_argument("--n0", type=float, default=0.0)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--recipe", default="r1")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train", default="kit", choices=("kit", "all"))
    ap.add_argument("--extra", default="")
    ap.add_argument("--tta", default="", help="'' or prior:alpha:n0")
    ap.add_argument("--fm-grl", type=float, default=None, help="T2-D649 (W29): the FM fine-tunes' subject-adversarial GRL at this lambda max (absent = off)")
    ap.add_argument("--fm-grl-lr", type=float, default=1e-3, help="T2-D649: the adversary's lr")
    ap.add_argument("--user-balance-pow", type=float, default=1.0, help="T2-D647 (W28): with --user-balance, weights = sessions^-p (1 = D630, bitwise)")
    ap.add_argument("--user-balance", action="store_true", help="T2-D638 W24' / T2-D642 W25 (the owner's typed line needed for any ship use): the "
                    "POOLED CNN nets of all three families AND the FM fine-tunes (--fm; t2p_fm.fit_eval_fm data['wtr']) train with T2-D630's user-balanced CE "
                    "(every training window weighs 1 / (its user's number of training sessions), = t2p_bench --user-balance, self-normalised per batch); the "
                    "per-user CNN copies fine-tune UNWEIGHTED from the balanced nets; each FM member records user_balance True; absent = bitwise as before")
    ap.add_argument("--train-protocol", default="", help="T2-D638 builder GOLDEN only: train on one t2p_bench fold's training rows (e.g. ss8_P01) "
                    "instead of --train; requires --golden-out")
    ap.add_argument("--golden-out", default="", help="T2-D638 builder GOLDEN only: after the MAIN family's pooled nets, save their prob_last on every "
                    "public window to this npz and exit (no copies / other families / FM / package)")
    ap.add_argument("--rc-ext-bank", default="", help="backlog 65 (a): t2p_covbank.py's npz, the re-colouring targets of an rc_ext --recipe / --recipe2")
    # ST-EEGFormer members (t2p_fm.fit_eval_fm, shipped through the pure-torch src/ni26/fm/steeg_pt.py)
    ap.add_argument("--fm", default="", help="'' (CNN only) or an ST name, e.g. st_large")
    ap.add_argument("--fm-seeds", default="0")
    ap.add_argument("--fm-align", default="sz")
    ap.add_argument("--fm-calib", default="gain")
    ap.add_argument("--fm-ckpt", default="", help="t2p_cpt.py checkpoint (CPT) instead of the pinned weights")
    ap.add_argument("--fm-epochs", type=int, default=30)
    ap.add_argument("--fm-drop-path", type=float, default=0.0)
    ap.add_argument("--fm-mixup", type=float, default=0.0, help="T2-D112a: Beta(a, a) mixup of the FM's training windows")
    ap.add_argument("--fm-aug", default="", help="T2-D557: the FM training augmentation 'g:s[:c]' = gain jitter : time shift (samples) [: channel "
                    "dropout] (t2p_bench --fm-aug, the same parser); '' = off (bitwise the old build)")
    ap.add_argument("--fm-recomb", default="", help="T2-D583: 'S:p' within-user within-class segment recombination of the FM training windows "
                    "(t2p_bench --fm-recomb; needs --fm-rc > 0 for the user index); '' = off (bitwise the old build)")
    ap.add_argument("--fm-phase", default="", help="T2-D586: 'p:amax' the shared-phase Fourier surrogate (t2p_bench --fm-phase); '' = off")
    ap.add_argument("--fm-cons", type=float, default=0.0, help="T2-D588: the two-view consistency weight (t2p_bench --fm-cons); 0 = off")
    ap.add_argument("--fm-wmargin", type=float, default=0.0, help="T2-D596: the WORD-margin loss weight (t2p_bench --fm-wmargin); 0 = off")
    ap.add_argument("--fm-llrd", type=float, default=1.0, help="T2-D248: layer-wise lr decay of the ST fine-tune (T2-D241 / D247; 1 = off)")
    ap.add_argument("--fm-rc", type=float, default=0.0, help="T2-D248: T2-D235's re-colouring of the ST's training windows (strength; 0 = off)")
    ap.add_argument("--fm-seed-knobs", default="", help="T2-D333: JSON {seed: {vp: k, lp_epochs: n}}, per-ST-seed fine-tune knobs on top of the "
                    "shared ones (a recipe-diverse ST slot); a vp seed trains through the bench's FM bank (t2p_bench's --fm-vp: vref 0, "
                    "the r1 defaults), as T2-D330 measured it; '' = every seed the shared recipe (the earlier build byte for byte)")
    ap.add_argument("--user-blend-every", type=int, default=1, help="fingerprint update every N windows (runtime)")
    ap.add_argument("--user-blend", type=float, default=0.0, help="T2-D124c: r of the blend toward the fingerprint-nearest "
                    "training user's per-user FT copy of the main CNN family (0 = off)")
    ap.add_argument("--fm-weight", type=float, default=0.2, help="output = (1 - w) CNN mean + w FM mean (D95: 4:1 = 0.2)")
    # T2-D140: our MOR as a second FM family (packaged by t2p_mor_pkg, which is not part of this public release); defaults =
    # the T2-D130 / D133 recipe (role-specific input-dropout LoRA r16, alpha 32, dropout 0.1, lr 1e-3, 12 epochs, wz + zwin)
    ap.add_argument("--fm2", default="", help="'' or the in-house MOR (mor_p1v2)")
    ap.add_argument("--fm2-seeds", default="0")
    ap.add_argument("--fm2-align", default="wz")
    ap.add_argument("--fm2-calib", default="zwin")
    ap.add_argument("--fm2-epochs", type=int, default=12)
    ap.add_argument("--fm2-lr-enc", type=float, default=1e-3)
    ap.add_argument("--fm2-lora-in", type=int, default=16)
    ap.add_argument("--fm2-lora-in-roles", type=int, default=1)
    ap.add_argument("--fm2-lora-in-dropout", type=float, default=0.1)
    ap.add_argument("--fm2-weight", type=float, default=0.1, help="the MOR's share of the OUTPUT (T2-D137 (c): 0.1)")
    # T2-D116: a second CNN family with its own alignment and output share (taken out of the CNN share)
    ap.add_argument("--model2", default="")
    ap.add_argument("--align2", default="ea")
    ap.add_argument("--n02", type=float, default=0.0)
    ap.add_argument("--seeds2", default="0,1,2,3,4")
    ap.add_argument("--epochs2", type=int, default=60)
    ap.add_argument("--weight2", type=float, default=0.2, help="share of the OUTPUT (T2-D116: 0.6 FBCNet / 0.2 ShallowNet / 0.2 ST-L)")
    # T2-D161: per-user FT copies of the second family too, routed by the main family's fingerprint (needs --user-blend > 0)
    ap.add_argument("--user-blend2", type=float, default=0.0, help="r of the --model2 blend toward the routed user's own --model2 "
                    "copies (0 = off; T2-D160: 0.5)")
    ap.add_argument("--recipe2", default="", help="the RECIPES entry whose pooled training makes the --model2 nets (default --recipe; "
                    "T2-D232: the vr line, as T2-D230's re-colouring is FBCNet only); its virtual-participant bank must be --recipe's")
    ap.add_argument("--template", default=str(TEMPLATE), help="T2-D352: the template file (default W9 = tracks/t2_bci/submission/t2p_submission.py; "
                    "a --model3 build needs W11 = t2p_submission_w11.py)")
    ap.add_argument("--model3", default="", help="T2-D352: a third CNN family (needs --template W11); '' = off = the earlier build byte for byte")
    ap.add_argument("--align3", default="ea")
    ap.add_argument("--n03", type=float, default=0.0)
    ap.add_argument("--seeds3", default="0,1,2,3,4")
    ap.add_argument("--epochs3", type=int, default=180)
    ap.add_argument("--weight3", type=float, default=0.1, help="T2-D352: the third family's share of the OUTPUT")
    ap.add_argument("--recipe3", default="", help="T2-D352: the third family's pooled recipe (its vp bank must equal --recipe's)")
    ap.add_argument("--user-blend3", type=float, default=0.0, help="T2-D352: r of the --model3 blend toward the routed user's own --model3 copies")
    ap.add_argument("--recipe3-pu", default="", help="T2-D352: the recipe whose pu_lr / pu_epochs train the --model3 copies (default --recipe3)")
    ap.add_argument("--recipe2-pu", default="", help="the RECIPES entry whose pu_lr / pu_epochs train the --model2 copies "
                    "(default --recipe; T2-D159: r1_pulx); the main family's copies always take --recipe's")
    ap.add_argument("--fm-micro", type=int, default=1, help="T2-D475e: gradient accumulation over k chunks per ST batch (t2p_fm's exact TA568 "
                    "path, as t2p_bench --fm-micro; memory / k, for a 32 GB GPU); 1 = off = the earlier build byte for byte")
    ap.add_argument("--fm-donor", action="store_true", help="T2-D475e: build PREPARATION only: the package is a donor of its FM member files "
                    "(CAR seeds need template W14, which the build template lacks, so the golden check is reported, not enforced); never a candidate")
    ap.add_argument("--fm-side", default="", choices=["", "dcmean"],
                    help="T2-D524 / D531 (build PREPARATION): every ST seed trains with the window-mean side head (t2p_fm side='dcmean': a zero-init "
                    "Linear(82, K) of the causal-z window means at the logits; z = t2p_bench.side_z over the DELIVERED windows with the training "
                    "rows' prior, n0 10 = side.pt's). The head is saved in its OWN file fm_<i>_side.pt (weights_only-safe), so fm_<i>.pt keeps the "
                    "member's exact keys and every earlier template loads it; donor builds only (no template reads the head yet)")
    ap.add_argument("--fm-init-zip", default="", help="T2-D546 (b) build PREPARATION: a package whose ST members are CONTINUED instead of "
                    "trained from the CPT checkpoint (donor builds only): each --fm-seeds seed starts from the member --fm-init-map names "
                    "(fp16 weights upcast to fp32, exact; its side head included), then --fm-cont-epochs epochs with every lr x --fm-cont-lr, "
                    "no warm-up, cosine to 0, data seed s + 50,000 = t2p_bench --fm-cont-arms b exactly; '' = off = the earlier build byte for byte")
    ap.add_argument("--fm-init-map", default="", help="T2-D546: 'seed:fm_<i>.pt,...', one entry per --fm-seeds seed; the member's own seed must match")
    ap.add_argument("--fm-cont-epochs", type=int, default=25, help="T2-D546: continuation length (0 = the stored member through the same path)")
    ap.add_argument("--fm-cont-lr", type=float, default=0.1, help="T2-D546: lr multiplier of every group (lr_enc / lr_head defaults x this)")
    args = ap.parse_args()
    assert args.user_balance or args.user_balance_pow == 1.0, "T2-D647 (Track3): --user-balance-pow without --user-balance would be ignored silently"
    if args.fm_init_zip and not args.fm_donor:   # T2-D546: continued members are build PREPARATION; the merge into a package is its own step
        raise SystemExit("!! --fm-init-zip is a donor build only (--fm-donor)")
    if bool(args.fm_init_zip) != bool(args.fm_init_map):
        raise SystemExit("!! --fm-init-zip and --fm-init-map go together")
    if args.fm_side and not args.fm_donor:   # T2-D524: no approved template adds the head -> a package would silently drop it
        raise SystemExit("!! --fm-side is a donor build only (--fm-donor): no approved template reads the FM side head")
    _car = any(k_.get("car") for k_ in (json.loads(args.fm_seed_knobs).values() if args.fm_seed_knobs else []))
    if _car and not args.fm_donor:       # T2-D475e: no template before W14 feeds a car member CAR input -> refuse before any training
        raise SystemExit("!! car seeds need template W14 (not approved / not built); a donor build needs --fm-donor")

    import torch
    from dataclasses import replace
    from ni26 import train as T
    t0 = time.time()
    ksha = kit.kit_sha()                       # review T2 P1c: package against the pinned kit only (sofia = 442edcd)
    if ksha != kit.KIT_SHA:
        raise SystemExit(f"!! kit {kit.kit_root()} is at {ksha}, not the pinned {kit.KIT_SHA}")
    X, idx, meta = B.load_export(args.export)
    ckpts_ = [args.fm_ckpt] + ([k_["ckpt"] for k_ in json.loads(args.fm_seed_knobs).values() if "ckpt" in k_] if args.fm_seed_knobs else [])   # T2-D392
    if args.fm and args.train == "kit":   # review T2 P2e: a kit package's FM never pretrained on the kit test (every per-seed checkpoint too)
        import t2p_fm as _fm
        for ck_ in [c for c in ckpts_ if c]:
            leak = _fm.cpt_leak(ck_, B.protocol_test_streams(idx, "kit"))
            if leak is None or leak:
                raise SystemExit(f"!! --train kit --fm-ckpt {ck_}: kit-test streams in its pools {leak} (None = no pool record)")
    y = idx["y"].to_numpy().astype(int)
    trm = idx["split"].isin(["train", "val"]).to_numpy() if args.train == "kit" else np.ones(len(idx), bool)
    if args.train_protocol:              # T2-D638 builder golden: one t2p_bench fold's training rows (its own alignment scale, as t2p_bench)
        assert args.golden_out and not args.extra, "--train-protocol is for the builder golden only (--golden-out, no --extra)"
        (_, trm, _, tem_gp_), = B.folds(idx, args.train_protocol)
    assert not args.golden_out or args.train_protocol, "--golden-out goes with --train-protocol"
    tem = (idx["split"] == "test").to_numpy()
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    prior = None
    if args.align == "ea" and args.n0 > 0:
        tr = np.flatnonzero(trm)
        sel = tr[np.linspace(0, len(tr) - 1, min(len(tr), 3000)).astype(int)]
        xs = np.asarray(X[sel], dtype=np.float64)
        prior = stream.trace_normalised_mean(stream.window_covs(xs - xs.mean(-1, keepdims=True)))
    XA = B.aligned_all(X, idx, args.align, trm, args.n0, args.shrink, scale)
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    wub = {}                              # T2-D638 W24': the user-balanced loss weights of the pooled CNN nets ({} = absent, bitwise)
    if args.user_balance:                # = t2p_bench --user-balance (T2-D630): 1 / (the user's number of TRAINING sessions) per training window
        assert not args.extra, "--user-balance with --extra: not wired"
        sub_tr_ = idx.iloc[tr]["subject"].astype(str).to_numpy()
        ses_tr_ = idx.iloc[tr]["session"].astype(str).to_numpy()
        ns_ = {u: len(set(ses_tr_[sub_tr_ == u])) for u in np.unique(sub_tr_)}
        pw_ = float(args.user_balance_pow)                # T2-D647 (W28): sessions^-p; p = 1 keeps D630's 1 / sessions (bitwise)
        wu_ = {u: (1.0 / n if pw_ == 1.0 else float(n) ** (-pw_)) for u, n in ns_.items()}
        wub = dict(wtr=np.array([wu_[u] for u in sub_tr_], np.float32))
        print(f"[t2p_final] T2-D638 user-balanced pooled CNN loss weights: {({u.split('/')[-1]: round(w_, 4) for u, w_ in wu_.items()})}"
              + (f" (T2-D647 power {pw_:g})" if pw_ != 1.0 else ""), flush=True)
    ev = np.arange(len(idx))   # golden rows = EVERY public window (all 14 streams; T2-D116 lead: later sessions too)
    if args.train_protocol:              # T2-D638 builder golden (Track3): predict EXACTLY the fold's test rows in the bench's order (the same
        ev = np.flatnonzero(tem_gp_)      # inference batch partition as t2p_bench, so a deterministic net can be bitwise)
    Xtr, ytr = XA[tr], y[tr]
    extras = [B.load_extra(d) for d in args.extra.split(",") if d]
    for e in extras:
        xe, ye = B.aligned_extra(e, args.align, args.shrink, list(meta["ch_names"]))
        Xtr, ytr = np.concatenate([Xtr, xe]), np.concatenate([ytr, ye])
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    recipe = replace(T.RECIPES[args.recipe], epochs=args.epochs)
    rc_u = {}
    if recipe.rc > 0 or recipe.user_adv > 0:   # T2-D232: T2-D230's re-colouring maps (T2-D293: the user adversary) need the training windows' user index (t2p_bench's: sorted training users)
        assert not extras, "--recipe with rc and --extra: not wired"
        rc_u = dict(utr=user_index(idx["subject"].to_numpy()[tr]))
        print(f"[t2p_final] re-colouring rc {recipe.rc}: {int(rc_u['utr'].max()) + 1} training users", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if recipe.vp:                        # T2-D196: T2-D195's virtual participants (t2p_bench.vp_bank, the same 64 draws) for BOTH families' pooled nets
        T.VP_BANK.update({k_: v_.to(dev) for k_, v_ in B.vp_bank(list(meta["ch_names"]), K=recipe.vp_k, sigma=0.005 * recipe.vp, gain=recipe.vp_g, shift=recipe.vp_s, vref=recipe.vp_r, cap=recipe.vp_cap).items()})
        print(f"[t2p_final] virtual participants: {len(T.VP_BANK['mats'])} draws, jitter {5 * recipe.vp} mm", flush=True)
    nb = rc_ext_bank(args.rc_ext_bank, meta["ch_names"], [recipe] + ([T.RECIPES[args.recipe2]] if args.recipe2 else []))
    if nb:                               # backlog 65 (a): before either family's pooled training
        print(f"[t2p_final] external re-colouring bank: {nb} subjects", flush=True)
    states, probs = [], []
    for s in [int(v) for v in args.seeds.split(",")]:
        data = dict(Xtr=Xtr, ytr=ytr, Xva=XA[:0], yva=y[:0], cva=cells[:0], Xte=XA[ev], yte=y[ev], cte=cells[ev], **rc_u, **wub)   # T2-D638: wub
        r = T.fit_eval(args.model, data, recipe, s, device=dev, sfreq=float(meta["sfreq"]), log_every=10, return_state=True)
        states.append({k: v.float() if v.is_floating_point() else v for k, v in r["state_last"].items()})
        probs.append(r["prob_last"])
        if args.train_protocol:          # T2-D638 golden: prob_last covers the fold's test rows only (ev)
            print(f"[t2p_final] seed {s}: {args.train_protocol} test cellBA {metrics.cell_balanced_accuracy(y[ev], r['prob_last'].argmax(1), cells[ev])[0]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"[t2p_final] seed {s}: kit-test cellBA {metrics.cell_balanced_accuracy(y[te], r['prob_last'][te].argmax(1), cells[te])[0]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if args.golden_out:                  # T2-D638 builder golden: the MAIN family's pooled nets only, then exit
        np.savez(args.golden_out, prob=np.stack(probs).astype(np.float32), seeds=np.array([int(v) for v in args.seeds.split(",")]), rows=ev,
                 model=args.model, recipe=args.recipe, align=args.align, user_balance=bool(args.user_balance), train_protocol=args.train_protocol)
        print(f"[t2p_final] T2-D638 builder golden saved: {args.golden_out} ({args.model}, {args.recipe}, seeds {args.seeds}, "
              f"user_balance {bool(args.user_balance)}, fold {args.train_protocol})", flush=True)
        raise SystemExit(0)
    pu_states, Ppu, kstar, ub_cfg = [], None, None, None
    if args.user_blend > 0:              # T2-D124c: per-user FT copies + the fingerprint state (t2p_route.fingerprint)
        assert not extras, "--user-blend with --extra: not wired"
        import t2p_route as RT
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        subj = idx["subject"].to_numpy()
        users = sorted(set(subj[tr]))
        Fp = RT.fingerprint(np.asarray(X, np.float64), float(meta["sfreq"]), list(meta["ch_names"]))
        sc_ = StandardScaler().fit(Fp[tr])
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(sc_.transform(Fp[tr]), subj[tr])
        assert list(lda.classes_) == users
        W = lda.scalings_[:, :lda._max_components]
        Ztr = sc_.transform(Fp[tr]) @ W
        mu = np.stack([Ztr[subj[tr] == u].mean(0) for u in users])
        Xt = torch.as_tensor(np.ascontiguousarray(XA[tr]), device=dev)
        yt = torch.as_tensor(y[tr], dtype=torch.long, device=dev)
        Xev = torch.as_tensor(np.ascontiguousarray(XA[ev]), device=dev)
        Ppu = np.zeros((len(users), len(ev), 3))
        pu_states = [[] for _ in users]
        for s, sd in zip([int(v) for v in args.seeds.split(",")], states):
            net = T.build_model(args.model, X.shape[1], 3, X.shape[2], float(meta["sfreq"])).to(dev)
            net.load_state_dict(sd)
            for k, u in enumerate(users):
                m_ = torch.as_tensor(subj[tr] == u, device=dev)
                nu = T.finetune(net, Xt[m_], yt[m_], T.peruser_for(recipe), s * 100 + k)
                pu_states[k].append({kk: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for kk, v in nu.state_dict().items()})
                Ppu[k] += T.predict_proba(nu, Xev) / len(states)
        Zev = sc_.transform(Fp[ev]) @ W
        kstar = np.empty(len(ev), int)
        import t2p_fpcalib
        for g in B.stream_groups(idx, ev):            # the template's causal running mean (current window included)
            kstar[g] = t2p_fpcalib.stream_route_every(Zev[g], mu, args.user_blend_every)
        ub_cfg = dict(r=args.user_blend, users=users, file="weights_pu.pt", mean=sc_.mean_.tolist(), scale=sc_.scale_.tolist(),
                      W=W.tolist(), mu=mu.tolist(), n_seeds=len(states), every=args.user_blend_every)
        late = (idx["subject"].isin(["P04", "P10"]) & idx["session"].astype(str).isin(["2", "3"])).to_numpy()
        print(f"[t2p_final] user blend r {args.user_blend}: {len(users)} users x {len(states)} seeds per-user copies; later sessions "
              f"P04 / P10 routed to self {float(np.mean(np.array(users)[kstar[late]] == subj[late])):.3f} ({time.time() - t0:.0f}s)", flush=True)
    states2, probs2, prior2 = [], [], None
    if args.model2:
        assert not extras, "--model2 with --extra: not wired"
        if args.align2 == "ea" and args.n02 > 0:
            tr_ = np.flatnonzero(trm)
            sel = tr_[np.linspace(0, len(tr_) - 1, min(len(tr_), 3000)).astype(int)]
            xs = np.asarray(X[sel], dtype=np.float64)
            prior2 = stream.trace_normalised_mean(stream.window_covs(xs - xs.mean(-1, keepdims=True)))
        XA2 = B.aligned_all(X, idx, args.align2, trm, args.n02, args.shrink, scale)
        recipe2 = cnn2_recipe(args.recipe, args.recipe2, args.epochs2)
        rc_u2 = dict(utr=user_index(idx["subject"].to_numpy()[tr])) if recipe2.rc > 0 or recipe2.user_adv > 0 else {}   # T2-D293: + the user adversary
        for s in [int(v) for v in args.seeds2.split(",")]:
            data = dict(Xtr=XA2[tr], ytr=y[tr], Xva=XA2[:0], yva=y[:0], cva=cells[:0], Xte=XA2[ev], yte=y[ev], cte=cells[ev], **rc_u2, **wub)   # T2-D638
            r = T.fit_eval(args.model2, data, recipe2, s, device=dev, sfreq=float(meta["sfreq"]), log_every=10, return_state=True)
            states2.append({k: v.float() if v.is_floating_point() else v for k, v in r["state_last"].items()})
            probs2.append(r["prob_last"])
            print(f"[t2p_final] {args.model2} seed {s}: kit-test cellBA "
                  f"{metrics.cell_balanced_accuracy(y[te], r['prob_last'][te].argmax(1), cells[te])[0]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pu_states2, Ppu2, ub2_cfg = [], None, None
    if args.user_blend2 > 0:             # T2-D161: the second family's per-user copies, routed by the SAME fingerprint (kstar)
        assert Ppu is not None and states2, "--user-blend2 needs --user-blend > 0 (the shared route) and --model2"
        rec2_pu = args.recipe2_pu or args.recipe
        rpu2 = T.peruser_for(T.RECIPES[rec2_pu])
        Xt2 = torch.as_tensor(np.ascontiguousarray(XA2[tr]), device=dev)
        Xev2 = torch.as_tensor(np.ascontiguousarray(XA2[ev]), device=dev)
        Ppu2 = np.zeros((len(users), len(ev), 3))
        pu_states2 = [[] for _ in users]
        for s, sd in zip([int(v) for v in args.seeds2.split(",")], states2):
            net = T.build_model(args.model2, X.shape[1], 3, X.shape[2], float(meta["sfreq"])).to(dev)
            net.load_state_dict(sd)
            for k, u in enumerate(users):
                m_ = torch.as_tensor(subj[tr] == u, device=dev)
                nu = T.finetune(net, Xt2[m_], yt[m_], rpu2, s * 100 + k)
                pu_states2[k].append({kk: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for kk, v in nu.state_dict().items()})
                Ppu2[k] += T.predict_proba(nu, Xev2) / len(states2)
        ub2_cfg = dict(r=args.user_blend2, users=users, file="weights2_pu.pt", n_seeds=len(states2), recipe_pu=rec2_pu,
                       pu_lr=rpu2.lr, pu_epochs=rpu2.epochs)
        print(f"[t2p_final] user blend 2 r {args.user_blend2}: {len(users)} users x {len(states2)} {args.model2} copies "
              f"({rec2_pu}: lr {rpu2.lr:g}, {rpu2.epochs} epochs; {time.time() - t0:.0f}s)", flush=True)
    states3, probs3, prior3, pu_states3, Ppu3, ub3_cfg = [], [], None, [], None, None
    if args.model3:                      # T2-D352: the third CNN family (W11), as the second: own aligner, pooled nets, routed copies
        assert "cnn3" in _tpl(args.template).read_text() and states2, "--model3 needs --template W11 and --model2"
        assert not extras, "--model3 with --extra: not wired"
        if args.align3 == "ea" and args.n03 > 0:
            tr_ = np.flatnonzero(trm)
            sel = tr_[np.linspace(0, len(tr_) - 1, min(len(tr_), 3000)).astype(int)]
            xs = np.asarray(X[sel], dtype=np.float64)
            prior3 = stream.trace_normalised_mean(stream.window_covs(xs - xs.mean(-1, keepdims=True)))
        XA3 = B.aligned_all(X, idx, args.align3, trm, args.n03, args.shrink, scale)
        recipe3 = cnn2_recipe(args.recipe, args.recipe3, args.epochs3)
        rc_u3 = dict(utr=user_index(idx["subject"].to_numpy()[tr])) if recipe3.rc > 0 or recipe3.user_adv > 0 else {}
        for s in [int(v) for v in args.seeds3.split(",")]:
            data = dict(Xtr=XA3[tr], ytr=y[tr], Xva=XA3[:0], yva=y[:0], cva=cells[:0], Xte=XA3[ev], yte=y[ev], cte=cells[ev], **rc_u3, **wub)   # T2-D638
            r = T.fit_eval(args.model3, data, recipe3, s, device=dev, sfreq=float(meta["sfreq"]), log_every=10, return_state=True)
            states3.append({k: v.float() if v.is_floating_point() else v for k, v in r["state_last"].items()})
            probs3.append(r["prob_last"])
            print(f"[t2p_final] {args.model3} seed {s}: kit-test cellBA "
                  f"{metrics.cell_balanced_accuracy(y[te], r['prob_last'][te].argmax(1), cells[te])[0]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if args.user_blend3 > 0:
            assert Ppu is not None, "--user-blend3 needs --user-blend > 0 (the shared route)"
            rec3_pu = args.recipe3_pu or args.recipe3 or args.recipe
            rpu3 = T.peruser_for(T.RECIPES[rec3_pu])
            Xt3 = torch.as_tensor(np.ascontiguousarray(XA3[tr]), device=dev)
            Xev3 = torch.as_tensor(np.ascontiguousarray(XA3[ev]), device=dev)
            Ppu3 = np.zeros((len(users), len(ev), 3))
            pu_states3 = [[] for _ in users]
            for s, sd in zip([int(v) for v in args.seeds3.split(",")], states3):
                net = T.build_model(args.model3, X.shape[1], 3, X.shape[2], float(meta["sfreq"])).to(dev)
                net.load_state_dict(sd)
                for k, u in enumerate(users):
                    m_ = torch.as_tensor(subj[tr] == u, device=dev)
                    nu = T.finetune(net, Xt3[m_], yt[m_], rpu3, s * 100 + k)
                    pu_states3[k].append({kk: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for kk, v in nu.state_dict().items()})
                    Ppu3[k] += T.predict_proba(nu, Xev3) / len(states3)
            ub3_cfg = dict(r=args.user_blend3, users=users, file="weights3_pu.pt", n_seeds=len(states3), recipe_pu=rec3_pu,
                           pu_lr=rpu3.lr, pu_epochs=rpu3.epochs)
            print(f"[t2p_final] user blend 3 r {args.user_blend3}: {len(users)} users x {len(states3)} {args.model3} copies "
                  f"({rec3_pu}: lr {rpu3.lr:g}, {rpu3.epochs} epochs; {time.time() - t0:.0f}s)", flush=True)
    fm_states, fm_probs, fm_members = [], [], []
    fm_side_files = {}                   # T2-D524: {file: side-head blob} (empty = the earlier build byte for byte)
    if args.fm:
        assert args.fm.startswith("st_"), "--fm: ST-EEGFormer only (the pure-torch module covers ST)"
        assert not extras, "--fm with --extra: not wired"
        import t2p_fm
        from ni26.fm import steeg_pt
        XF = B.aligned_all(X, idx, args.fm_align, trm, 0.0, args.shrink, scale)
        fm_kw, fm_u = fm_knobs(args.fm_llrd, args.fm_rc, idx["subject"].to_numpy()[tr])
        xk_ = {}                             # T2-D583 / D586 / D588: training-only knobs ({} = the earlier build byte for byte)
        if args.fm_recomb:
            assert args.fm_rc > 0, "--fm-recomb needs the training windows' user index (--fm-rc > 0, as the D557 donors)"
            xk_["recomb"] = tuple(float(v) for v in args.fm_recomb.split(":"))
        if args.fm_phase:
            xk_["phase"] = tuple(float(v) for v in args.fm_phase.split(":"))
        if args.fm_cons:
            xk_["cons"] = float(args.fm_cons)
        if args.fm_wmargin:
            xk_["wmargin"] = float(args.fm_wmargin)
        if xk_:
            print(f"[t2p_final] training-only FM knobs {xk_}", flush=True)
        xmeta_ = {k_: v_ for k_, v_ in (("recomb", args.fm_recomb), ("phase", args.fm_phase), ("cons", args.fm_cons), ("wmargin", args.fm_wmargin)) if v_}
        fm_aug_kw = {}
        if args.fm_aug:                      # T2-D557: t2p_bench.fm_aug's parser, imported so the two can never disagree
            spec_ = importlib.util.spec_from_file_location("t2p_bench_aug", Path(__file__).resolve().parent / "t2p_bench.py")
            src_ = spec_.loader.get_source("t2p_bench_aug")
            node_ = [n_ for n_ in __import__("ast").parse(src_).body if getattr(n_, "name", "") == "fm_aug"]
            ns_ = {}
            exec(compile(__import__("ast").Module(node_, []), "t2p_bench(fm_aug)", "exec"), ns_)
            fm_aug_kw = {"aug": ns_["fm_aug"](args.fm_aug)}
            print(f"[t2p_final] FM training augmentation {args.fm_aug} -> {fm_aug_kw['aug']}", flush=True)
        if fm_kw:
            print(f"[t2p_final] ST fine-tune knobs {fm_kw}" + (f"; {int(fm_u['utr'].max()) + 1} training users" if fm_u else ""), flush=True)
        fm_seed_list = [int(v) for v in args.fm_seeds.split(",")]
        seed_kn = fm_seed_knobs(args.fm_seed_knobs, fm_seed_list)   # T2-D333 ({} = the earlier build byte for byte)
        XF_by = {}                                                  # T2-D360: the other alignments a seed asks for
        XC, scale_c = None, None                                    # T2-D475e: the CAR windows and their scale (t2p_bench --car's order)
        side_kw, side_blob = {}, None   # fm_side_files: set before this branch (the zip / save steps read it with or without --fm)
        if args.fm_side:                                            # T2-D524: the side vectors from the DELIVERED windows (before CAR / alignment)
            Fr = B.side_raw(X, args.fm_side)
            mu0, sd0 = np.nanmean(Fr[tr], 0), np.nanstd(Fr[tr], 0) + 1e-6       # = t2p_bench.side_z's prior = t2p_side_final's side.pt mu / sd
            Zs = B.side_z(Fr, idx, tr, args.fm_side)
            side_kw = dict(side_tr=Zs[tr], side_te=Zs[ev], side_va=Zs[:0])
            side_blob = dict(kind=args.fm_side, n0=10.0, mu=torch.as_tensor(mu0, dtype=torch.float64), sd=torch.as_tensor(sd0, dtype=torch.float64))
            print(f"[t2p_final] FM side head ({args.fm_side}): {Zs.shape[1]} features, prior over {len(tr)} training rows", flush=True)
        for s in fm_seed_list:
            al_s = seed_kn.get(s, {}).get("align", args.fm_align)      # T2-D360: this seed's alignment (default = --fm-align)
            car_s = bool(seed_kn.get(s, {}).get("car", 0))
            if car_s:                    # as t2p_bench --car: CAR per window first, then the scale and the alignment on the CAR windows
                if XC is None:
                    XC = stream.car_windows(np.asarray(X))
                    scale_c = float(np.median(np.asarray(XC[np.flatnonzero(trm)[::7]]).std(-1)))
                    print(f"[t2p_final] CAR windows for the car seeds: scale {scale_c:.4f} (plain {scale:.4f})", flush=True)
                if ("car", al_s) not in XF_by:
                    XF_by[("car", al_s)] = B.aligned_all(XC, idx, al_s, trm, 0.0, args.shrink, scale_c)
                XFs = XF_by[("car", al_s)]
            else:
                if al_s != args.fm_align and al_s not in XF_by:
                    XF_by[al_s] = B.aligned_all(X, idx, al_s, trm, 0.0, args.shrink, scale)
                XFs = XF if al_s == args.fm_align else XF_by[al_s]
            data = dict(Xtr=XFs[tr], ytr=y[tr], Xva=XFs[:0], yva=y[:0], cva=cells[:0], Xte=XFs[ev], yte=y[ev], cte=cells[ev], **fm_u, **side_kw, **wub)   # T2-D642: wub
            kn = {k_: v_ for k_, v_ in seed_kn.get(s, {}).items() if k_ not in ("align", "epochs", "ckpt", "car")}
            ep_s = int(seed_kn.get(s, {}).get("epochs", args.fm_epochs))   # T2-D384: this seed's length (default = --fm-epochs)
            ck_s = seed_kn.get(s, {}).get("ckpt", args.fm_ckpt)            # T2-D392: this seed's CPT checkpoint (default = --fm-ckpt)
            cur_ = (t2p_fm.t1.CPT_INFO.get(args.fm) or {}).get("ckpt")
            if cur_ is not None and cur_ not in (str(ck_s), str(Path(ck_s).resolve()) if ck_s else ""):
                t2p_fm.reset_cpt(args.fm)                                  # another seed's checkpoint was active: back to the pinned file first
            saved_bank = None
            if kn.get("vp"):                 # T2-D333: the bench's FM bank (t2p_bench --fm-vp with recipe r1: vref 0), as T2-D330 measured
                r1 = T.RECIPES["r1"]
                saved_bank = dict(T.VP_BANK)
                T.VP_BANK.clear()
                T.VP_BANK.update({k_: v_.to(dev) for k_, v_ in B.vp_bank(list(meta["ch_names"]), K=r1.vp_k, sigma=0.005 * kn["vp"],
                                                                         gain=r1.vp_g, shift=r1.vp_s, vref=r1.vp_r, cap=r1.vp_cap).items()})
                print(f"[t2p_final] FM seed {s}: knobs {kn}; FM bank {len(T.VP_BANK['mats'])} draws, jitter {5 * kn['vp']} mm, vref {r1.vp_r}", flush=True)
            elif kn or al_s != args.fm_align or ep_s != args.fm_epochs or ck_s != args.fm_ckpt or car_s:
                print(f"[t2p_final] FM seed {s}: knobs {kn}" + ("; CAR input" if car_s else "") + (f"; align {al_s}" if al_s != args.fm_align else "")
                      + (f"; epochs {ep_s}" if ep_s != args.fm_epochs else "") + (f"; ckpt {ck_s}" if ck_s != args.fm_ckpt else ""), flush=True)
            init_kw, s_fit, ep_fit, cont_m = {}, s, ep_s, {}
            if args.fm_init_zip:             # T2-D546 (b): continue the packaged member (= t2p_bench --fm-cont-arms b on the dev folds)
                imap = dict((int(a_), b_) for a_, b_ in (e_.split(":") for e_ in args.fm_init_map.split(",")))
                assert set(imap) == set(fm_seed_list), ("T2-D546: one --fm-init-map entry per --fm-seeds seed", imap, fm_seed_list)
                st0, m0, sb0 = t2p_fm.load_member_state(args.fm_init_zip, imap[s])
                mine = dict(seed=s, car=car_s, align=al_s, epochs=ep_s, drop_path=args.fm_drop_path, mixup=args.fm_mixup, calib=args.fm_calib,
                            side=args.fm_side or None, aug=args.fm_aug or None, **xmeta_, **fm_kw, **kn)
                theirs = dict(seed=m0["seed"], car=bool(m0.get("car")), align=m0["align"]["mode"], epochs=m0["epochs"], drop_path=m0["drop_path"],
                              mixup=m0["mixup"], calib=m0["calib"], side=(m0.get("side") or {}).get("kind"), aug=m0.get("aug"),
                              **{k_: m0.get(k_) for k_ in xmeta_},
                              **{k_: m0.get(k_) for k_ in list(fm_kw) + list(kn)})
                assert mine == theirs, ("T2-D546: the member's recipe differs from this build's", mine, theirs)
                assert abs(m0["align"]["scale"] - (scale_c if car_s else scale)) < 1e-4 * max(1.0, abs(m0["align"]["scale"])), \
                    ("T2-D546: the alignment scale differs (another training set?)", m0["align"]["scale"], scale_c if car_s else scale)
                if sb0 is not None:          # the side prior over the same training rows must be the stored one
                    assert torch.allclose(sb0["mu"].double(), side_blob["mu"], atol=1e-6) and torch.allclose(sb0["sd"].double(), side_blob["sd"], atol=1e-6), \
                        "T2-D546: the side head's prior differs from this build's (another training set?)"
                import inspect
                dflt = inspect.signature(t2p_fm.fit_eval_fm).parameters
                assert not ({"lr_enc", "lr_head", "warm", "init_state"} & (set(fm_kw) | set(kn))), (fm_kw, kn)
                init_kw = dict(init_state=st0, lr_enc=dflt["lr_enc"].default * args.fm_cont_lr, lr_head=dflt["lr_head"].default * args.fm_cont_lr,
                               warm=0.0)
                s_fit, ep_fit = s + 50000, args.fm_cont_epochs
                cont_m = dict(cont=dict(decision="T2-D546b", source=dict(zip=str(args.fm_init_zip), file=imap[s], seed=s),
                                        epochs=args.fm_cont_epochs, lr_mult=args.fm_cont_lr, lr_enc=init_kw["lr_enc"], lr_head=init_kw["lr_head"],
                                        warm=0.0, data_seed=s_fit))
                print(f"[t2p_final] FM seed {s}: T2-D546 continuation of {imap[s]} ({len(st0)} tensors, recipe checked): {ep_fit} epochs, "
                      f"lr x {args.fm_cont_lr}, no warm-up, data seed {s_fit}", flush=True)
            r = t2p_fm.fit_eval_fm(args.fm, data, s_fit, meta["ch_names"], epochs=ep_fit, drop_path=args.fm_drop_path,
                                   mixup=args.fm_mixup, **fm_kw, **kn, **init_kw, **fm_aug_kw, **xk_,
                                   calib=args.fm_calib, fm_ckpt=ck_s, device=dev, return_state=True, micro=args.fm_micro,
                                   **({"grl": args.fm_grl, "grl_lr": args.fm_grl_lr} if args.fm_grl is not None else {}),   # T2-D649 (below the pinned calib line: test_t2p_final_fm_knobs)
                                   **({"side": args.fm_side} if args.fm_side else {}),
                                   log=lambda m: print(f"[t2p_final fm s{s}] {m}", flush=True))
            if args.fm_init_zip and args.fm_cont_epochs == 0:   # T2-D546 smoke: 0 epochs = the stored member, bit for bit after the fp16 round trip
                assert all(torch.equal(r["state_last"][k_].cpu().to(st0[k_].dtype), st0[k_]) for k_ in st0), "T2-D546: 0-epoch continuation moved a weight"
                print(f"[t2p_final] FM seed {s}: 0-epoch continuation = the stored member (every tensor equal)", flush=True)
            if saved_bank is not None:       # the CNN recipe's bank back (nothing after the FM loop reads it, but keep the global as it was)
                T.VP_BANK.clear()
                T.VP_BANK.update(saved_bank)
            st_ = dict(r["state_last"])
            side_m = {}
            if args.fm_side:                 # T2-D524: the head leaves the member's state for its own file (fp32: 3 x 82 + 3 values)
                hd = {k[len("dc_side."):]: st_.pop(k).float().cpu() for k in [k for k in st_ if k.startswith("dc_side.")]}
                assert set(hd) == {"weight", "bias"} and tuple(hd["weight"].shape) == (int(meta.get("n_classes", 3)), Zs.shape[1]), \
                    (sorted(hd), tuple(hd["weight"].shape) if "weight" in hd else None)
                side_m = {"side": dict(kind=args.fm_side, n0=10.0, file=f"fm_{len(fm_members)}_side.pt")}
                fm_side_files[side_m["side"]["file"]] = dict(side_blob, weight=hd["weight"], bias=hd["bias"])   # written with the members
            assert not any(k.startswith("dc_side.") for k in st_), "T2-D524: a side-head key left in the member state"
            fm_states.append({k: v.half() if v.is_floating_point() else v for k, v in st_.items()})
            fm_probs.append(r["prob_last"])
            fm_members.append(dict(family="st", fm=args.fm, size=args.fm, file=f"fm_{len(fm_members)}.pt", seed=s,
                                   chan_idx=r["chan_idx"], n_samples=int(round(steeg_pt.ST_FS * X.shape[-1] / float(meta["sfreq"]))),
                                   calib=args.fm_calib, layer=-1, align=dict(mode=al_s, scale=scale_c if car_s else scale, shrink=args.shrink),
                                   **({"car": True} if car_s else {}), **({"micro": args.fm_micro} if args.fm_micro > 1 else {}),   # T2-D475e: read by template W14 only (W13 and earlier would feed it plain input)
                                   epochs=ep_s, drop_path=args.fm_drop_path, mixup=args.fm_mixup, **fm_kw, **kn, **side_m, **cont_m,
                                   **({"aug": args.fm_aug} if args.fm_aug else {}),    # T2-D557: training-only (the template never reads it)
                                   **xmeta_,                                            # T2-D583 / D586 / D588: training-only too
                                   precision="bf16 autocast, fp16 weights",
                                   **({"user_balance": True} if wub else {}),          # T2-D642: training-only (the template never reads it)
                                   **({"user_balance_pow": float(args.user_balance_pow)} if wub and args.user_balance_pow != 1.0 else {}),   # T2-D647
                                   **({"grl": float(args.fm_grl)} if args.fm_grl is not None else {}),   # T2-D649: training-only (the adversary is not shipped)
                                   provenance=r["provenance"], cpt=dict(t2p_fm.t1.CPT_INFO.get(args.fm) or {}) or None))
            print(f"[t2p_final] FM seed {s}: kit-test cellBA "
                  f"{metrics.cell_balanced_accuracy(y[te], r['prob_last'][te].argmax(1), cells[te])[0]:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    mor_states, mor_probs, mor_members = [], [], []
    if args.fm2:
        import t2p_fm
        import t2p_mor_pkg
        assert args.fm2 in t2p_fm.t1.MOR_NAMES and args.fm2_calib == "zwin", (args.fm2, args.fm2_calib)
        assert not extras, "--fm2 with --extra: not wired"
        XM = B.aligned_all(X, idx, args.fm2_align, trm, 0.0, args.shrink, scale)
        for s in [int(v) for v in args.fm2_seeds.split(",")]:
            data = dict(Xtr=XM[tr], ytr=y[tr], Xva=XM[:0], yva=y[:0], cva=cells[:0], Xte=XM[ev], yte=y[ev], cte=cells[ev])
            r = t2p_fm.fit_eval_fm(args.fm2, data, s, meta["ch_names"], epochs=args.fm2_epochs, lr_enc=args.fm2_lr_enc,
                                   calib=args.fm2_calib, device=dev, return_model=True, lora_in=args.fm2_lora_in,
                                   lora_in_roles=bool(args.fm2_lora_in_roles), lora_in_dropout=args.fm2_lora_in_dropout,
                                   log=lambda m: print(f"[t2p_final mor s{s}] {m}", flush=True))
            mem, sd = t2p_mor_pkg.merge(r["net_last"])
            mor_states.append({k: v.half() if v.is_floating_point() else v for k, v in sd.items()})
            mor_probs.append(r["prob_last"])
            mor_members.append(dict(mem, family="mor", fm=args.fm2, seed=s, align=dict(mode=args.fm2_align, scale=scale, shrink=args.shrink),
                                    epochs=args.fm2_epochs, lr_enc=args.fm2_lr_enc, lora_in=args.fm2_lora_in,
                                    lora_in_roles=bool(args.fm2_lora_in_roles), lora_in_dropout=args.fm2_lora_in_dropout,
                                    precision="bf16 autocast, fp16 weights", provenance=r["provenance"]))
            print(f"[t2p_final] MOR seed {s}: kit-test cellBA "
                  f"{metrics.cell_balanced_accuracy(y[te], r['prob_last'][te].argmax(1), cells[te])[0]:.4f}; merged roles "
                  f"{mem['roles']} ({time.time() - t0:.0f}s)", flush=True)
    if mor_members:                      # per-member FM output weights (template: their sum = fm_weight)
        for m in fm_members:
            m["weight"] = args.fm_weight / len(fm_members)
        for m in mor_members:
            m["weight"] = args.fm2_weight / len(mor_members)
        for i, m in enumerate(mor_members):
            m["file"] = f"fm_{len(fm_members) + i}.pt"
    fm_total = (args.fm_weight if fm_members else 0.0) + (args.fm2_weight if mor_members else 0.0)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if mor_members:                      # T2-D140: ST and / or the MOR source inlined (t2p_mor_pkg.template_source)
        import t2p_mor_pkg
        src = t2p_mor_pkg.template_source(st=bool(fm_members), mor=True)
    else:
        src = _tpl(args.template).read_text()
        if fm_members:
            marker = next(ln for ln in src.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
            st_src = STEEG_PT.read_text()
            src = src.replace(marker, f"# ---- inlined by t2p_final.py: src/ni26/fm/steeg_pt.py (sha256 "
                                      f"{hashlib.sha256(st_src.encode()).hexdigest()[:12]})\n" + st_src + "# ---- end inlined steeg_pt.py")
    (out / "submission.py").write_text(src)
    torch.save(states, out / "weights.pt")
    if pu_states:
        torch.save(pu_states, out / "weights_pu.pt")
    if states2:
        torch.save(states2, out / "weights2.pt")
    if pu_states2:
        torch.save(pu_states2, out / "weights2_pu.pt")
    if states3:                          # T2-D352
        torch.save(states3, out / "weights3.pt")
    if pu_states3:
        torch.save(pu_states3, out / "weights3_pu.pt")
    for m, sd in zip(fm_members + mor_members, fm_states + mor_states):
        torch.save(sd, out / m["file"])
    for f_, blob_ in fm_side_files.items():          # T2-D524: the FM side heads, one file per member
        torch.save(blob_, out / f_)
    tta = None
    if args.tta.startswith("prior"):
        p_ = args.tta.split(":")
        tta = dict(kind="prior", alpha=float(p_[1]), n0=float(p_[2]))
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_times=int(X.shape[-1]), n_classes=3,
               braindecode_class=T.MODEL_SPECS[args.model][0], model=args.model,
               model_kwargs={k: ([list(b) for b in v] if k == "n_bands" else v) for k, v in T.MODEL_SPECS[args.model][1].items()},
               align=dict(mode=args.align, scale=scale, n0=args.n0, shrink=args.shrink,
                          prior=prior.tolist() if prior is not None else None),
               tta=tta, recipe=args.recipe, epochs=args.epochs, seeds=args.seeds, train=args.train,
               train_subjects=sorted(idx.loc[trm, "subject"].unique().tolist()),
               extra=[dict(name=e["name"], sha=e["sha"], n=int(len(e["keep"]))) for e in extras],
               export=str(args.export), export_sha=meta.get("x_sha256"), repo_commit=git, kit_sha=ksha,
               versions=_versions(),
               fm_members=fm_members + mor_members, fm_weight=fm_total, user_blend=ub_cfg,
               cnn2=dict(model=args.model2, braindecode_class=T.MODEL_SPECS[args.model2][0], file="weights2.pt", weight=args.weight2,
                         model_kwargs=dict(T.MODEL_SPECS[args.model2][1]),
                         seeds=args.seeds2, epochs=args.epochs2,
                         align=dict(mode=args.align2, scale=scale, n0=args.n02, shrink=args.shrink,
                                    prior=prior2.tolist() if prior2 is not None else None)) if states2 else None)
    if ub2_cfg:                          # T2-D161: the key only exists when on (flag off = the earlier config.json byte for byte)
        cfg["user_blend2"] = ub2_cfg
    if args.recipe2:                     # T2-D232: the key only exists when set (unset = the earlier config.json byte for byte)
        cfg["cnn2"]["recipe"] = args.recipe2
    if states3:                          # T2-D352: the keys only exist with --model3 (off = the earlier config.json byte for byte)
        cfg["cnn3"] = dict(model=args.model3, braindecode_class=T.MODEL_SPECS[args.model3][0], file="weights3.pt", weight=args.weight3,
                           model_kwargs=dict(T.MODEL_SPECS[args.model3][1]), seeds=args.seeds3, epochs=args.epochs3,
                           recipe=args.recipe3 or args.recipe,
                           align=dict(mode=args.align3, scale=scale, n0=args.n03, shrink=args.shrink,
                                      prior=prior3.tolist() if prior3 is not None else None))
        if ub3_cfg:
            cfg["user_blend3"] = ub3_cfg
        assert 1.0 - fm_total - args.weight2 - args.weight3 > 0, "the FBCNet share must stay positive"
    if args.user_balance:                # T2-D638 W24': recorded only when on (absent = config.json bitwise)
        cfg["user_balance"] = True
        if args.user_balance_pow != 1.0:                  # T2-D647: recorded only when not D630's 1 / sessions
            cfg["user_balance_pow"] = float(args.user_balance_pow)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    # ---- golden: research path (vectorised) vs the packaged StreamModel fed one window at a time
    pr = np.mean(np.asarray(probs, np.float64), 0)
    if Ppu is not None:
        pr = t2p_fpcalib.blend_route(pr, Ppu, kstar, args.user_blend)
    w3_ = args.weight3 if states3 else 0.0   # T2-D352: = the template's w2 / (1 - fm - w3); 0.0 = the earlier arithmetic exactly
    if states2:
        w2n = args.weight2 / (1.0 - fm_total - w3_)
        if Ppu2 is None:
            pr = (1.0 - w2n) * pr + w2n * np.mean(np.asarray(probs2, np.float64), 0)
        else:                            # T2-D161: the second family's blend toward the same routed user's copies
            pr = (1.0 - w2n) * pr + w2n * t2p_fpcalib.blend_route(np.mean(np.asarray(probs2, np.float64), 0), Ppu2, kstar, args.user_blend2)
    if states3:                          # T2-D352
        w3n = args.weight3 / (1.0 - fm_total)
        p3_ = np.mean(np.asarray(probs3, np.float64), 0)
        if Ppu3 is not None:
            p3_ = t2p_fpcalib.blend_route(p3_, Ppu3, kstar, args.user_blend3)
        pr = (1.0 - w3n) * pr + w3n * p3_
    if mor_members:                      # T2-D140: per-member weights (ST share + MOR share = fm_total)
        pf = sum(m["weight"] * np.asarray(q, np.float64) for m, q in zip(fm_members + mor_members, fm_probs + mor_probs)) / fm_total
        pr = (1.0 - fm_total) * pr + fm_total * pf
    elif fm_members:
        pr = (1.0 - args.fm_weight) * pr + args.fm_weight * np.mean(np.asarray(fm_probs, np.float64), 0)
    if tta is not None:
        strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
        pr = causal_prior_correction(pr, (strm[ev], idx["order"].to_numpy()[ev]), tta["alpha"], tta["n0"])
    kit.use_track("bci_decoding")
    spec = importlib.util.spec_from_file_location("ni26_t2p_sub", out / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver.__new__(mod.Solver)
    m = solver.load_model(dict(submission_dir=out, device=dev, ch_names=list(meta["ch_names"]), sfreq=meta["sfreq"],
                               n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3))
    pk = np.empty(len(ev), int)
    sub = idx.iloc[ev].copy()
    sub["pos"] = np.arange(len(ev))
    t1 = time.time()
    for _, g in sub.groupby("stream", sort=False):
        mm = copy.deepcopy(m)
        mm.reset_state()
        for _, row in g.sort_values("order").iterrows():
            xw = torch.as_tensor(np.asarray(X[int(row["i"])])[None], device=dev)
            pk[int(row["pos"])] = int(mm.predict(xw)[0])
    s_per_window = (time.time() - t1) / max(len(ev), 1)   # the streamed predict loop only (not the zip write / guard check)
    # research path ran in the kit order too (aligned_all per stream in "order"); compare argmax
    agree = float((pk == pr.argmax(1)).mean())
    late = (idx["subject"].isin(["P04", "P10"]) & idx["session"].astype(str).isin(["2", "3"])).to_numpy()
    agree_by = {k: float((pk[m] == pr[m].argmax(1)).mean()) for k, m in (("kit_test", tem), ("later_sessions_P04_P10", late))}
    print(f"[t2p_final] golden agreement {agree:.4f} on {len(ev)} windows (14 streams); {agree_by}", flush=True)
    sc_pkg = metrics.cell_balanced_accuracy(y[te], pk[te], cells[te])[0]
    sc_res = metrics.cell_balanced_accuracy(y[te], pr[te].argmax(1), cells[te])[0]
    zp = Path(str(out) + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["submission.py", "config.json", "weights.pt"] + (["weights2.pt"] if states2 else []) + \
                 (["weights_pu.pt"] if pu_states else []) + (["weights2_pu.pt"] if pu_states2 else []) + \
                 (["weights3.pt"] if states3 else []) + (["weights3_pu.pt"] if pu_states3 else []) + \
                 [m["file"] for m in fm_members + mor_members] + sorted(fm_side_files):
            z.write(out / f, f)
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    guard = guarded_check(out, dev)
    res = dict(config=cfg, golden_agreement=agree, golden_agreement_by=agree_by, golden_windows=int(len(ev)),
               kit_test_cellba_research=sc_res, kit_test_cellba_package=sc_pkg,
               package_seconds_per_window=s_per_window, zip=str(zp), zip_sha256=sha,
               zip_mb=round(zp.stat().st_size / 1e6, 2), worker_guard=guard, seconds=round(time.time() - t0))
    Path(str(out) + ".final.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1), flush=True)
    if args.fm_donor:                    # T2-D475e: the donor's FM files are the product; its package is not a candidate
        print(f"[t2p_final] DONOR build (--fm-donor): golden agreement {agree:.4f} reported, not enforced (car members are fed plain input "
              f"by this template); not a candidate", flush=True)
        Path(str(out) + ".DONOR_NOT_A_CANDIDATE").write_text("T2-D475e: FM member donor; the package itself must never be uploaded\n")
    elif agree < 0.995:   # fp noise (TF32 conv in training-time predict vs fp32 in the package) may flip a near-tie
        raise SystemExit(f"!! package != research path on {(1 - agree) * len(ev):.0f} windows")
    if not guard["ok"]:
        raise SystemExit(f"!! package fails under the worker-image import guard (refused: {guard['refused_imports']})")


if __name__ == "__main__":
    main()
