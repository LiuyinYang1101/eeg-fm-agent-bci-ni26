#!/usr/bin/env python
"""Track 2: re-package an EXISTING T2 zip with the current submission template (no training), then prove it is the
same model on clean data and robust on corrupted data (hub LESSONS B24).

  python t2p_repack.py --zip OLD.zip --out NEW_DIR --export <export_proteus_v1> [--device cuda]

1. NEW_DIR/ = config.json, weights.pt and every FM member file copied byte for byte from OLD.zip; submission.py rebuilt
   from tracks/t2_bci/submission/t2p_submission.py exactly as t2p_final.py does (ST members: src/ni26/fm/steeg_pt.py
   inlined; its sha must equal the one the old package inlined, so the model code is unchanged). NEW_DIR.zip written.
2. clean equality: every public PROTEUS stream (all 14 participant x session streams, every split), fed one window at
   a time from deepcopy + reset_state as benchmark_utils.streaming does, through the OLD and the NEW package: the
   predictions must be identical (the guards are inert on public data); the kit-test cell-BA is reported (= the
   official warm-up replay of the old zip).
3. robustness cases on the kit-test streams, NEW and OLD package side by side: one window all-zero / 10 % NaN / all
   NaN / 1 % +-inf / x1e4 amplitude (at window 20, and x1e4 at window 0), and one channel flat for the whole stream.
   NEW must give finite probabilities at every window (StreamModel.last_p) and, for the one-window cases, the windows
   after the corrupted one must agree with the clean predictions on >= 90 %. OLD is reported, not asserted.
Writes NEW_DIR.repack.json.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from ni26 import kit, metrics  # noqa: E402

TEMPLATE = REPO / "tracks" / "t2_bci" / "submission" / "t2p_submission.py"
STEEG_PT = REPO / "src" / "ni26" / "fm" / "steeg_pt.py"
K_BAD = 20


def load(d, tag, meta):
    spec = importlib.util.spec_from_file_location(f"ni26_t2p_{tag}", d / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver.__new__(mod.Solver).load_model(dict(meta, submission_dir=d))


def run_stream(m, xs, dev, probs=None):
    """one stream, one window at a time from a fresh copy; returns (preds, all-probabilities-finite or None).
    probs: a list that receives each window's output probabilities (StreamModel.last_p) when the template has them."""
    mm = copy.deepcopy(m)
    mm.reset_state()
    pr, fin = [], True
    for x in xs:
        pr.append(int(mm.predict(torch.as_tensor(x[None], device=dev))[0]))
        if hasattr(mm, "last_p"):
            fin &= bool(np.isfinite(mm.last_p).all())
            if probs is not None:
                probs.append(np.asarray(mm.last_p, np.float64))
    return np.asarray(pr), (fin if hasattr(mm, "last_p") else None)


def run_parts(m, xs, dev, rec=None, static=False):
    """T2-D210: one stream from a fresh copy (NI26_PARTS=1): (each window's StreamModel.last_p, each window's last_parts, the W16 window-mean
    family's probability, the W18 imcoh family's probability). T2-D524 (W18): rec = a list that receives each window's FM member probabilities
    going into and coming out of the side heads (only for a package with heads). T2-D534: static = the copy runs with in-stream adaptation OFF
    (is_cfg None: the W15 copies are neither refitted nor blended), so its CNN parts are the frozen nets' alone."""
    mm = copy.deepcopy(m)
    if static:
        mm.is_cfg = None
    mm.reset_state()
    if rec is not None and getattr(mm, "fm_heads", None) is not None:
        f0 = mm._fm_head_apply

        def _rec(ps, f0=f0):
            out_ = f0(ps)
            rec.append(([np.asarray(q, np.float64) for q in ps], [np.asarray(q, np.float64) for q in out_]))
            return out_
        mm._fm_head_apply = _rec
    lp, parts, dcs, ics = [], [], [], []
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
        lp.append(np.asarray(mm.last_p, np.float64))
        parts.append([None if q is None else np.asarray(q, np.float64) for q in mm.last_parts])
        dc_ = getattr(mm, "_dc_now", None) if getattr(mm, "dc", None) is not None else None   # T2-D516 (W16): the window-mean family's probability
        dcs.append(None if dc_ is None else np.asarray(dc_, np.float64))
        ic_ = getattr(mm, "_ic_now", None) if getattr(mm, "ic", None) is not None else None   # T2-D525 (W18): the imcoh family's probability
        ics.append(None if ic_ is None else np.asarray(ic_, np.float64))
    return lp, parts, dcs, ics


def set_cfg(cfg, kv):
    """T2-D210 --set: one config edit dotted.key=JSON in place; a numeric part indexes a list (config-only repacks, 2026-11-01:
    fm_members.0.weight=0.3). Returns (key, old value, new value); a missing dict key reads as None, a bad list index raises."""
    k_, v_ = kv.split("=", 1)
    path = k_.split(".")
    node = cfg
    for part in path[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = int(path[-1]) if isinstance(node, list) else path[-1]
    was = node[last] if isinstance(node, list) else node.get(last)
    if v_.startswith("@"):                     # T2-D476: @<repo path>[#key] = the JSON value from a committed file (no JSON through ssh quoting)
        fp, _, sub = v_[1:].partition("#")
        fp = Path(fp) if Path(fp).is_absolute() else REPO / fp
        val = json.loads(fp.read_text())
        node[last] = val[sub] if sub else val
    else:
        node[last] = json.loads(v_)
    return k_, was, node[last]


def check_weights(cfg):
    """the template's own asserts, before any build: weighted FM members sum to fm_weight (1e-9), and the
    FBCNet share 1 - fm_weight - cnn2.weight stays positive."""
    fmw = float(cfg.get("fm_weight", 0.0))
    mw = [m.get("weight") for m in cfg.get("fm_members", [])]
    if mw and all(w is not None for w in mw):
        assert abs(sum(mw) - fmw) < 1e-9, f"FM member weights {mw} do not sum to fm_weight {fmw}"
    w2 = float((cfg.get("cnn2") or {}).get("weight", 0.0))
    w3 = float((cfg.get("cnn3") or {}).get("weight", 0.0))          # T2-D352 (W11)
    assert 0.0 <= fmw < 1.0 and 0.0 <= w2 and 0.0 <= w3 and fmw + w2 + w3 < 1.0, f"shares out of range: fm_weight {fmw}, cnn2 {w2}, cnn3 {w3}"


def _tensors(obj):
    """every tensor in a nested list / dict (state dicts, lists of state dicts), in a deterministic order."""
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        return [t for k in sorted(obj, key=str) for t in _tensors(obj[k])]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _tensors(v)]
    return []


ROUTE_KEYS = ("users", "mean", "scale", "W", "mu", "n_seeds")    # data-derived fingerprint route + copy count: must match the base


def merge_copies(cfg, dcfg, recipes=None):
    """The S1 copies swap (2026-11-01): the base's user_blend / user_blend2 entries with the DONOR's per-user copies.
    The donor = a CNN-only t2p_final build of the SAME draw (same seeds / epochs / data) whose copies trained under another
    per-user recipe (pv10: pu_vp 1.0); its pooled training must be field-for-field the base's (the RECIPES differ in name / pu_* only).
    Returns (the new entries {key: dict}, the copy files to take from the donor). The base keeps r / every (the route is unchanged)."""
    if recipes is None:
        from ni26 import train as T
        recipes = T.RECIPES
    for k in ("model", "seeds", "epochs", "align", "train", "train_subjects", "n_times", "sfreq", "ch_names", "export_sha"):
        assert dcfg.get(k) == cfg.get(k), f"copies donor: {k} differs ({dcfg.get(k)!r:.80} vs {cfg.get(k)!r:.80})"
    for k in ("model", "seeds", "epochs", "align", "recipe"):
        assert (dcfg.get("cnn2") or {}).get(k) == (cfg.get("cnn2") or {}).get(k), f"copies donor: cnn2.{k} differs"
    ra, rb = recipes[cfg["recipe"]], recipes[dcfg["recipe"]]
    pooled = [f for f in vars(ra) if f != "name" and not f.startswith("pu_")]
    diff = [f for f in pooled if getattr(ra, f) != getattr(rb, f)]
    assert not diff, f"copies donor: the pooled recipe differs from the base's in {diff} ({cfg['recipe']} vs {dcfg['recipe']})"
    new, files = {}, []
    for key in ("user_blend", "user_blend2"):
        a, b = cfg.get(key), dcfg.get(key)
        assert a and b, f"copies donor: {key} missing (base {bool(a)}, donor {bool(b)})"
        assert a["file"] == b["file"], f"copies donor: {key}.file {b['file']} vs {a['file']}"
        for f in ROUTE_KEYS:
            if f in a or f in b:
                va, vb = a.get(f), b.get(f)
                same = va == vb if f in ("users", "n_seeds") else np.allclose(np.asarray(va, np.float64), np.asarray(vb, np.float64), atol=1e-6, rtol=0)
                assert same, f"copies donor: {key}.{f} differs (not the same draw / data)"
        e = dict(a)
        for f, v in b.items():
            if f not in ROUTE_KEYS and f not in ("r", "every", "file"):
                e[f] = v
        e["copies_from_recipe"] = dcfg["recipe"] if key == "user_blend" else b.get("recipe_pu", dcfg["recipe"])
        new[key] = e
        files.append(a["file"])
    return new, files


FM_ADAPT_KEYS = ("fm_side_refit", "fm_proto", "cnn_lp")   # 2026-11-22 (T2-D652 one-hop): the base's in-stream adaptation that reads the FM members


def reconfig_parts_gate(parts_dev, parts_pre, side_info, fm_adapt, tol=1e-6):
    """The reconfig parts gate; returns the list of failed checks ([] = PASS).
    STRICT (every repack except the case below): every part (p1 / p2 / p3 / pf / recombine) and, with donor FM side heads, in_vs_donor and
    head_vs_indep <= tol on EVERY window (the reference is static: the donor's members / heads at their trained weights, no in-stream refit).
    FM SWAP ONTO A BASE WITH IN-STREAM FM ADAPTATION (fm_adapt: --fm-from and the base has any of FM_ADAPT_KEYS): NEW's FM part, its heads and the
    recombination are EXPECTED to move once the in-stream refits fire (the static reference does not model fm_side_refit / fm_proto / cnn_lp);
    so the CNN parts p1 / p2 / p3 stay strict on every window (they are compared with in-stream OFF on both sides), and pf / recombine /
    in_vs_donor / head_vs_indep are strict on the windows BEFORE the first refit point (where the adaptation is inert): the members are the donor's
    exactly there. The all-window values are then descriptive. Refuses an fm_adapt gate with no pre-refit window."""
    bad = []
    if not fm_adapt:
        bad += [f"{k} {v:.3e}" for k, v in parts_dev.items() if v > tol]
        if side_info is not None:
            bad += [f"{k} {side_info[k]:.3e}" for k in ("in_vs_donor", "head_vs_indep") if side_info[k] > tol]
        return bad
    bad += [f"{k} {parts_dev[k]:.3e}" for k in ("p1", "p2", "p3") if parts_dev.get(k, 0.0) > tol]
    if parts_pre.get("windows", 0) <= 0:
        bad.append("no pre-refit window to check the FM part on")
    keys = ("pf", "recombine") + (("in_vs_donor", "head_vs_indep") if side_info is not None else ())
    bad += [f"pre-refit {k} {parts_pre.get(k, 0.0):.3e}" for k in keys if parts_pre.get(k, 0.0) > tol]
    return bad


def _sinkhorn_restart(P, iters, n0, lam, start, cks, mode):
    """T2-D595 / D603 (the repack's reference recombination for a W22 decoder-state package): t2p_report.causal_sinkhorn of ONE stream with its
    rows restarted at each refit c < len(P) (the template's fired refits). mode 'start': the restart's n0 rows = `start` (D595's fam_prior reset);
    'mean': = the normalised mean of THIS decoder's own outputs on the stream's windows < c (D603's informed restart). = Track3's lab decoders
    (lead/diag_word_2026-11-18/d595.py sinkhorn_reset, d60x.py sinkhorn_seg); golden in tests/test_t2p_repack_restart.py."""
    P = np.asarray(P, np.float64)
    K = P.shape[1]
    st0 = np.asarray(start, np.float64)
    out = np.empty_like(P)
    refs = [int(c) for c in cks if int(c) < len(P)]
    seg_start = {}
    for t in range(len(P)):
        seg = max([0] + [c for c in refs if c <= t])
        if seg not in seg_start:
            if seg == 0 or mode == "start":
                seg_start[seg] = st0
            else:
                assert mode == "mean", mode
                m = out[:seg].mean(0)
                seg_start[seg] = m / m.sum()
        M = np.vstack([np.tile(seg_start[seg], (int(n0), 1)), P[seg:t + 1]])
        if lam == 1.0:
            for _ in range(int(iters)):
                M = M * (M.shape[0] / K) / M.sum(0, keepdims=True)
                M = M / M.sum(1, keepdims=True)
        else:
            w = float(lam) ** np.arange(M.shape[0] - 1, -1, -1, dtype=np.float64)
            for _ in range(int(iters)):
                M = M * (w.sum() / K) / (M * w[:, None]).sum(0, keepdims=True)
                M = M / M.sum(1, keepdims=True)
        out[t] = M[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-streams", type=int, default=0, help="smoke only: limit the clean-equality streams")
    ap.add_argument("--allow-differ", type=int, default=0, help="clean windows the NEW package may predict differently "
                    "(only for a template change that is MEANT to move outputs, named in the DECISIONS row; max |dp| reported)")
    ap.add_argument("--donor-check-only", action="store_true", help="T2-D549: run the --fm-from donor checks (config keys, input references, "
                    "side-head priors), print OK and exit: no package is written (a dry check before a build slot)")
    ap.add_argument("--fm-from", default="", help="T2-D210: a DONOR zip whose FM member files (and fm_members config) replace the old ones "
                    "(the same FM recipe: every member config key but provenance must match)")
    ap.add_argument("--fm-add-from", default="", help="T2-D482 (W14, PREPARED): a DONOR zip (t2p_final --fm-donor) whose FM members with seed in "
                    "--fm-add-seeds are APPENDED to the base's (files copied as fm_<n+j>.pt, their car / ref and align kept); equal-weight mean "
                    "over all members; checked against cref = a package holding only the added members")
    ap.add_argument("--fm-add-seeds", default="", help="T2-D482: the donor seeds to append, e.g. 600,601,602")
    ap.add_argument("--fm-keep", default="", help="T2-D496 (W14-lite): comma indices (ascending) of the base's fm_members to KEEP; the equal-weight "
                    "FM mean is then over the kept members only, the dropped members' files leave the zip; checked against kref = the base's "
                    "CNNs + only the kept members, built from the OLD config independently")
    ap.add_argument("--copies-from", default="", help="the S1 copies swap: a DONOR zip (a CNN-only t2p_final build of the same draw, "
                    "another per-user recipe) whose per-user copy files replace the base's; every other byte stays the base's (merge_copies)")
    ap.add_argument("--template", default=str(TEMPLATE), help="T2-D352: the template the new submission.py is built from (default W9; a "
                    "package with cnn3 needs W11 = t2p_submission_w11.py)")
    ap.add_argument("--set-ref", action="store_true", help="T2-D500: apply the --set edits to REF as well: for a config change MEANT to move the "
                    "CNN parts (e.g. the copies' r): the parts check then verifies NEW against a REF computing the same thing, the OLD vs REF "
                    "inert-template check is replaced by OLD vs NEW (reported: the evidence that the edit took); the new template must be the "
                    "old package's own, byte for byte")
    ap.add_argument("--add-file", action="append", default=[], help="T2-D516 (W16): copy this file into the package (the new template reads it, "
                    "e.g. the window-mean family's dcfam.npz); allowed with --set-ref")
    ap.add_argument("--side-from", default="", help="T2-D549 R2: a side.pt with MORE side nets per family (the base's nets first, byte for byte; the same "
                    "kind / n0 / mu / sd / scale / recipe / export_sha) replaces the package's; its own repack (no other edit, the old template). Gate T: "
                    "NEW's cnn2 / cnn3 parts = (n_old REF + n_add B) / n_new on every public window, B = REF with only the added nets")
    ap.add_argument("--side-replace", default="", help="T2-D580: a side.pt with the SAME side nets per family (the same architecture: state-dict "
                    "keys and shapes; the same kind / n0 / scale / export_sha / mu / sd / seeds; another recipe) REPLACES the package's; its own repack "
                    "(no other edit, the old template byte for byte); every other package entry is asserted byte-identical (config.json as parsed JSON); "
                    "the parts gate: NEW's FBCNet part and FM part == REF's on every window (the p1 / pf gates), only the side families' parts move")
    ap.add_argument("--set", action="append", default=[], help="T2-D210: a config edit dotted.key=JSON, e.g. cnn2.weight=0.3 or fm_members.0.weight=0.3 (a numeric part indexes a list; repeatable)")
    ap.add_argument("--gpu-graph", action="store_true", help="T2-D208: set config gpu_graph = true in the NEW package (the CUDA-graph path for a package without a MOR member); the clean-equality check then proves graph = eager on every public window")
    ap.add_argument("--drop-family", default="", help="row 75: drop this FM family (e.g. mor) from the package: fm_members minus it, every "
                    "remaining weight (FM members, cnn2.weight) / (1 - its weight); its file is not copied; the template branch follows the families")
    ap.add_argument("--tta-vp", type=int, default=0, help="row 75: the CNN test-time averaging over K views (cfg cnn_tta_vp; tta_vp.pt = the "
                    "first K-1 draws of the training vp bank of --tta-vp-recipe)")
    ap.add_argument("--tta-copies", action="store_true", help="row 75: the per-user copies take the views too")
    ap.add_argument("--tta-vp-recipe", default="", help="row 75: the CNN recipe whose vp bank the nets trained through (t2p_final's call)")
    args = ap.parse_args()
    t0 = time.time()
    reconfig = bool(args.fm_from or args.set or args.copies_from or args.fm_add_from or args.fm_keep or args.side_from or args.side_replace)
    if reconfig:                               # T2-D210: the template's pre-mix parts (set before any template module loads)
        import os
        os.environ["NI26_PARTS"] = "1"
    ksha = kit.kit_sha()                       # review T2 P1c: repack against the pinned kit only
    if ksha != kit.KIT_SHA:
        raise SystemExit(f"!! kit {kit.kit_root()} is at {ksha}, not the pinned {kit.KIT_SHA}")
    out, old = Path(args.out), Path(args.out + "_old")
    for d in (out, old):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(old)
    cfg = json.loads((old / "config.json").read_text())
    r75 = bool(args.drop_family or args.tta_vp)
    assert not (r75 and (args.fm_from or args.set or args.copies_from)), "row 75 mode is its own reconfiguration"
    assert not (args.copies_from and args.fm_from), "one donor per repack (--copies-from, then --fm-from in a second repack)"
    if args.drop_family:                     # row 75: the family leaves; the rest re-weighted proportionally (no re-pick)
        drop = [m for m in cfg["fm_members"] if m.get("family", "st") == args.drop_family]
        keep = [m for m in cfg["fm_members"] if m.get("family", "st") != args.drop_family]
        assert drop and keep and all(m.get("weight") for m in cfg["fm_members"]) and cfg.get("cnn2"), "row 75: weighted members + cnn2 needed"
        sc = 1.0 / (1.0 - sum(m["weight"] for m in drop))
        for m in keep:
            m["weight"] = m["weight"] * sc
        cfg["fm_members"], cfg["fm_weight"] = keep, sum(m["weight"] for m in keep)
        cfg["cnn2"]["weight"] = cfg["cnn2"]["weight"] * sc
        print(f"[repack] row 75: {args.drop_family} dropped; fm_weight {cfg['fm_weight']:.4f}, cnn2.weight {cfg['cnn2']['weight']:.4f}", flush=True)
    fm_files = [m["file"] for m in cfg.get("fm_members", [])] + ([cfg["cnn2"]["file"]] if cfg.get("cnn2") else []) + \
        ([cfg["user_blend"]["file"]] if cfg.get("user_blend") else []) + \
        ([cfg["user_blend2"]["file"]] if cfg.get("user_blend2") else []) + \
        ([cfg["cnn3"]["file"]] if cfg.get("cnn3") else []) + ([cfg["user_blend3"]["file"]] if cfg.get("user_blend3") else []) + \
        ([cfg["cnn_tta_vp"]["file"]] if int((cfg.get("cnn_tta_vp") or {}).get("k", 1)) > 1 else []) + \
        ([cfg["dcfam"]["file"]] if cfg.get("dcfam") else []) + \
        ([cfg["side"]["file"]] if cfg.get("side") else []) + \
        ([cfg["imcoh"]["file"]] if cfg.get("imcoh") else []) + \
        [m["side"]["file"] for m in cfg.get("fm_members", []) if m.get("side")]   # T2-D524 (W18): the FM side heads; T2-D161; row 75 / D304: the view draws; T2-D516 (W16): the window-mean family; T2-D517 (W17): side nets; T2-D525 (W18): imcoh
    fams = {m.get("family", "st") for m in cfg.get("fm_members", [])}
    if "mor" in fams:                     # T2-D140: the MOR source inlined too; both inlined sources must be the old package's
        import t2p_mor_pkg
        old_src = (old / "submission.py").read_text()
        for path, label in ((t2p_mor_pkg.MOR_SRC, "the in-house MOR source"),) + \
                           (((STEEG_PT, "src/ni26/fm/steeg_pt.py"),) if "st" in fams else ()):
            sha_ = hashlib.sha256(path.read_text().encode()).hexdigest()[:12]
            m_ = re.search(r"inlined by t2p_final\.py: " + re.escape(label) + r" \(sha256 ([0-9a-f]{12})\)", old_src)
            assert m_ and m_.group(1) == sha_, f"{label} changed since the old package ({m_} vs {sha_})"
        src = t2p_mor_pkg.template_source(st="st" in fams, mor=True)
    else:
        src = (Path(args.template) if Path(args.template).is_absolute() else REPO / args.template).read_text()
    assert not cfg.get("cnn3") or "cnn3" in src, "a cnn3 package needs --template W11"      # T2-D352
    if cfg.get("fm_members") and "mor" not in fams:
        st_src = STEEG_PT.read_text()
        sha = hashlib.sha256(st_src.encode()).hexdigest()[:12]
        old_sha = re.search(r"inlined by t2p_final\.py: src/ni26/fm/steeg_pt\.py \(sha256 ([0-9a-f]{12})\)",
                            (old / "submission.py").read_text())
        assert old_sha and old_sha.group(1) == sha, f"steeg_pt.py changed since the old package ({old_sha} vs {sha})"
        marker = next(ln for ln in src.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
        src = src.replace(marker, f"# ---- inlined by t2p_final.py: src/ni26/fm/steeg_pt.py (sha256 {sha})\n" + st_src +
                          "# ---- end inlined steeg_pt.py")
    (out / "submission.py").write_text(src)
    for f in ["config.json", "weights.pt"] + fm_files:
        shutil.copyfile(old / f, out / f)
        assert (old / f).read_bytes() == (out / f).read_bytes(), f
    extra = []
    for af in args.add_file:                   # T2-D516 (W16): a data file the new template reads (e.g. dcfam.npz); its sha is reported
        dst = out / Path(af).name
        assert not dst.exists(), f"--add-file {af}: {dst.name} is already a package file"
        shutil.copyfile(af, dst)
        extra.append(dst.name)
        print(f"[repack] added file {dst.name} (sha256 {hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}, {dst.stat().st_size / 1e6:.2f} MB)", flush=True)
    if r75:                                  # row 75: the edited config (+ the view draws); REF / NEW0 = tv off under the OLD / this template
        c75 = json.loads(json.dumps(cfg))
        if args.gpu_graph:
            c75["gpu_graph"] = True
        for nm, sub in (("ref75", (old / "submission.py").read_text()), ("new0", src)):
            d_ = Path(args.out + "_" + nm)
            if d_.exists():
                shutil.rmtree(d_)
            d_.mkdir(parents=True)
            for f in ["weights.pt"] + fm_files:
                shutil.copyfile(old / f, d_ / f)
            (d_ / "submission.py").write_text(sub)
            (d_ / "config.json").write_text(json.dumps(c75, indent=1))
        if args.tta_vp:
            import t2p_bench as B
            from ni26 import train as T
            rc_ = T.RECIPES[args.tta_vp_recipe]
            names = list(json.loads((Path(args.export) / "meta.json").read_text())["ch_names"])
            assert names == list(cfg["ch_names"]), "row 75: the bank is built on the training montage"
            bk = B.vp_bank(names, K=rc_.vp_k, sigma=0.005 * rc_.vp, gain=rc_.vp_g, shift=rc_.vp_s, vref=rc_.vp_r, cap=rc_.vp_cap)
            n_ = args.tta_vp - 1
            torch.save({k_: bk[k_][:n_].clone().cpu() for k_ in ("mats", "gains", "shifts")}, out / "tta_vp.pt")
            c75["cnn_tta_vp"] = {"k": args.tta_vp, "copies": bool(args.tta_copies), "file": "tta_vp.pt"}
            extra = ["tta_vp.pt"]
            print(f"[repack] row 75: tta_vp k {args.tta_vp} (copies {bool(args.tta_copies)}), the first {n_} draws of {args.tta_vp_recipe}'s bank", flush=True)
        (out / "config.json").write_text(json.dumps(c75, indent=1))
    if args.gpu_graph and not r75:          # T2-D208: the only config change; every weight file stays byte-identical
        c2 = json.loads((out / "config.json").read_text())
        c2["gpu_graph"] = True
        (out / "config.json").write_text(json.dumps(c2, indent=1))
        print("[repack] config gpu_graph = true (T2-D208)", flush=True)
    refs, copies_info, add_info, keep_info = {}, {}, {}, {}
    side_mix = None                              # T2-D549 R2: set by --side-from
    side_info, fm_side_on = None, False          # T2-D524 (W18): set by a donor with FM side heads
    instream_fm = bool(args.fm_from and cfg.get("instream"))   # T2-D534: a donor FM on a base that adapts in-stream (see the parts check)
    fm_adapt = bool(args.fm_from and any(cfg.get(k) for k in FM_ADAPT_KEYS))   # 2026-11-22: see reconfig_parts_gate
    fr0 = min(int(c) for c in (cfg.get("instream") or {}).get("refits", [60])) if fm_adapt else 0   # the first refit point (windows before it: inert)
    instream_moved = 0.0
    if reconfig:                               # T2-D210: ref = the old files + this template; dref = the donor's files + this template
        ref = Path(args.out + "_ref")
        if ref.exists():
            shutil.rmtree(ref)
        shutil.copytree(out, ref)
        refs["ref"] = ref
        c2 = json.loads((out / "config.json").read_text())
        if args.side_from:                       # T2-D549 R2: S_old -> S_new side nets (one change: no other edit, the old template)
            assert not (args.fm_from or args.fm_add_from or args.fm_keep or args.copies_from or args.set), "--side-from: its own repack (one change)"
            assert cfg.get("side") and "def _side_blend" in src, "--side-from needs a side-net package (template W17+)"
            assert (old / "submission.py").read_text() == (out / "submission.py").read_text(), \
                "--side-from: the template must be the old package's own, byte for byte (one change per repack)"
            sf_ = cfg["side"]["file"]
            bo_ = torch.load(old / sf_, map_location="cpu", weights_only=True)
            bn_ = torch.load(args.side_from, map_location="cpu", weights_only=True)
            for k_ in ("kind", "n0", "scale", "recipe", "export_sha"):
                assert bo_.get(k_) == bn_.get(k_), f"--side-from: {k_} differs ({bo_.get(k_)!r} vs {bn_.get(k_)!r})"
            for k_ in ("mu", "sd"):
                assert torch.equal(torch.as_tensor(bo_[k_]), torch.as_tensor(bn_[k_])), f"--side-from: the prior {k_} differs"
            fams_ = list(cfg["side"].get("families", ["cnn2", "cnn3"]))
            n_old_ = {fm_: len(bo_[fm_]) for fm_ in fams_}
            n_new_ = {fm_: len(bn_[fm_]) for fm_ in fams_}
            for fm_ in fams_:
                assert n_new_[fm_] > n_old_[fm_], f"--side-from: {fm_} has {n_new_[fm_]} nets, not more than the base's {n_old_[fm_]}"
                for i_ in range(n_old_[fm_]):
                    for part_ in ("net", "head"):
                        a_, b_ = bo_[fm_][i_][part_], bn_[fm_][i_][part_]
                        assert a_.keys() == b_.keys() and all(torch.equal(a_[k_], b_[k_]) for k_ in a_), f"--side-from: {fm_}[{i_}].{part_} != the base's"
            if "seeds" in bo_:
                assert list(bn_["seeds"][:len(bo_["seeds"])]) == list(bo_["seeds"]), (bo_["seeds"], bn_["seeds"])
            shutil.copyfile(args.side_from, out / sf_)
            bref = Path(args.out + "_bref")      # B = REF with ONLY the added nets (the same template, every other file the old one)
            if bref.exists():
                shutil.rmtree(bref)
            shutil.copytree(ref, bref)
            bb_ = dict(bn_, **{fm_: bn_[fm_][n_old_[fm_]:] for fm_ in fams_})
            if "seeds" in bn_:
                bb_["seeds"] = list(bn_["seeds"][len(bo_["seeds"]):])
            torch.save(bb_, bref / sf_)
            refs["bref"] = bref
            side_mix = dict(file=sf_, fams=fams_, n_old=n_old_, n_new=n_new_, side_from=str(args.side_from),
                            sha256=hashlib.sha256(Path(args.side_from).read_bytes()).hexdigest(), seeds=list(bn_.get("seeds", [])),
                            gate_T=0.0, windows=0)
            print(f"[repack] T2-D549 R2: side nets {n_old_} -> {n_new_} from {args.side_from} (sha256 {side_mix['sha256'][:16]}; the base's nets "
                  f"first, byte for byte; prior / scale equal); B = REF with the {[n_new_[k] - n_old_[k] for k in fams_]} added nets", flush=True)
        if args.side_replace:                    # T2-D580: S -> S' side nets of the SAME architecture (one change: no other edit, the old template)
            assert not (args.fm_from or args.fm_add_from or args.fm_keep or args.copies_from or args.set or args.side_from), "--side-replace: its own repack (one change)"
            assert cfg.get("side") and "def _side_blend" in src, "--side-replace needs a side-net package (template W17+)"
            assert (old / "submission.py").read_text() == (out / "submission.py").read_text(), \
                "--side-replace: the template must be the old package's own, byte for byte (one change per repack)"
            sf_ = cfg["side"]["file"]
            bo_ = torch.load(old / sf_, map_location="cpu", weights_only=True)
            bn_ = torch.load(args.side_replace, map_location="cpu", weights_only=True)
            for k_ in ("kind", "n0", "scale", "export_sha"):
                assert bo_.get(k_) == bn_.get(k_), f"--side-replace: {k_} differs ({bo_.get(k_)!r} vs {bn_.get(k_)!r})"
            for k_ in ("mu", "sd"):
                assert torch.equal(torch.as_tensor(bo_[k_]), torch.as_tensor(bn_[k_])), f"--side-replace: the prior {k_} differs"
            assert list(bo_.get("seeds", [])) == list(bn_.get("seeds", [])), ("--side-replace: the seeds differ", bo_.get("seeds"), bn_.get("seeds"))
            fams_ = list(cfg["side"].get("families", ["cnn2", "cnn3"]))
            for fm_ in fams_:
                assert len(bn_[fm_]) == len(bo_[fm_]), f"--side-replace: {fm_} has {len(bn_[fm_])} nets, not the base's {len(bo_[fm_])}"
                for i_ in range(len(bo_[fm_])):
                    for part_ in ("net", "head"):
                        a_, b_ = bo_[fm_][i_][part_], bn_[fm_][i_][part_]
                        assert a_.keys() == b_.keys() and all(tuple(a_[k_].shape) == tuple(b_[k_].shape) and a_[k_].dtype == b_[k_].dtype for k_ in a_), \
                            f"--side-replace: {fm_}[{i_}].{part_}: not the same architecture"
            assert bo_.get("recipe") != bn_.get("recipe"), ("--side-replace: the same recipe (nothing to replace)", bo_.get("recipe"))
            shutil.copyfile(args.side_replace, out / sf_)
            import filecmp
            f_old = sorted(str(x.relative_to(old)) for x in old.rglob("*") if x.is_file())
            f_out = sorted(str(x.relative_to(out)) for x in out.rglob("*") if x.is_file())
            assert f_old == f_out, ("--side-replace: the package's entries differ", sorted(set(f_old) ^ set(f_out)))
            for rel_ in f_old:
                if rel_ == sf_:
                    continue
                if rel_ == "config.json":
                    assert json.loads((old / rel_).read_text()) == json.loads((out / rel_).read_text()), "--side-replace: config.json changed"
                    continue
                assert filecmp.cmp(old / rel_, out / rel_, shallow=False), f"--side-replace: {rel_} is not byte-identical"
            side_mix = dict(mode="replace", file=sf_, fams=fams_, n_old={fm_: len(bo_[fm_]) for fm_ in fams_}, n_new={fm_: len(bn_[fm_]) for fm_ in fams_},
                            side_replace=str(args.side_replace), sha256=hashlib.sha256(Path(args.side_replace).read_bytes()).hexdigest(),
                            recipe_old=bo_.get("recipe"), recipe_new=bn_.get("recipe"), seeds=list(bn_.get("seeds", [])), others_identical=len(f_old) - 1,
                            gate_T=0.0, moved=0.0, windows=0)
            print(f"[repack] T2-D580: side nets REPLACED from {args.side_replace} (sha256 {side_mix['sha256'][:16]}; recipe {side_mix['recipe_old']} -> "
                  f"{side_mix['recipe_new']}; the same nets / keys / shapes / prior / scale / seeds; {len(f_old) - 1} other entries identical)", flush=True)
        if args.fm_from:
            dref = Path(args.out + "_dref")
            if dref.exists():
                shutil.rmtree(dref)
            dref.mkdir(parents=True)
            with zipfile.ZipFile(args.fm_from) as z:
                z.extractall(dref)
            dcfg = json.loads((dref / "config.json").read_text())
            assert len(dcfg.get("fm_members", [])) > 0 and len(cfg.get("fm_members", [])) > 0
            fm_side_on = any(m_.get("side") for m_ in dcfg["fm_members"])      # T2-D524 (W18): a donor whose members carry window-mean side heads
            if fm_side_on:
                assert "def set_fm_side" in src, "donor FM side heads need the W18 template (--template tracks/t2_bci/submission/t2p_submission_w18.py)"
                assert cfg.get("side"), "the FM side heads read the side nets' z: the base needs cfg['side'] (W17)"
                assert all(m_.get("weight") is None for m_ in dcfg["fm_members"]), "the side-head check assumes the equal-weight FM mean"
                side_blob_ = torch.load(old / cfg["side"]["file"], map_location="cpu", weights_only=True)
                side_info = dict(heads=[None if not m_.get("side") else
                                        torch.load(dref / m_["side"]["file"], map_location="cpu", weights_only=True) for m_ in dcfg["fm_members"]],
                                 mu=np.asarray(side_blob_["mu"], np.float64), sd=np.asarray(side_blob_["sd"], np.float64),
                                 n0=float(side_blob_.get("n0", cfg["side"].get("n0", 10.0))), in_vs_donor=0.0, head_vs_indep=0.0, windows=0)
                for h_ in side_info["heads"]:
                    if h_ is not None:
                        assert np.abs(np.asarray(h_["mu"], np.float64) - side_info["mu"]).max() <= 1e-9 and \
                            np.abs(np.asarray(h_["sd"], np.float64) - side_info["sd"]).max() <= 1e-9, "donor head prior != the base side.pt's"
            same_n = len(dcfg["fm_members"]) == len(cfg["fm_members"])
            # T2-D512: a donor may differ in TRAINING-only keys (its seed, the tilt augmentation) that no template reads at inference;
            # T2-D549: + 'cont' (D546b's record of the low-LR continuation that trained the member: source, epochs, lr, data seed)
            # T2-D557: + 'mixup' / 'aug' (the FM training augmentation; W20's augmented donors), never read at inference
            # the night plan 2026-11-18 (the Fable review's blocker): + 'vp' (D582) / 'recomb' (D583) / 'phase' (D586) / 'cons' (D588)
            train_only = ("seed", "tilt", "cont", "mixup", "aug", "vp", "recomb", "phase", "cons", "wmargin", "user_balance", "user_balance_pow", "grl")   # T2-D642 / D647 / D649
            assert not re.search(r'\bm_?\s*\[\s*"(seed|tilt|cont|mixup|aug|vp|recomb|phase|cons|wmargin|user_balance|user_balance_pow|grl)"|\bm_?\.get\(\s*"(seed|tilt|cont|mixup|aug|vp|recomb|phase|cons|wmargin|user_balance|user_balance_pow|grl)"|"(seed|tilt|cont|mixup|aug|vp|recomb|phase|cons|wmargin|user_balance|user_balance_pow|grl)" in m\b', src), \
                "the template reads an FM member's seed / tilt / cont / mixup / aug / vp / recomb / phase / cons / wmargin / user_balance / user_balance_pow: not training-only"
            # T2-D512: the member's input reference as the template derives it (W14: m.get("ref") or ("car" if m.get("car") else None)):
            # a W14-merged base says ref 'car', a t2p_final donor says car True; the same input. 'source' = the merge's provenance.
            eff_ref = lambda m: m.get("ref") or ("car" if m.get("car") else None)  # noqa: E731
            if same_n:
                pairs = list(zip(cfg["fm_members"], dcfg["fm_members"]))
            else:                                # T2-D512 build 2: an equal-weight FM slot of another size (M7r2-W14: 5 + 3) takes the donor's WHOLE slot (3 + 3,
                assert all(m.get("weight") is None for m in cfg["fm_members"] + dcfg["fm_members"]), \
                    "a different FM member count needs the equal-weight FM mean (no member weights)"   # D512's read shape); each donor member is
                pairs = []                       # compared with the base's first member of the same input reference
                for b_ in dcfg["fm_members"]:
                    pa = [a_ for a_ in cfg["fm_members"] if eff_ref(a_) == eff_ref(b_)]
                    assert pa, f"donor member {b_['file']}: no base member with input reference {eff_ref(b_)}"
                    pairs.append((pa[0], b_))
                print(f"[repack] donor FM slot of {len(dcfg['fm_members'])} members replaces the base's {len(cfg['fm_members'])} (equal-weight mean)", flush=True)
            for a_, b_ in pairs:
                assert eff_ref(a_) == eff_ref(b_), f"donor member {b_['file']}: input reference {eff_ref(b_)} != the base's {eff_ref(a_)}"
                skip = ("provenance", "cpt", "source", "file", "ref", "car") + train_only + (("side",) if fm_side_on else ())   # T2-D524
                diff = sorted(k for k in set(a_) | set(b_) if k not in skip and a_.get(k) != b_.get(k))
                assert not diff, f"donor FM member config differs in {diff}"
                tdiff = {k: (a_.get(k), b_.get(k)) for k in train_only + ("file", "ref", "car", "source") if a_.get(k) != b_.get(k)}
                if tdiff:
                    print(f"[repack] donor member {b_['file']} -> slot {a_['file']}: keys differ {tdiff} (allowed: training-only / naming / "
                          f"the same input reference {eff_ref(a_)})", flush=True)
            if args.donor_check_only:            # T2-D549: the donor checks above passed; nothing is written
                print(f"[repack] DONOR CHECK OK: {args.fm_from} vs {args.zip}: {len(pairs)} member pairs, keys / input references / "
                      f"side priors as the build would check them", flush=True)
                return
            (dref / "submission.py").write_text(src)
            if fm_side_on:                       # T2-D524: dref = the donor WITHOUT its heads (its FM part = the members' plain mean, as before)
                dnh = dict(dcfg, fm_members=[{k_: v_ for k_, v_ in m_.items() if k_ != "side"} for m_ in dcfg["fm_members"]])
                dnh.pop("fm_side", None)
                (dref / "config.json").write_text(json.dumps(dnh, indent=1))
            refs["dref"] = dref
            new_members = []
            if same_n:
                for a_, m_ in zip(cfg["fm_members"], dcfg["fm_members"]):   # the donor's member k takes the BASE's slot-k file name (fm_files = the base's)
                    shutil.copyfile(dref / m_["file"], out / a_["file"])
                    assert (dref / m_["file"]).read_bytes() == (out / a_["file"]).read_bytes(), m_["file"]
                    new_members.append(dict(m_, file=a_["file"]))
                    if m_.get("side"):                   # T2-D524: the member's head follows it under its slot's name
                        sf_ = f"{Path(a_['file']).stem}_side.pt"
                        shutil.copyfile(dref / m_["side"]["file"], out / sf_)
                        assert (dref / m_["side"]["file"]).read_bytes() == (out / sf_).read_bytes(), m_["side"]["file"]
                        new_members[-1]["side"] = dict(m_["side"], file=sf_)
            else:                                # the base's FM files leave; the donor's enter under their own names; the zip list follows the new slot
                old_fm = [m_["file"] for m_ in cfg["fm_members"]]
                for f_ in old_fm:
                    (out / f_).unlink()
                for m_ in dcfg["fm_members"]:
                    shutil.copyfile(dref / m_["file"], out / m_["file"])
                    assert (dref / m_["file"]).read_bytes() == (out / m_["file"]).read_bytes(), m_["file"]
                    new_members.append(dict(m_))
                    if m_.get("side"):                   # T2-D524: the member's head under the donor's own name
                        shutil.copyfile(dref / m_["side"]["file"], out / m_["side"]["file"])
                fm_files = [m_["file"] for m_ in new_members] + [f_ for f_ in fm_files if f_ not in old_fm]
                assert len(set(fm_files)) == len(fm_files), fm_files
            c2["fm_members"] = new_members
            base_side = [m_["side"]["file"] for m_ in cfg["fm_members"] if m_.get("side")]   # a W18 base's heads leave with their members
            fm_files = [f_ for f_ in fm_files if f_ not in base_side]
            if fm_side_on:                           # T2-D524 (W18): the switch on; the head files enter the zip
                c2["fm_side"] = True
                hf_ = [m_["side"]["file"] for m_ in new_members if m_.get("side")]
                assert len(set(hf_)) == len(hf_) and not set(hf_) & set(fm_files), hf_
                extra += hf_
                print(f"[repack] T2-D524: {len(hf_)} FM side heads from the donor -> {hf_}; cfg fm_side = true", flush=True)
            elif base_side:
                c2.pop("fm_side", None)              # a head-free donor replaces a W18 base's headed members
            for f_ in base_side:
                if not (fm_side_on and f_ in hf_):
                    (out / f_).unlink()
            print(f"[repack] FM member files from the donor {args.fm_from}: {[m_['file'] for m_ in dcfg['fm_members']]} -> "
                  f"{[m_['file'] for m_ in new_members]}", flush=True)
        if args.fm_add_from:                   # T2-D482 (W14): append donor members (CAR / Laplacian views) to the base's FM slot
            assert not args.fm_from and "car_windows" in src, "--fm-add-from needs template W14 and no --fm-from"
            assert all(m_.get("weight") is None for m_ in cfg["fm_members"]), "appending assumes the equal-weight FM mean"
            addd = Path(args.out + "_add")
            if addd.exists():
                shutil.rmtree(addd)
            addd.mkdir(parents=True)
            with zipfile.ZipFile(args.fm_add_from) as z:
                z.extractall(addd)
            dcfg = json.loads((addd / "config.json").read_text())
            seeds_ = [int(v) for v in args.fm_add_seeds.split(",")]
            sel = [m_ for m_ in dcfg["fm_members"] if m_["seed"] in seeds_]
            assert [m_["seed"] for m_ in sel] == seeds_, (seeds_, [m_["seed"] for m_ in dcfg["fm_members"]])
            for m_ in sel:
                assert (m_.get("ref") or ("car" if m_.get("car") else None)) in ("car", "lap"), "an added member must carry its reference"
                for k_ in ("family", "fm", "size", "n_samples", "calib", "layer", "chan_idx"):
                    assert m_.get(k_) == cfg["fm_members"][0].get(k_), f"added member {m_['seed']}: {k_} differs from the base's"
            nb_ = len(c2["fm_members"])
            new_m = []
            for j_, m_ in enumerate(sel):
                f_ = f"fm_{nb_ + j_}.pt"
                shutil.copyfile(addd / m_["file"], out / f_)
                new_m.append(dict(m_, file=f_, ref=m_.get("ref") or "car", source=dict(zip=args.fm_add_from, file=m_["file"])))
            extra += [m_["file"] for m_ in new_m]
            shutil.rmtree(addd)                              # the donor's files now live in out (and below in cref)
            cref = Path(args.out + "_cref")                  # the check's reference: the base's CNNs + ONLY the added members, this template
            if cref.exists():
                shutil.rmtree(cref)
            shutil.copytree(ref, cref)
            for m_ in new_m:
                shutil.copyfile(out / m_["file"], cref / m_["file"])
            cc = json.loads((cref / "config.json").read_text())
            cc["fm_members"] = new_m
            (cref / "config.json").write_text(json.dumps(cc, indent=1))
            refs["cref"] = cref
            c2["fm_members"] = c2["fm_members"] + new_m
            add_info = dict(nb=nb_, na=len(new_m), seeds=seeds_, zip=args.fm_add_from,
                            zip_sha256=hashlib.sha256(Path(args.fm_add_from).read_bytes()).hexdigest())
            print(f"[repack] T2-D482: {len(new_m)} donor members appended ({seeds_}, refs {[m_['ref'] for m_ in new_m]}): {nb_} -> "
                  f"{len(c2['fm_members'])} FM members, equal weight {float(c2['fm_weight']) / len(c2['fm_members']):.5f} each", flush=True)
        if args.fm_keep:                       # T2-D496 (W14-lite): a member SUBSET of the base's FM slot (equal-weight mean over the kept)
            assert not (args.fm_from or args.fm_add_from), "--fm-keep is its own member edit (no --fm-from / --fm-add-from in the same repack)"
            assert all(m_.get("weight") is None for m_ in cfg["fm_members"]), "--fm-keep assumes the equal-weight FM mean"
            kidx = [int(v) for v in args.fm_keep.split(",")]
            nb_ = len(c2["fm_members"])
            assert kidx == sorted(set(kidx)) and 0 <= kidx[0] and kidx[-1] < nb_ and 0 < len(kidx) < nb_, (kidx, nb_)
            assert [m_["file"] for m_ in c2["fm_members"]] == [m_["file"] for m_ in cfg["fm_members"]], "--fm-keep acts on the base's members"
            dropped = [m_["file"] for j_, m_ in enumerate(c2["fm_members"]) if j_ not in kidx]
            kref = Path(args.out + "_kref")                  # the check's reference: REF (this template, the OLD config) with ONLY the kept members
            if kref.exists():
                shutil.rmtree(kref)
            shutil.copytree(ref, kref)
            cc = json.loads((kref / "config.json").read_text())
            cc["fm_members"] = [cc["fm_members"][j_] for j_ in kidx]
            (kref / "config.json").write_text(json.dumps(cc, indent=1))
            refs["kref"] = kref
            c2["fm_members"] = [c2["fm_members"][j_] for j_ in kidx]
            for f_ in dropped:
                (out / f_).unlink()
            fm_files = [f_ for f_ in fm_files if f_ not in dropped]
            assert (old / "submission.py").read_text() == (out / "submission.py").read_text(), \
                "--fm-keep: the new template must be the old package's own, byte for byte (the inert-template check is then skipped)"
            refs.pop("ref")                                  # GPU memory (the smoke OOM'd with old 8 + new 6 + ref 8 + kref 6 ST-L): kref is the
            shutil.rmtree(ref)                               # only reference; OLD is not loaded (its template is the new one, byte for byte)
            keep_info = dict(kept=kidx, dropped=dropped, n_before=nb_, n_after=len(kidx), inert_check="skipped: old submission.py == new's",
                             kept_refs=[m_.get("ref") for m_ in c2["fm_members"]], kept_seeds=[m_.get("seed") for m_ in c2["fm_members"]])
            print(f"[repack] T2-D496: FM members {nb_} -> {len(kidx)} (kept {kidx}, refs {keep_info['kept_refs']}, seeds {keep_info['kept_seeds']}); "
                  f"dropped {dropped}; equal weight {float(c2['fm_weight']) / len(kidx):.5f} each", flush=True)
        if args.copies_from:                   # P2 (S1): the donor's copies; cdon = the donor package as built (its own template)
            cdon = Path(args.out + "_cdon")
            if cdon.exists():
                shutil.rmtree(cdon)
            cdon.mkdir(parents=True)
            with zipfile.ZipFile(args.copies_from) as z:
                z.extractall(cdon)
            dcfg = json.loads((cdon / "config.json").read_text())
            new_ub, cfiles = merge_copies(c2, dcfg)
            for f in cfiles:
                assert (cdon / f).read_bytes() != (old / f).read_bytes(), f"copies donor: {f} is byte-identical to the base's (nothing to probe)"
                shutil.copyfile(cdon / f, out / f)
            c2.update(new_ub)
            refs["cdon"] = cdon
            for f in ("weights.pt", c2["cnn2"]["file"]):          # descriptive: the donor's pooled nets vs the base's (kept)
                ta, tb = _tensors(torch.load(old / f, map_location="cpu")), _tensors(torch.load(cdon / f, map_location="cpu"))
                assert len(ta) == len(tb) and all(x.shape == y.shape for x, y in zip(ta, tb)), f"copies donor: {f} has another structure"
                dmax = max((float((x.double() - y.double()).abs().max()) for x, y in zip(ta, tb) if x.is_floating_point()), default=0.0)
                copies_info.setdefault("pooled_param_max_abs", {})[f] = dmax
                print(f"[repack] copies donor: pooled {f} max |dθ| {dmax:.2e} over {len(ta)} / {len(tb)} tensors (the base's are kept)", flush=True)
            copies_info.update(zip=args.copies_from, zip_sha256=hashlib.sha256(Path(args.copies_from).read_bytes()).hexdigest(),
                               recipe=dcfg["recipe"], recipe_pu2=dcfg["user_blend2"].get("recipe_pu"), files=cfiles)
            print(f"[repack] per-user copies from the donor {args.copies_from} ({dcfg['recipe']} / {dcfg['user_blend2'].get('recipe_pu')}): {cfiles}", flush=True)
        for kv in args.set:
            k_, was, now = set_cfg(c2, kv)
            print(f"[repack] config {k_}: {was} -> {now}", flush=True)
        if args.set_ref:                       # T2-D500: REF carries the same edits (its template is asserted to be the old one's)
            assert args.set and not (args.fm_from or args.fm_add_from or args.fm_keep or args.copies_from), "--set-ref: --set edits only"
            assert (old / "submission.py").read_text() == (out / "submission.py").read_text(), \
                "--set-ref: the new template must be the old package's own, byte for byte"
            rc_ = json.loads((ref / "config.json").read_text())
            for kv in args.set:
                set_cfg(rc_, kv)
            (ref / "config.json").write_text(json.dumps(rc_, indent=1))
            refs.pop("ref")                      # GPU memory (old + new + ref = 18 ST-L OOM'd on 32 GB): REF would equal NEW byte for byte, so
            shutil.rmtree(ref)                   # NEW serves as REF (its parts check is then trivially 0); the checks that bind are the
            print(f"[repack] T2-D500: REF = NEW with the same {len(args.set)} edit(s) (not loaded twice); the binding checks: OLD vs NEW "
                  "(the edit took), the independent decode recombination, robustness, kit replay", flush=True)
        check_weights(c2)
        mo, mn = [m.get("weight") for m in cfg.get("fm_members", [])], [m.get("weight") for m in c2.get("fm_members", [])]
        if not args.fm_from and len(mo) == len(mn) > 1 and all(mo) and all(mn):   # 11-01: refuse early, not after the clean pass
            assert np.abs(np.asarray(mo) / sum(mo) - np.asarray(mn) / sum(mn)).max() < 1e-9, \
                f"the FM members' ratio changed ({mo} -> {mn}): the parts check cannot verify it; move fm_weight with the ratios kept"
        (out / "config.json").write_text(json.dumps(c2, indent=1))
    zp = Path(str(out) + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["submission.py", "config.json", "weights.pt"] + fm_files + extra:
            z.write(out / f, f)
    zsha = hashlib.sha256(zp.read_bytes()).hexdigest()
    print(f"[repack] {args.zip} -> {zp} (sha256 {zsha[:16]}, {zp.stat().st_size / 1e6:.1f} MB)", flush=True)
    if side_mix is not None and side_mix.get("mode") == "replace":   # T2-D580 (Track3 F1 on 595c1844): the identity re-checked on the WRITTEN zip vs OLD's
        with zipfile.ZipFile(args.zip) as zo_, zipfile.ZipFile(zp) as zn_:
            no_, nn_ = sorted(zo_.namelist()), sorted(zn_.namelist())
            assert no_ == nn_, ("--side-replace: the written zip's entries differ from OLD's", sorted(set(no_) ^ set(nn_)))
            for nm_ in no_:
                if nm_ == side_mix["file"]:
                    assert zo_.read(nm_) != zn_.read(nm_), "--side-replace: the written side file equals OLD's"
                elif nm_ == "config.json":
                    assert json.loads(zo_.read(nm_)) == json.loads(zn_.read(nm_)), "--side-replace: the written config.json differs (parsed)"
                else:
                    assert zo_.read(nm_) == zn_.read(nm_), f"--side-replace: the written {nm_} is not byte-identical to OLD's"
        side_mix["zip_entries_identical"] = len(no_) - 1
        print(f"[repack] T2-D580: the written zip = OLD's {len(no_) - 1} other entries byte for byte (config.json parsed), only {side_mix['file']} replaced", flush=True)

    E = Path(args.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    kit.use_track("bci_decoding")
    meta = dict(device=args.device, ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"],
                n_chans=len(meta_e["ch_names"]), n_times=int(X.shape[-1]), n_classes=3)
    m_new = load(out, "new", meta)
    m_new0 = load(Path(args.out + "_new0"), "new0", meta) if r75 else None   # row 75: must equal REF (= m_old here) exactly
    n_diff0, dp0 = 0, 0.0
    if keep_info:                                   # T2-D496: kref (the kept members, OLD's template and CNN files) serves as REF and OLD
        m_ref = {"ref": load(refs["kref"], "kref", meta)}
        m_old = m_ref["ref"]
    elif args.set_ref:                              # T2-D500: REF = NEW (same template, same edits); OLD is loaded for OLD vs NEW
        m_old, m_ref = load(old, "old", meta), {"ref": m_new}
    else:
        m_old = load(old if not r75 else Path(args.out + "_ref75"), "old", meta)
        m_ref = {k: load(d, k, meta) for k, d in refs.items()}
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    if args.max_streams:
        streams = streams[:args.max_streams]

    # ---- 2. clean equality on every public stream (T2-D210 reconfig: OLD vs REF = the template change is inert; NEW is checked by parts)
    pk_new = np.full(len(idx), -1)
    n_diff, n_win, fin_all, dp_max, t_new = 0, 0, True, 0.0, 0.0
    parts_dev = dict(p1=0.0, p2=0.0, pf=0.0, recombine=0.0)
    parts_pre = dict(pf=0.0, recombine=0.0, in_vs_donor=0.0, head_vs_indep=0.0, windows=0)   # 2026-11-22: t < fr0 (fm_adapt only)
    n_new_vs_old, set_moved, set_dp = 0, 0, 0.0
    if reconfig:
        c2 = json.loads((out / "config.json").read_text())
        fmw, w2_ = float(c2.get("fm_weight", 0.0)), float(c2.get("cnn2", {}).get("weight", 0.0))
        w3_ = float((c2.get("cnn3") or {}).get("weight", 0.0))           # T2-D352 (W11): 0.0 = the earlier arithmetic exactly
        tt = c2.get("tta") or {}
    for g in streams:
        xs = np.asarray(X[g["i"].to_numpy()], np.float32)
        pa, pb = [], []
        a, _ = run_stream(m_old if not reconfig else m_ref["ref"], xs, args.device, pa)
        if r75:                                  # row 75: the template change is inert with tv off
            p0 = []
            a0, _ = run_stream(m_new0, xs, args.device, p0)
            n_diff0 += int((a0 != a).sum())
            if pa and p0:
                dp0 = max(dp0, float(np.abs(np.stack(pa) - np.stack(p0)).max()))
        if reconfig:
            if not keep_info:                   # T2-D496: skipped with --fm-keep (OLD's template == NEW's, asserted at build)
                p0l = [] if args.set_ref else None
                a0, _ = run_stream(m_old, xs, args.device, p0l)            # the inert-template check: OLD vs REF
                if not args.set_ref:                # T2-D500: REF carries the edit; OLD vs NEW is counted below instead
                    n_diff += int((a0 != a).sum())
            parts = {}
            head_rec = [] if side_info is not None else None
            for key, mm_ in [("new", m_new)] + list(m_ref.items()):
                parts[key] = run_parts(mm_, xs, args.device, rec=head_rec if key == "new" else None)
            pn, pr_ = parts["new"], parts["ref"]
            if fm_adapt:
                parts_pre["windows"] += min(len(xs), fr0)
            pd_ = parts.get("dref", pr_)
            if side_mix is not None and side_mix.get("mode") == "replace":   # T2-D580: the side families' parts move (p1 / pf stay bound below)
                for t_ in range(len(xs)):
                    for fm_, k_ in (("cnn2", 1), ("cnn3", 3)):
                        if fm_ not in side_mix["fams"] or len(pn[1][t_]) <= k_ or pn[1][t_][k_] is None:
                            continue
                        assert np.isfinite(np.asarray(pn[1][t_][k_], np.float64)).all(), "T2-D580: a non-finite side-family part"
                        side_mix["moved"] = max(side_mix["moved"], float(np.abs(np.asarray(pn[1][t_][k_], np.float64) - np.asarray(pr_[1][t_][k_], np.float64)).max()))
                side_mix["windows"] += len(xs)
            elif side_mix is not None:           # T2-D549 Gate T: part_new = (n_old part_ref + n_add part_B) / n_new (the pre-side parts are shared)
                pb_ = parts["bref"]
                for t_ in range(len(xs)):
                    for fm_, k_ in (("cnn2", 1), ("cnn3", 3)):
                        if fm_ not in side_mix["fams"] or len(pn[1][t_]) <= k_ or pn[1][t_][k_] is None:
                            continue
                        wo_ = side_mix["n_old"][fm_] / side_mix["n_new"][fm_]
                        mix_ = wo_ * np.asarray(pr_[1][t_][k_], np.float64) + (1.0 - wo_) * np.asarray(pb_[1][t_][k_], np.float64)
                        side_mix["gate_T"] = max(side_mix["gate_T"], float(np.abs(np.asarray(pn[1][t_][k_], np.float64) - mix_).max()))
                side_mix["windows"] += len(xs)
            if side_info is not None:            # T2-D524 (W18): NEW's FM part = the donor's members through the donor's heads, recomputed here
                import t2p_bench as B_
                assert len(head_rec) == len(xs), (len(head_rec), len(xs))
                Fs_ = B_.side_raw(xs, "dcmean")          # the side vector of the DELIVERED windows, causal z with side.pt's prior (t2p_bench.side_z's form)
                mu_, sd_, n0_ = side_info["mu"], side_info["sd"], side_info["n0"]
                xf_ = np.where(np.isfinite(Fs_), Fs_, mu_)
                nn_ = n0_ + np.arange(1, len(xs) + 1)[:, None]
                mm2 = (n0_ * mu_ + np.cumsum(xf_, 0)) / nn_
                vv_ = (n0_ * (sd_ ** 2 + mu_ ** 2) + np.cumsum(xf_ * xf_, 0)) / nn_ - mm2 ** 2
                zs_ = ((xf_ - mm2) / np.sqrt(np.maximum(vv_, 1e-12))).astype(np.float32).astype(np.float64)
                pfi = []
                for t_, (pin, pout) in enumerate(head_rec):
                    assert len(pin) == len(side_info["heads"]), (len(pin), len(side_info["heads"]))
                    side_info["in_vs_donor"] = max(side_info["in_vs_donor"], float(np.abs(np.mean(pin, 0) - pd_[1][t_][2]).max()))
                    if fm_adapt and t_ < fr0:
                        parts_pre["in_vs_donor"] = max(parts_pre["in_vs_donor"], float(np.abs(np.mean(pin, 0) - pd_[1][t_][2]).max()))
                    ex_ = []
                    for p_, h_ in zip(pin, side_info["heads"]):
                        if h_ is not None:
                            lg_ = np.log(np.maximum(p_, 1e-30)) + np.asarray(h_["weight"], np.float64) @ zs_[t_] + np.asarray(h_["bias"], np.float64)
                            e_ = np.exp(lg_ - lg_.max())
                            p_ = e_ / e_.sum()
                        ex_.append(p_)
                    side_info["head_vs_indep"] = max(side_info["head_vs_indep"], float(np.abs(np.stack(ex_) - np.stack(pout)).max()))
                    if fm_adapt and t_ < fr0:
                        parts_pre["head_vs_indep"] = max(parts_pre["head_vs_indep"], float(np.abs(np.stack(ex_) - np.stack(pout)).max()))
                    pfi.append(np.mean(ex_, 0))
                side_info["windows"] += len(xs)
                pd_ = (None, [[None, None, pfi[t_]] for t_ in range(len(xs))])   # the parts check and the recombination read this FM part
            if "cref" in parts:                 # T2-D482: the merged FM part = the equal-weight mix of the base's and the added members' parts
                pc2 = parts["cref"]
                pd_ = (None, [[None, None, (add_info["nb"] * np.asarray(pr_[1][t_][2], np.float64) + add_info["na"] * np.asarray(pc2[1][t_][2], np.float64))
                                / (add_info["nb"] + add_info["na"])] for t_ in range(len(xs))])
            if keep_info:                       # T2-D496: REF = kref, so pr_ / pd_ are kref's parts (CNN parts and the kept members' FM mix)
                assert set(parts) == {"new", "ref"}, sorted(parts)
            pc_ = parts.get("cdon")                 # P2: the CNN parts carry the donor's copies -> compared with the donor as built
            # T2-D534 (W18 on W17): the base's in-stream copies refit on the STACK's own decoded labels, so a donor FM moves those labels and
            # the adapted CNN parts with them. The CNN parts are then compared with in-stream OFF on both sides (the frozen nets must be equal),
            # and the recombination below reads NEW's own in-stream parts.
            pcmp_n, pcmp_r = pn, pr_
            if instream_fm:
                pcmp_n, pcmp_r = run_parts(m_new, xs, args.device, static=True), run_parts(m_ref["ref"], xs, args.device, static=True)
                for t_ in range(len(xs)):
                    for k_ in (1, 3):
                        if len(pn[1][t_]) > k_ and pn[1][t_][k_] is not None:
                            instream_moved = max(instream_moved, float(np.abs(pn[1][t_][k_] - pr_[1][t_][k_]).max()))
            for t_ in range(len(xs)):
                if pc_ is None:
                    parts_dev["p1"] = max(parts_dev["p1"], float(np.abs(pcmp_n[1][t_][0] - pcmp_r[1][t_][0]).max()))
                    if pn[1][t_][1] is not None and side_mix is None:      # T2-D549: with --side-from, Gate T binds the side families
                        parts_dev["p2"] = max(parts_dev["p2"], float(np.abs(pcmp_n[1][t_][1] - pcmp_r[1][t_][1]).max()))
                else:
                    cd = copies_info.setdefault("parts_vs_donor", dict(p1=0.0, p2=0.0))
                    cd["p1"] = max(cd["p1"], float(np.abs(pn[1][t_][0] - pc_[1][t_][0]).max()))
                    if pn[1][t_][1] is not None:
                        cd["p2"] = max(cd["p2"], float(np.abs(pn[1][t_][1] - pc_[1][t_][1]).max()))
                if pn[1][t_][2] is not None:
                    parts_dev["pf"] = max(parts_dev["pf"], float(np.abs(pn[1][t_][2] - pd_[1][t_][2]).max()))
                    if fm_adapt and t_ < fr0:
                        parts_pre["pf"] = max(parts_pre["pf"], float(np.abs(pn[1][t_][2] - pd_[1][t_][2]).max()))
                if len(pn[1][t_]) > 3 and pc_ is None and side_mix is None:      # T2-D352: the third family's part (T2-D549: Gate T with --side-from)
                    parts_dev["p3"] = max(parts_dev.get("p3", 0.0), float(np.abs(pcmp_n[1][t_][3] - pcmp_r[1][t_][3]).max()))
            sk = tt.get("kind") == "sinkhorn"       # T2-D286: SK10's reference = t2p_report.causal_sinkhorn (independent of the template)
            pc = None if sk else (type(m_new.prior)(tt["alpha"], tt["n0"], 3) if getattr(m_new, "prior", None) is not None else None)   # the package's own PriorCorrector
            Q = []
            pcnn = pn if (pc_ is not None or instream_fm or side_mix is not None) else pr_      # P2: the new CNN parts (the copies changed); T2-D534: NEW's in-stream parts; T2-D549: NEW's side-mixed parts
            for t_ in range(len(xs)):
                p1, p2, pf = pcnn[1][t_][0], pcnn[1][t_][1], pd_[1][t_][2]
                p3 = pcnn[1][t_][3] if len(pcnn[1][t_]) > 3 else None
                q = p1
                if p2 is not None:
                    w2n = w2_ / (1.0 - fmw - w3_)
                    q = (1.0 - w2n) * q + w2n * p2
                if p3 is not None:                      # T2-D352
                    w3n = w3_ / (1.0 - fmw)
                    q = (1.0 - w3n) * q + w3n * p3
                if pf is not None:
                    q = (1.0 - fmw) * q + fmw * pf
                if not np.isfinite(q).all():
                    q = np.full(len(q), 1.0 / len(q))
                Q.append(q)
                if not sk:
                    q = pc(q) if pc is not None else q
                    parts_dev["recombine"] = max(parts_dev["recombine"], float(np.abs(q - pn[0][t_]).max()))
                    if fm_adapt and t_ < fr0:
                        parts_pre["recombine"] = max(parts_pre["recombine"], float(np.abs(q - pn[0][t_]).max()))
            if sk and tt.get("per_family"):        # T2-D476 (W13): each family's part through causal_sinkhorn(its start), then the weighted pool
                from t2p_report import causal_sinkhorn
                key_ = (np.zeros(len(xs)), np.arange(len(xs)))
                fw = {"cnn": 1.0 - fmw - w2_ - w3_, "cnn2": w2_, "cnn3": w3_, "fm": fmw}
                seq = {"cnn": [pcnn[1][t_][0] for t_ in range(len(xs))],
                       "cnn2": [pcnn[1][t_][1] for t_ in range(len(xs))] if pcnn[1][0][1] is not None else None,
                       "cnn3": [pcnn[1][t_][3] for t_ in range(len(xs))] if len(pcnn[1][0]) > 3 else None,
                       "fm": [pd_[1][t_][2] for t_ in range(len(xs))] if pd_[1][0][2] is not None else None}
                assert sorted(f for f, v in seq.items() if v is not None) == sorted(tt["per_family"]), (sorted(tt["per_family"]), seq.keys())
                rs_ = "start" if c2.get("p1_prior_reset") else ("mean" if c2.get("p1_prior_informed") else None)   # T2-D595 / D603 packages
                assert not (c2.get("p1_prior_reset") and c2.get("p1_prior_informed")), "p1_prior_reset and p1_prior_informed together"
                isf_ = set(((c2.get("instream") or {}).get("families") or {})) if rs_ else set()
                cks_ = sorted(int(c) for c in ((c2.get("instream") or {}).get("refits") or [])) if rs_ else []

                def _dec_f(f):                      # the P1 families of a D595 / D603 package restart at the fired refits, as the template does
                    if rs_ and f in isf_:
                        return _sinkhorn_restart(np.stack(seq[f]).astype(np.float64), tt["iters"], tt["n0"], float(tt.get("lam", 1.0)), tt["per_family"][f], cks_, rs_)
                    return causal_sinkhorn(np.stack(seq[f]).astype(np.float64), key_, tt["iters"], tt["n0"], float(tt.get("lam", 1.0)), tt["per_family"][f])
                Qs = sum(fw[f] * _dec_f(f) for f in tt["per_family"])
                if c2.get("dcfam"):                 # T2-D516 (W16): + the window-mean family through its own causal_sinkhorn (the npz start)
                    assert all(v is not None for v in pn[2]), "W16: the window-mean family's per-window probability is missing"
                    w_dc = float(c2["dcfam"]["weight"])
                    st_dc = np.asarray(np.load(out / c2["dcfam"]["file"])["start"], np.float64)
                    Qs = (1.0 - w_dc) * Qs + w_dc * causal_sinkhorn(np.stack(pn[2]), key_, tt["iters"], tt["n0"], float(tt.get("lam", 1.0)), st_dc)
                if c2.get("imcoh"):                 # T2-D525 (W18): + the imcoh family through its own causal_sinkhorn (the npz start), after the window means
                    assert all(v is not None for v in pn[3]), "W18: the imcoh family's per-window probability is missing"
                    w_ic = float(c2["imcoh"]["weight"])
                    st_ic = np.asarray(np.load(out / c2["imcoh"]["file"])["start"], np.float64)
                    Qs = (1.0 - w_ic) * Qs + w_ic * causal_sinkhorn(np.stack(pn[3]), key_, tt["iters"], tt["n0"], float(tt.get("lam", 1.0)), st_ic)
                parts_dev["recombine"] = max(parts_dev["recombine"], float(np.abs(Qs - np.stack(pn[0])).max()))
                if fm_adapt and fr0 > 0:            # causal decode: its windows before the first refit read only inert parts
                    parts_pre["recombine"] = max(parts_pre["recombine"], float(np.abs(Qs - np.stack(pn[0]))[:fr0].max()))
            elif sk:
                from t2p_report import causal_sinkhorn
                Qs = causal_sinkhorn(np.stack(Q), (np.zeros(len(Q)), np.arange(len(Q))), tt["iters"], tt["n0"], float(tt.get("lam", 1.0)), tt.get("start"))
                parts_dev["recombine"] = max(parts_dev["recombine"], float(np.abs(Qs - np.stack(pn[0])).max()))
                if fm_adapt and fr0 > 0:            # causal decode: its windows before the first refit read only inert parts
                    parts_pre["recombine"] = max(parts_pre["recombine"], float(np.abs(Qs - np.stack(pn[0]))[:fr0].max()))
        t1 = time.time()
        b, fin = run_stream(m_new, xs, args.device, pb)
        t_new += time.time() - t1
        if pa and pb:
            dp_max = max(dp_max, float(np.abs(np.stack(pa) - np.stack(pb)).max()))
        if not reconfig:
            n_diff += int((a != b).sum())
        else:
            n_new_vs_old += int((a != b).sum())
        if args.set_ref:                         # argmax alone cannot show it on an all-user package (public windows are in-sample)
            set_moved += int((a0 != b).sum())
            set_dp = max(set_dp, float(np.abs(np.stack(p0l) - np.stack(pb)).max()))
        n_win += len(a)
        fin_all &= fin
        pk_new[g["i"].to_numpy()] = b
        print(f"[repack] clean {g['stream'].iloc[0]} ({g['split'].iloc[0]}): {len(a)} windows, {int((a != b).sum())} differ",
              flush=True)
    te = (idx["split"] == "test").to_numpy() & (pk_new >= 0)
    y, cells = idx["y"].to_numpy().astype(int), idx["cell"].to_numpy()
    test_cellba = metrics.cell_balanced_accuracy(y[te], pk_new[te], cells[te])[0] if te.any() else None
    print(f"[repack] clean: {n_diff} / {n_win} predictions differ, max |dp| {dp_max:.2e}; NEW {1e3 * t_new / max(n_win, 1):.1f} "
          f"ms / window; kit-test cell-BA {test_cellba}", flush=True)
    if r75:                                      # row 75: NEW0 = REF exactly; NEW (tv on) vs REF is descriptive (the views move outputs)
        print(f"[repack] row 75: NEW0 vs REF {n_diff0} / {n_win} differ, max |dp| {dp0:.2e}; NEW (tv) vs REF {n_diff} / {n_win} differ (descriptive)", flush=True)
        assert n_diff0 == 0 and dp0 <= 1e-5, f"row 75: the template change is not inert with tv off ({n_diff0}, {dp0:.2e})"
    assert (r75 or n_diff <= args.allow_differ) and fin_all, \
        f"repacked package differs from the old one on clean data ({n_diff} / {n_win} > --allow-differ {args.allow_differ})"
    if reconfig:                                # T2-D210: each part = its source package's; the new output = their re-weighted mix
        print(f"[repack] reconfig parts max |d|: {parts_dev}; NEW vs REF argmax differ {n_new_vs_old} / {n_win} (descriptive)"
              + (f"; T2-D500 OLD vs NEW argmax differ {set_moved} / {n_win}, max |dp| {set_dp:.3e} (the edit took iff > 0)" if args.set_ref else ""), flush=True)
        if fm_adapt:
            print(f"[repack] FM swap onto a base with in-stream FM adaptation ({[k for k in FM_ADAPT_KEYS if cfg.get(k)]}): p1 / p2 / p3 strict on every "
                  f"window; pf / recombine / heads strict BEFORE the first refit ({fr0}): pre-refit {parts_pre}; all-window values descriptive", flush=True)
        bad_ = reconfig_parts_gate(parts_dev, parts_pre, side_info, fm_adapt)
        assert not bad_, f"reconfig parts check failed: {bad_} (all windows {parts_dev}; pre-refit {parts_pre})"
        if instream_fm:                         # T2-D534: descriptive, how far the donor FM moved the base's in-stream CNN parts
            print(f"[repack] T2-D534: in-stream base + donor FM: CNN parts compared with in-stream OFF (bound above); with in-stream ON they move by "
                  f"max |d| {instream_moved:.3e} (the adapted copies follow the stack's new labels; descriptive)", flush=True)
        if side_info is not None:               # T2-D524 (W18): the members are the donor's (their plain mean == dref's FM part); the heads == the donor's files
            print(f"[repack] T2-D524 FM side heads on {side_info['windows']} windows: members' plain mean vs the head-free donor max |d| "
                  f"{side_info['in_vs_donor']:.2e}; heads vs the independent recomputation max |d| {side_info['head_vs_indep']:.2e}", flush=True)
            assert fm_adapt or (side_info["in_vs_donor"] <= 1e-6 and side_info["head_vs_indep"] <= 1e-6), f"FM side-head check failed: {side_info}"   # fm_adapt: the gate above
        if side_mix is not None and side_mix.get("mode") == "replace":   # T2-D580: the replaced side nets moved the side families' parts
            print(f"[repack] T2-D580 on {side_mix['windows']} windows: the side families' parts moved by max |d| {side_mix['moved']:.3e} (bar > 1e-4); "
                  f"the FBCNet / FM parts are bound by the p1 / pf gates", flush=True)
            assert side_mix["moved"] > 1e-4 and side_mix["windows"] > 0, f"T2-D580: the side replacement did not move the side parts: {side_mix}"
        elif side_mix is not None:              # T2-D549 Gate T
            print(f"[repack] T2-D549 Gate T on {side_mix['windows']} windows: NEW's side-family parts vs (n_old REF + n_add B) / n_new max |d| "
                  f"{side_mix['gate_T']:.2e} (bar 1e-6)", flush=True)
            assert side_mix["gate_T"] <= 1e-6 and side_mix["windows"] > 0, f"T2-D549 Gate T failed: {side_mix}"
        assert not args.set_ref or set_dp > 1e-4, f"T2-D500: --set-ref but OLD and NEW give the same probabilities (max |dp| {set_dp:.2e}): the edit did not take"
        if copies_info.get("parts_vs_donor"):   # P2: the new CNN parts = the donor's up to its pooled nets' training noise (D298: < 1e-4)
            print(f"[repack] copies: CNN parts vs the donor as built, max |d| {copies_info['parts_vs_donor']} (bar 1e-3)", flush=True)
            assert max(copies_info["parts_vs_donor"].values()) <= 1e-3, f"copies parts check failed: {copies_info['parts_vs_donor']}"

    # ---- 3. robustness cases on the kit-test streams
    rng = np.random.default_rng(0)
    cases = {
        "zero": lambda xs: _at(xs, K_BAD, lambda x: 0 * x),
        "nan10": lambda xs: _at(xs, K_BAD, lambda x: np.where(rng.random(x.shape) < 0.1, np.nan, x)),
        "nan_all": lambda xs: _at(xs, K_BAD, lambda x: np.full_like(x, np.nan)),
        "inf1": lambda xs: _at(xs, K_BAD, lambda x: np.where(rng.random(x.shape) < 0.01,
                                                              np.where(rng.random(x.shape) < 0.5, np.inf, -np.inf), x)),
        "x1e4": lambda xs: _at(xs, K_BAD, lambda x: x * 1e4),
        "x1e4_first": lambda xs: _at(xs, 0, lambda x: x * 1e4),
        "flat_ch": lambda xs: _flat(xs, 5),
    }
    rob = {}
    test_streams = [g for g in streams if g["split"].iloc[0] == "test"]
    for name, fn in cases.items():
        r = dict(new_finite=True, new_agree=[], old_agree=[], old_class0_after=[])
        for g in test_streams:
            xs = np.asarray(X[g["i"].to_numpy()], np.float32)
            clean, _ = run_stream(m_new, xs, args.device)
            xb = fn(xs.copy())
            b, fin = run_stream(m_new, xb, args.device)
            a, _ = run_stream(m_old, xb, args.device)
            k0 = 0 if name == "flat_ch" else (1 if name == "x1e4_first" else K_BAD + 1)
            r["new_finite"] &= bool(fin) and set(b.tolist()) <= {0, 1, 2}
            r["new_agree"].append(float((b[k0:] == clean[k0:]).mean()))
            r["old_agree"].append(float((a[k0:] == clean[k0:]).mean()))
            r["old_class0_after"].append(float((a[k0:] == 0).mean()))
        r = {k: (round(float(np.mean(v)), 4) if isinstance(v, list) else v) for k, v in r.items()}
        r["pass"] = bool(r["new_finite"] and (name == "flat_ch" or r["new_agree"] >= 0.9))
        rob[name] = r
        print(f"[repack] case {name:10s}: NEW finite {r['new_finite']}, agrees with clean {r['new_agree']:.3f} | "
              f"OLD agrees {r['old_agree']:.3f} (class 0 after: {r['old_class0_after']:.3f}) -> {'PASS' if r['pass'] else 'FAIL'}",
              flush=True)
    res = dict(old_zip=str(args.zip), old_zip_sha256=hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
               zip=str(zp), zip_sha256=zsha, zip_mb=round(zp.stat().st_size / 1e6, 2), clean_windows=n_win,
               clean_differ=n_diff, clean_max_dp=dp_max, allow_differ=args.allow_differ,
               new_ms_per_window=round(1e3 * t_new / max(n_win, 1), 2), kit_test_cellba=test_cellba, robustness=rob,
               kit_sha=ksha, versions=_versions(),
               reconfig=(dict(fm_from=args.fm_from, set=args.set, parts_max_abs=parts_dev, parts_pre_refit=parts_pre if fm_adapt else None,
                              parts_mode="fm_swap_onto_fm_adapting_base" if fm_adapt else "strict", new_vs_ref_differ=n_new_vs_old,
                              copies=copies_info or None, fm_add=add_info or None, fm_keep=keep_info or None, side_from=side_mix,
                              set_ref=dict(old_vs_new_differ=set_moved, old_vs_new_max_dp=set_dp) if args.set_ref else None) if reconfig else None),
               all_pass=all(r["pass"] for r in rob.values()), seconds=round(time.time() - t0))
    Path(str(out) + ".repack.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "robustness"}, indent=1), flush=True)
    shutil.rmtree(old)
    for d_ in refs.values():
        shutil.rmtree(d_)
    sys.exit(0 if res["all_pass"] else 3)


def _versions():
    import importlib.metadata as md
    out = {}
    for d in ("torch", "braindecode", "numpy", "scipy", "mne"):
        try:
            out[d] = md.version(d)
        except Exception:
            out[d] = None
    return out


def _at(xs, k, f):
    xs[k] = f(xs[k].astype(np.float64)).astype(np.float32)
    return xs


def _flat(xs, c):
    xs[:, c] = 0.0
    return xs


if __name__ == "__main__":
    main()
