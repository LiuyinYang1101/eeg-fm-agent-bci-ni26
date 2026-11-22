#!/usr/bin/env python
"""T2-D595 (W22 candidate, PREPARED as code only): the post-build check of a W22-D595 package (template W22 + cfg p1_prior_reset) against its BASE
(OLD = W21 by default, or the FM chain's final W22 zip when D595 is the LAST step: the manager 2026-11-18), on real public streams, on the path the worker takes. The template's SinkhornCorrector is wrapped (instances tagged by family) to record
every call's input / output and every reset:
  (0) the config: NEW == OLD + p1_prior_reset true, every other key identical; NEW's template = t2p_submission_w22.py;
  (a) finite; (b) NEW == OLD (BASE) on every window before the first refit (<= 1e-12); (c) a re-run reproduces NEW (<= 1e-6);
  (R) NEW's P1 families' (cnn2 / cnn3) correctors reset exactly at the fired refit windows of each stream (none in OLD);
  (P) every family's corrector INPUTS (the parts) == OLD's on every window (<= 1e-5: the reset never touches the parts, the refits or the teacher);
  (G) NEW's cnn2 / cnn3 corrector outputs == Track3's lab decoder sinkhorn_reset of those inputs (lead/diag_word_2026-11-18/d595.py, re-implemented
      in tests/test_t2p_w22_d595.py's golden; here the template's own corrector restarted at the recorded resets, <= 1e-12) and every OTHER
      family's corrector outputs == OLD's (<= 1e-9);
  NEW's argmax on every public window -> .pred.npz (K2).
  python tracks/t2_bci/scripts/t2p_w22_d595_pkgcheck.py --old <unzipped BASE dir> --new <unzipped W22-D595 dir> [--out res.json] [--streams N]
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402

LOG = []


def tag_and_wrap(model):
    """tag each fam_prior corrector with its family and wrap the class's __call__ / reset once (they log to LOG while LOG_ON)."""
    for f, c in (model.fam_prior or {}).items():
        c._fam = f
    cls = type(next(iter(model.fam_prior.values())))
    if not getattr(cls, "_d595_wrapped", False):
        call0, reset0 = cls.__call__, cls.reset

        def call(self, p):
            out = call0(self, p)
            if getattr(self, "_fam", None) is not None:
                LOG.append(("call", self._fam, np.asarray(p, np.float64).copy(), np.asarray(out, np.float64).copy(), len(self.rows)))
            return out

        def reset(self):
            reset0(self)
            if getattr(self, "_fam", None) is not None:
                LOG.append(("reset", self._fam, None, None, None))
        cls.__call__, cls.reset, cls._d595_wrapped = call, reset, True
    return cls


def run(model, xs, dev):
    LOG.clear()
    probs = []
    RP.run_stream(model, xs, dev, probs)
    return np.stack(probs), list(LOG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    ap.add_argument("--streams", type=int, default=0, help="the first N public streams only (a smoke test; 0 = all, and the .pred.npz for K2)")
    a = ap.parse_args()
    os.environ["NI26_PARTS"] = "1"
    RP.kit.use_track("bci_decoding")
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    cfo = json.loads((Path(a.old) / "config.json").read_text())
    rest = {k: v for k, v in cfg.items() if k != "p1_prior_reset"}
    src_new = (Path(a.new) / "submission.py").read_text()
    cfg0 = dict(p1_prior_reset=cfg.get("p1_prior_reset"), old_flag=cfo.get("p1_prior_reset"), rest_equal=rest == cfo,
                template_w22="cfg.get(\"p1_prior_reset\")" in src_new and "T2-D595 (W22 candidate" in src_new, refits=cfo.get("instream", {}).get("refits"))
    print(f"[d595 check] config: {cfg0}", flush=True)
    assert cfg0["p1_prior_reset"] is True and cfg0["old_flag"] is None and cfg0["rest_equal"] and cfg0["template_w22"], cfg0
    refits = sorted(int(c) for c in cfo["instream"]["refits"])
    first = refits[0]
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    tag_and_wrap(old)
    tag_and_wrap(new)
    fams_p1 = list(new.is_fams)
    streams = list(dict.fromkeys(idx.sort_values(["stream", "order"])["stream"].tolist()))
    if a.streams:
        streams = streams[:a.streams]
    pred_new = np.full(len(idx), -1)
    worst = dict(pre=0.0, rerun=0.0, inputs=0.0, other_out=0.0, golden=0.0)
    n_reset_ok, n_streams, finite = 0, 0, True
    for st in streams:
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        po, lo_ = run(old, xs, dev)
        pn, ln_ = run(new, xs, dev)
        pn2, _ = run(new, xs, dev)
        finite &= bool(np.isfinite(pn).all())
        pred_new[rows] = pn.argmax(1)
        k = min(first, len(xs))
        worst["pre"] = max(worst["pre"], float(np.abs(pn[:k] - po[:k]).max()))
        worst["rerun"] = max(worst["rerun"], float(np.abs(pn2 - pn).max()))
        calls_o = {}
        for e in lo_:
            if e[0] == "call":
                calls_o.setdefault(e[1], []).append(e)
        calls_n, resets_n, cnt = {}, {f: [] for f in fams_p1}, {}
        for e in ln_:
            if e[0] == "call":
                calls_n.setdefault(e[1], []).append(e)
                cnt[e[1]] = cnt.get(e[1], 0) + 1
            elif e[0] == "reset" and e[1] in resets_n and cnt.get(e[1], 0) > 0:   # a reset after the stream's first call = a refit reset
                resets_n[e[1]].append(cnt[e[1]])
        want = [c for c in refits if c < len(xs)]
        ok_r = all(resets_n[f] == want for f in fams_p1)
        n_reset_ok += int(ok_r)
        n_streams += 1
        assert ok_r, (st, {f: resets_n[f] for f in fams_p1}, want)
        for f, cn in calls_n.items():
            co = calls_o[f]
            assert len(cn) == len(co), (st, f, len(cn), len(co))
            worst["inputs"] = max(worst["inputs"], max(float(np.abs(x[2] - y[2]).max()) for x, y in zip(cn, co)))
            if f in fams_p1:                               # (G): the template's own corrector restarted at the recorded resets
                ref = copy.deepcopy(old.fam_prior[f])
                ref._fam = None
                ref.reset()
                for i_, e in enumerate(cn):
                    if i_ in want:
                        ref.reset()
                    worst["golden"] = max(worst["golden"], float(np.abs(np.asarray(ref(e[2]), np.float64) - e[3]).max()))
            else:
                worst["other_out"] = max(worst["other_out"], max(float(np.abs(x[3] - y[3]).max()) for x, y in zip(cn, co)))
        print(f"[d595 check] {st}: {len(xs)} windows, resets {resets_n[fams_p1[0]]} (want {want}); pre {worst['pre']:.1e} rerun {worst['rerun']:.1e} "
              f"inputs {worst['inputs']:.1e} other {worst['other_out']:.1e} golden {worst['golden']:.1e}", flush=True)
    res = dict(config=cfg0, streams=n_streams, finite=finite, resets_ok=n_reset_ok, worst=worst,
               gates=dict(pre=worst["pre"] <= 1e-12, rerun=worst["rerun"] <= 1e-6, P=worst["inputs"] <= 1e-5, G=worst["golden"] <= 1e-12,
                          other=worst["other_out"] <= 1e-9, R=n_reset_ok == n_streams, a=finite))
    print(f"[d595 check] {res}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not a.streams:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert all(res["gates"].values()), res["gates"]
    print("[d595 check] PASS: config (BASE + p1_prior_reset only, template W22); finite; == BASE before the first refit; reproducible; the P1 "
          "correctors reset exactly at the fired refits; the parts unchanged; their decode = the restarted corrector; every other family == BASE",
          flush=True)


if __name__ == "__main__":
    main()
