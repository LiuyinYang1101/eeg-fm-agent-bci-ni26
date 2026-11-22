#!/usr/bin/env python
"""T2-D248: the package builder's ST fine-tune knobs (t2p_final --fm-llrd / --fm-rc, fm_knobs), CPU, no export needed:
  1. fm_knobs: both unset = ({}, {}) (the earlier build byte for byte: no kwarg reaches fit_eval_fm, no key reaches the FM manifest entry,
     no `utr` in the FM data); --fm-llrd 0.75 --fm-rc 0.75 = T2-D247's knobs, with `utr` = t2p_bench's training-user index;
  2. the kwarg names are fit_eval_fm parameters whose defaults are the "off" values (llrd 1, rc 0), so a knob set to its off value and an
     unset knob train the same net;
  3. t2p_final: the flags default off and fm_knobs' output reaches the FM data dict, the fit_eval_fm call and the manifest entry;
  4. T2-D546 (b): --fm-init-zip / --fm-init-map default off (seed s, the seed's epochs, no init_state, no cont entry = the shipped path).
  python tests/test_t2p_final_fm_knobs.py  ->  "OK"
"""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_final as F  # noqa: E402
import t2p_fm  # noqa: E402


AUGL = '                                   **({"aug": args.fm_aug} if args.fm_aug else {}),    # T2-D557: training-only (the template never reads it)\n'
XL = '                                   **xmeta_,                                            # T2-D583 / D586 / D588: training-only too\n'   # T2-D583 / D586 / D588: the training-only knobs ({} unless set)


def main():
    rng = np.random.default_rng(0)
    subj = rng.choice(["P01", "P02", "P03", "P06", "P07", "P10"], 400)
    assert F.fm_knobs(1.0, 0.0, subj) == ({}, {}), "unset knobs must add nothing"
    kw, u = F.fm_knobs(0.75, 0.75, subj)
    assert kw == {"llrd": 0.75, "rc": 0.75} and set(u) == {"utr"} and np.array_equal(u["utr"], F.user_index(subj))
    kw1, u1 = F.fm_knobs(0.75, 0.0, subj)
    assert kw1 == {"llrd": 0.75} and u1 == {}
    print("fm_knobs: unset = ({}, {}); T2-D247's knobs with the training-user index; llrd alone needs no utr")
    sig = inspect.signature(t2p_fm.fit_eval_fm).parameters
    assert sig["llrd"].default == 1.0 and sig["rc"].default == 0.0, (sig["llrd"].default, sig["rc"].default)
    print("fit_eval_fm: llrd 1 / rc 0 by default (= unset)")
    src = (ROOT / "tracks/t2_bci/scripts/t2p_final.py").read_text()
    assert 'ap.add_argument("--fm-llrd", type=float, default=1.0' in src and 'ap.add_argument("--fm-rc", type=float, default=0.0' in src
    assert "fm_kw, fm_u = fm_knobs(args.fm_llrd, args.fm_rc, idx[\"subject\"].to_numpy()[tr])" in src
    assert "cte=cells[ev], **fm_u, **side_kw, **wub)" in src    # T2-D524: + the FM side vectors ({} unless --fm-side); T2-D642: + wub ({} unless --user-balance)
    assert "mixup=args.fm_mixup, **fm_kw, **kn, **init_kw, **fm_aug_kw, **xk_,\n                                   calib=args.fm_calib" in src, "fit_eval_fm call"
    assert '**({"grl": args.fm_grl, "grl_lr": args.fm_grl_lr} if args.fm_grl is not None else {}),' in src and '"--fm-grl", type=float, default=None' in src, "T2-D649: --fm-grl unset = no grl kwargs (the old call)"   # T2-D557: + fm_aug_kw ({} by default)
    assert 'ap.add_argument("--fm-aug", default=""' in src and "fm_aug_kw = {}\n        if args.fm_aug:" in src, "T2-D557: --fm-aug defaults off"   # T2-D333: + the per-seed knobs ({} by default); T2-D546: + init_kw ({} by default)
    assert "mixup=args.fm_mixup, **fm_kw, **kn, **side_m, **cont_m,\n" + AUGL + XL + "                                   precision=" in src, "manifest entry"   # T2-D557: + aug (only when set)   # T2-D524: + side ({} unless --fm-side); T2-D546: + cont ({} by default)
    # T2-D546 (b): --fm-init-zip off by default = the shipped build path unchanged (seed s, epochs ep_s, no init_kw, no cont entry)
    assert 'ap.add_argument("--fm-init-zip", default=""' in src and 'ap.add_argument("--fm-init-map", default=""' in src, "T2-D546: off by default"
    assert "init_kw, s_fit, ep_fit, cont_m = {}, s, ep_s, {}\n            if args.fm_init_zip:" in src, "T2-D546: unset = seed s, ep_s, {} / {}"
    assert 'r = t2p_fm.fit_eval_fm(args.fm, data, s_fit, meta["ch_names"], epochs=ep_fit, drop_path=args.fm_drop_path,' in src, "T2-D546: the fit call"
    assert src.count("s_fit, ep_fit = s + 50000, args.fm_cont_epochs") == 1 and src.count("init_kw = dict(init_state=st0") == 1, "T2-D546: set only under --fm-init-zip"
    assert "if args.fm_init_zip and not args.fm_donor:" in src, "T2-D546: --fm-init-zip is donor-only"
    sig_ = inspect.signature(t2p_fm.fit_eval_fm).parameters
    assert sig_["init_state"].default is None and sig_["warm"].default == 0.1 and callable(t2p_fm.load_member_state), "T2-D546: fit_eval_fm defaults"
    assert 'ap.add_argument("--fm-side", default="", choices=["", "dcmean"]' in src and "side_kw, side_blob = {}, None" in src \
        and "fm_side_files = {}" in src and "**({\"side\": args.fm_side} if args.fm_side else {})" in src, "T2-D524: --fm-side off by default"
    assert "if args.fm_side and not args.fm_donor:" in src, "T2-D524: --fm-side is donor-only"
    print("t2p_final: flags off by default; the knobs reach the FM data, the fit and the manifest entry")
    print("OK")


if __name__ == "__main__":
    main()
