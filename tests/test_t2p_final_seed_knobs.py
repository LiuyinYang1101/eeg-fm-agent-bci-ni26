#!/usr/bin/env python
"""T2-D333: t2p_final --fm-seed-knobs (per-ST-seed vp / lp_epochs), CPU, no export needed:
  1. '' = {} (the earlier build byte for byte); a JSON spec parses to {int seed: knobs}; unknown keys, negative values or seeds
     outside --fm-seeds are refused;
  2. the knob names are fit_eval_fm parameters whose defaults are the off values (vp 0, lp_epochs 0);
  3. t2p_final wires the knobs into the fit_eval_fm call AND the manifest entry, and swaps in the bench's FM bank (recipe r1: vref 0)
     for a vp seed, then restores the CNN bank.
  python tests/test_t2p_final_seed_knobs.py  ->  "OK"
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_final as F  # noqa: E402
import t2p_fm  # noqa: E402


AUGL = '                                   **({"aug": args.fm_aug} if args.fm_aug else {}),    # T2-D557: training-only (the template never reads it)\n'
XL = '                                   **xmeta_,                                            # T2-D583 / D586 / D588: training-only too\n'   # T2-D583 / D586 / D588: the training-only knobs ({} unless set)


def refused(spec, seeds):
    try:
        F.fm_seed_knobs(spec, seeds)
    except (AssertionError, ValueError):
        return True
    return False


def main():
    assert F.fm_seed_knobs("", [0, 1, 2]) == {}
    k = F.fm_seed_knobs('{"3": {"vp": 2}, "4": {"lp_epochs": 5}}', [0, 1, 2, 3, 4])
    assert k == {3: {"vp": 2}, 4: {"lp_epochs": 5}}, k
    assert refused('{"9": {"vp": 2}}', [0, 1]), "a seed outside --fm-seeds must be refused"
    assert refused('{"0": {"mixup": 0.4}}', [0]), "an unknown knob must be refused"
    assert refused('{"0": {"vp": -1}}', [0]), "a negative value must be refused"
    assert refused('{"0": {"vp": 1.5}}', [0]), "a non-integer value must be refused"
    assert F.fm_seed_knobs('{"3": {"align": "ea"}, "4": {"lp_epochs": 10}}', [3, 4]) == {3: {"align": "ea"}, 4: {"lp_epochs": 10}}   # T2-D360
    assert refused('{"0": {"align": "zz"}}', [0]), "an unknown alignment must be refused"
    assert refused('{"0": {"align": 1}}', [0]), "align must be a mode name"
    k = F.fm_seed_knobs('{"0": {"lp_epochs": 10, "lr_head": 3e-3}, "2": {"align": "ea"}}', [0, 1, 2])          # T2-D380
    assert k == {0: {"lp_epochs": 10, "lr_head": 3e-3}, 2: {"align": "ea"}}, k
    assert refused('{"0": {"lr_head": 0}}', [0]) and refused('{"0": {"lr_head": -1e-3}}', [0]), "lr_head must be > 0"
    assert refused('{"0": {"lr_head": "3e-3"}}', [0]), "lr_head must be a number"
    k = F.fm_seed_knobs('{"0": {"epochs": 100}, "3": {"epochs": 100, "lp_epochs": 10}}', [0, 3])                 # T2-D384
    assert k == {0: {"epochs": 100}, 3: {"epochs": 100, "lp_epochs": 10}}, k
    assert refused('{"0": {"epochs": 0}}', [0]) and refused('{"0": {"epochs": 100.5}}', [0]), "epochs must be a positive integer"
    k = F.fm_seed_knobs('{"0": {"ckpt": "/x/ck_p1.pth", "lp_epochs": 10}, "2": {"align": "ea"}}', [0, 2])          # T2-D392
    assert k == {0: {"ckpt": "/x/ck_p1.pth", "lp_epochs": 10}, 2: {"align": "ea"}}, k
    assert refused('{"0": {"ckpt": 3}}', [0]) and refused('{"0": {"ckpt": "/x/notes.txt"}}', [0]), "ckpt must be a .pth / .pt path"
    k = F.fm_seed_knobs('{"0": {"lp_epochs": 10, "lr_head": 3e-3, "tilt": 0.15}, "2": {"align": "ea", "tilt": 0.15}}', [0, 2])   # T2-D400
    assert k == {0: {"lp_epochs": 10, "lr_head": 3e-3, "tilt": 0.15}, 2: {"align": "ea", "tilt": 0.15}}, k
    assert refused('{"0": {"tilt": 0}}', [0]) and refused('{"0": {"tilt": 1.5}}', [0]) and refused('{"0": {"tilt": "0.15"}}', [0]), "tilt in (0, 1)"
    # T2-D392: reset_cpt restores the pinned file after a per-seed redirect, and is a no-op when nothing is redirected
    fm = "st_large"
    ck0, sha0 = t2p_fm.t1.CKPT[fm], t2p_fm.t1.SHA12[fm]
    t2p_fm.reset_cpt(fm)
    assert (t2p_fm.t1.CKPT[fm], t2p_fm.t1.SHA12[fm]) == (ck0, sha0) and fm not in t2p_fm.t1.CPT_INFO
    t2p_fm.t1.CKPT[fm], t2p_fm.t1.SHA12[fm], t2p_fm.t1.CPT_INFO[fm] = Path("/x/ck_p1.pth"), "deadbeef0000", dict(ckpt="/x/ck_p1.pth")
    t2p_fm.reset_cpt(fm)
    assert (t2p_fm.t1.CKPT[fm], t2p_fm.t1.SHA12[fm]) == (ck0, sha0) and fm not in t2p_fm.t1.CPT_INFO, "reset_cpt must restore the pinned file"
    sig = inspect.signature(t2p_fm.fit_eval_fm).parameters
    assert sig["vp"].default == 0 and sig["lp_epochs"].default == 0, "the knobs' off values must be fit_eval_fm's defaults"
    assert sig["tilt"].default == 0.0, "T2-D400: an unset tilt must be off"
    assert sig["lr_head"].default == 1e-3, "T2-D380: an unset lr_head must equal the shipped head lr"
    assert "lr_head" not in inspect.getsource(F.fm_knobs), "T2-D380: fm_knobs must not also pass lr_head (no duplicate kwarg)"
    src = inspect.getsource(F.main)
    assert 'ap.add_argument("--fm-seed-knobs", default=""' in inspect.getsource(F) or '"--fm-seed-knobs", default=""' in src
    assert "mixup=args.fm_mixup, **fm_kw, **kn," in src, "the knobs must reach fit_eval_fm"
    assert "mixup=args.fm_mixup, **fm_kw, **kn, **side_m, **cont_m,\n" + AUGL + XL + "                                   precision=" in src, \
        "the knobs must reach the manifest entry"   # T2-D524 / D546: + side_m / cont_m; T2-D557: + aug (only when set)
    assert 'T.RECIPES["r1"]' in src and "vref=r1.vp_r" in src and "T.VP_BANK.update(saved_bank)" in src, "the FM bank swap"
    assert 'kn = {k_: v_ for k_, v_ in seed_kn.get(s, {}).items() if k_ not in ("align", "epochs", "ckpt", "car")}' in src and "align=dict(mode=al_s" in src, \
        "T2-D360: a per-seed alignment feeds its own aligned input and the member manifest, and never reaches fit_eval_fm"
    assert 'ep_s = int(seed_kn.get(s, {}).get("epochs", args.fm_epochs))' in src, "T2-D384: unset epochs = --fm-epochs"
    assert src.count("epochs=ep_s, drop_path=args.fm_drop_path") == 2 and "epochs=args.fm_epochs, drop_path=args.fm_drop_path, mixup" not in src, \
        "T2-D384: this seed's length reaches fit_eval_fm AND the manifest entry, never both epochs and **kn"
    assert 'ck_s = seed_kn.get(s, {}).get("ckpt", args.fm_ckpt)' in src and "fm_ckpt=ck_s," in src and "t2p_fm.reset_cpt(args.fm)" in src, \
        "T2-D392: this seed's checkpoint reaches fit_eval_fm, after a reset when another seed's was active"
    assert 'k_ not in ("align", "epochs", "ckpt", "car")' in src, "T2-D392 / D475e: ckpt and car never reach fit_eval_fm as **kn knobs"
    # T2-D475e: the car knob (0 / 1), CAR windows as t2p_bench --car (car_windows first, then the scale and the alignment), the member's
    # manifest carries car + the CAR scale, and a car build is refused without --fm-donor before any training
    assert F.fm_seed_knobs('{"0": {"car": 1}, "1": {"car": 0, "lp_epochs": 10}}', [0, 1]) == {0: {"car": 1}, 1: {"car": 0, "lp_epochs": 10}}
    assert refused('{"0": {"car": 2}}', [0]) and refused('{"0": {"car": "1"}}', [0]), "car is 0 or 1"
    assert "XC = stream.car_windows(np.asarray(X))" in src and "B.aligned_all(XC, idx, al_s, trm, 0.0, args.shrink, scale_c)" in src
    assert "scale_c = float(np.median(np.asarray(XC[np.flatnonzero(trm)[::7]]).std(-1)))" in src and \
        "scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))" in src, "the CAR scale is the plain scale's formula on CAR windows"
    assert "align=dict(mode=al_s, scale=scale_c if car_s else scale, shrink=args.shrink)" in src and '**({"car": True} if car_s else {})' in src
    assert src.index("if _car and not args.fm_donor:") < src.index("X, idx, meta = B.load_export(args.export)"), "refuse before loading / training"
    assert "elif agree < 0.995:" in src and "if args.fm_donor:" in src, "the golden check is enforced unless --fm-donor"
    assert "return_state=True, micro=args.fm_micro," in src and '**({"micro": args.fm_micro} if args.fm_micro > 1 else {})' in src, \
        "T2-D475e: --fm-micro reaches fit_eval_fm and the manifest (default 1 = no key)"
    assert inspect.signature(t2p_fm.fit_eval_fm).parameters["micro"].default == 1
    assert "for ck_ in [c for c in ckpts_ if c]:" in inspect.getsource(F), "T2-D392: the kit leak check covers every per-seed checkpoint"
    print("OK")


if __name__ == "__main__":
    main()
