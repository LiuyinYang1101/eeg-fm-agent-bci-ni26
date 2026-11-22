#!/usr/bin/env python
"""T2 FAST LAB (a registered TOOL, not a decision row; the owner's "fast iterations", the manager's spec 2026-11-17): screen head / refit /
decoder / calibration / feature-space ideas on cached fresh-draw stacks in seconds, then confirm a PRE-DECLARED top few on a second draw set.

ENGINE = the registered stack reader (scripts/local/t2_d540a_stack_read.py, its head exec'd unchanged: files, W19's parts, the causal Sinkhorn
decoder, the cell-mean scorer and D540a's guards), with:
  * every file read memoized in memory (members' dumps, side nets, CNN parts, teacher posteriors): the first idea pays the I/O, later ones not;
  * the stack MIX re-implemented as the reader's mix() and ASSERTED equal to read(W19) at start (max |dp| == 0);
  * every decoded family part memoized by a content hash: an idea that changes only the FM part re-decodes only the FM family.
Draw sets: A = 825-827 (screen), B = 828-830 (confirm); both use D555's re-run FM members (pre-head logits, side heads + z, 1024-d pooled
features + main heads), so FM head / feature ideas are expressible on both. W19 = BASE + D539b + D543 + D546b + D553.
An IDEA (scripts/local/t2_lab_ideas.py) = a name + params + a function that may (a) give each FM member's probabilities from its dumps,
(b) transform the four family parts before decoding, (c) change the decoder's (iters, n0, lam) per family, (d) change the family weights.
SCREEN: every idea on set A, logged (outputs/t2_lab/screen_<set>.jsonl: idea, params, ten, draws, per stranger, returning, sealed mix,
paired SE over the ten strangers, D540a guards). CONFIRM: the TOP_K = 3 (pre-declared) screen passes by ten-stranger delta go to set B, each
read SINGLY vs W19 under D540a's stack bar (ten >= +0.001, >= 2 / 3 draws > 0, no stranger < -0.020, returning >= -0.003, sealed mix > 0);
logged in confirm_<set>.jsonl. The number of screened configs per confirm is printed, so the forking is visible.
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_lab.py screen --set A --batch batch1
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_lab.py confirm --set B --from screen_A.jsonl
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
MODE = sys.argv[1] if len(sys.argv) > 1 else "screen"
ARGS = dict(zip(sys.argv[2::2], sys.argv[3::2]))
SET = ARGS.get("--set", "A")
DSET = {"A0": 822, "A": 825, "B": 828, "C": 831, "D": 834, "E": 860}[SET]   # T2-D615: E           # A0 = development / non-feature screens (D553's members: no pooled features)
TOP_K = 3
ITEMS = {822: "D539b,D543,D546b,D553", 825: "D539b,D543,D546b,D553,D555", 828: "D539b,D543,D546b,D553,D557",
         831: "D539b,D543,D546b,D553,D555", 834: "D539b,D543,D546b,D553,D557", 860: "D539b,D543,D546b,D553,D557"}[DSET]          # C (T2-D560's third read): D558's new-seed members (837-842, fresh fits)
import os as _os0
LAB_D557 = _os0.environ.get("NI26_LAB_D557") == "1"   # T2-D557's interaction read (C only): the headed members = their augmented re-runs
assert not LAB_D557 or DSET == 831, "NI26_LAB_D557: set C only (the augmented members with pooled features)"
if LAB_D557:
    ITEMS = "D539b,D543,D546b,D553,D557"
LAB_D567 = _os0.environ.get("NI26_LAB_D567") == "1"   # T2-D567 (registered 2026-11-18 06:40): slot 0 = the augmented MI line (B: on W20; C: on W20+D557)
assert not LAB_D567 or DSET == 828 or (DSET == 831 and LAB_D557), "NI26_LAB_D567: set B, or set C with NI26_LAB_D557=1 (the registered reads)"
if LAB_D567:
    ITEMS += ",D567"
LAB_D568 = _os0.environ.get("NI26_LAB_D568") == "1"   # T2-D568 (registered 2026-11-18 11:34:11): the five headed members at the stronger dose (C on W20+D557)
assert not LAB_D568 or (DSET == 831 and LAB_D557), "NI26_LAB_D568: set C with NI26_LAB_D557=1 only (the registered read)"
if LAB_D568:
    ITEMS += ",D568"
LAB_D582 = _os0.environ.get("NI26_LAB_D582") == "1"   # T2-D582: the five headed members + vp 2 (C on W20+D557, or D)
assert not LAB_D582 or (DSET == 831 and LAB_D557) or DSET == 834, "NI26_LAB_D582: set C with NI26_LAB_D557=1, or set D"
if LAB_D582:
    ITEMS += ",D582"
LAB_D583 = _os0.environ.get("NI26_LAB_D583") == "1"   # T2-D583 C: the five headed members + segment recombination (C on W20+D557 only)
assert not LAB_D583 or (((DSET == 831 and LAB_D557) or DSET == 834) and not LAB_D582 and not LAB_D568), "NI26_LAB_D583: set C with NI26_LAB_D557=1, or set D; not with D568 / D582"
if LAB_D583:
    ITEMS += ",D583"
LAB_D586 = _os0.environ.get("NI26_LAB_D586") == "1"   # T2-D586 C: the plain SD member (837) -> its shared-phase surrogate twin (C on W20+D557 only)
assert not LAB_D586 or (((DSET == 831 and LAB_D557) or DSET == 834) and not (LAB_D582 or LAB_D568 or LAB_D583)), "NI26_LAB_D586: set C with NI26_LAB_D557=1, or set D; alone"
if LAB_D586:
    ITEMS += ",D586"
LAB_D588 = _os0.environ.get("NI26_LAB_D588") == "1"   # T2-D588: the five headed members + two-view consistency (C on W20+D557, or D)
assert not LAB_D588 or (((DSET == 831 and LAB_D557) or DSET == 834) and not (LAB_D582 or LAB_D568 or LAB_D583 or LAB_D586)), "NI26_LAB_D588: set C with NI26_LAB_D557=1, or set D; alone"
if LAB_D588:
    ITEMS += ",D588"
LAB_D588B = _os0.environ.get("NI26_LAB_D588B", "")      # T2-D588 B: 'arm' only (the REF = D582's bref via NI26_LAB_D582B=ref on the REF side)
assert LAB_D588B in ("", "arm") and (not LAB_D588B or DSET == 828), "NI26_LAB_D588B: arm, set B only"
if LAB_D588B:
    ITEMS += ",D588B"
LAB_D596 = _os0.environ.get("NI26_LAB_D596") == "1"   # T2-D596: the five headed members + the WORD-margin loss (C on W20+D557, or D)
assert not LAB_D596 or (((DSET == 831 and LAB_D557) or DSET == 834) and not (LAB_D582 or LAB_D568 or LAB_D583 or LAB_D586 or LAB_D588)), "NI26_LAB_D596: set C with NI26_LAB_D557=1, or set D; alone"
if LAB_D596:
    ITEMS += ",D596"
LAB_D596B = _os0.environ.get("NI26_LAB_D596B", "")      # T2-D596 B: 'arm' only (the REF = D582's bref via NI26_LAB_D582B=ref on the REF side)
assert LAB_D596B in ("", "arm") and (not LAB_D596B or DSET == 828), "NI26_LAB_D596B: arm, set B only"
if LAB_D596B:
    ITEMS += ",D596B"
LAB_D583B = _os0.environ.get("NI26_LAB_D583B", "")      # T2-D583 B: 'arm' only (the REF = D582's bref via NI26_LAB_D582B=ref on the REF side)
assert LAB_D583B in ("", "arm") and (not LAB_D583B or DSET == 828), "NI26_LAB_D583B: arm, set B only"
if LAB_D583B:
    ITEMS += ",D583B"
LAB_D611 = _os0.environ.get("NI26_LAB_D611") == "1"   # T2-D611 C / D: the CAR SD slot 831 -> its 8-30 Hz band-pass specialist (C on W20+D557 only)
assert not LAB_D611 or (((DSET == 831 and LAB_D557) or DSET == 834) and not (LAB_D582 or LAB_D568 or LAB_D583 or LAB_D586 or LAB_D588 or LAB_D596)), "NI26_LAB_D611: set C with NI26_LAB_D557=1, or set D; alone"
if LAB_D611:
    ITEMS += ",D611"
LAB_D611B = _os0.environ.get("NI26_LAB_D611B", "")      # T2-D611 B: 'arm' = D582's REF (NI26_LAB_D582B=ref on the ARM side too) with slot 831 swapped
assert LAB_D611B in ("", "arm") and (not LAB_D611B or (DSET == 828 and _os0.environ.get("NI26_LAB_D582B") == "ref")), "NI26_LAB_D611B: arm, set B, with NI26_LAB_D582B=ref"
if LAB_D611B:
    ITEMS += ",D611B"
LAB_D618 = _os0.environ.get("NI26_LAB_D618") == "1"   # T2-D618 C / D: the CAR SD slot 831 -> its left-right mirror twin (C on W20+D557 only)
assert not LAB_D618 or (((DSET == 831 and LAB_D557) or DSET == 834) and not (LAB_D582 or LAB_D568 or LAB_D583 or LAB_D586 or LAB_D588 or LAB_D596 or LAB_D611)), "NI26_LAB_D618: set C with NI26_LAB_D557=1, or set D; alone"
if LAB_D618:
    ITEMS += ",D618"
LAB_D618B = _os0.environ.get("NI26_LAB_D618B", "")      # T2-D618 B: 'arm' = D582's REF (NI26_LAB_D582B=ref on the ARM side too) with slot 831 swapped
assert LAB_D618B in ("", "arm") and (not LAB_D618B or (DSET == 828 and _os0.environ.get("NI26_LAB_D582B") == "ref" and not LAB_D611B)), "NI26_LAB_D618B: arm, set B, with NI26_LAB_D582B=ref"
if LAB_D618B:
    ITEMS += ",D618B"
LAB_D586B = _os0.environ.get("NI26_LAB_D586B", "")      # T2-D586 B: 'arm' = D582's REF (NI26_LAB_D582B=ref on the ARM side too) with slot 827 swapped
assert LAB_D586B in ("", "arm") and (not LAB_D586B or (DSET == 828 and _os0.environ.get("NI26_LAB_D582B") == "ref")), "NI26_LAB_D586B: arm, set B, with NI26_LAB_D582B=ref"
if LAB_D586B:
    ITEMS += ",D586B"
LAB_D582B = _os0.environ.get("NI26_LAB_D582B", "")      # T2-D582 B: ref | arm, set B only
assert LAB_D582B in ("", "ref", "arm") and (not LAB_D582B or DSET == 828), "NI26_LAB_D582B: ref | arm, set B only"
if LAB_D582B:
    ITEMS += ",D582B"
LAB_D568B = _os0.environ.get("NI26_LAB_D568B", "")      # T2-D568's B stage (amended 11:39): ref | arm, set B only (on W20 with the dumped members)
assert LAB_D568B in ("", "ref", "arm") and (not LAB_D568B or DSET == 828), "NI26_LAB_D568B: ref | arm, set B only"
if LAB_D568B:
    ITEMS += ",D568B"
# T2-D569 amendment F1 (Track3's pre-number review, registered before the A number): a D569 P1 root is read ONLY as a mix under w20_fb, so the
# LP labels never reach the D553 / D560 FM refits through ctx.teacher (fm_feedback rebuilds the FM part from the R_* / vec_* rows only)
assert not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d569_") or (MODE == "mix" and ARGS.get("--idea") == "w20_fb"), \
    "T2-D569 F1: a t2_d569_ root is read only with mix --idea w20_fb"
assert not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d571_") or (MODE == "mix" and ARGS.get("--idea") in ("w20_fb", "w20_st", "w20_fb_zlp", "w20_fb_fmlp", "w20_fb_softproto", "w20_fb_wref", "w20_fb_mapproto", "w20_fb_fm45")), \
    "T2-D571 / D572 / D579 / D581: a t2_d571_ root is read only as a mix with a registered feedback-teacher idea (the kind <-> idea map below)"
assert not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d580_") or (MODE == "mix" and ARGS.get("--idea") == "w20_fb"), \
    "T2-D580: a t2_d580_ root is read only with mix --idea w20_fb (W21's form; only the side nets swapped)"
assert not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d576_") or (MODE == "mix" and ARGS.get("--idea") == "w20_fb"), \
    "T2-D576: a t2_d576_ root is read only with mix --idea w20_fb (W20+D557's real-teacher form; the FBCNet part swapped)"
_R571 = _os0.environ.get("NI26_D559_ROOT", "")             # T2-D571 F6 (Track3): the root's teacher kind <-> the idea (fb / lp -> w20_fb, un -> w20_st)
_FMX = __import__("re").search(r"fm(4[05]|55)", _os0.environ.get("NI26_D559_ROOT", ""))   # T2-D591 fm45 / T2-D655 fm40 / T2-D658 fm55 roots
assert MODE != "mix" or (ARGS.get("--idea") in ("w20_fb_fm45", "w20_fb_fm40", "w20_fb_fm55")) == bool(_FMX) and (not _FMX or ARGS.get("--idea") == f"w20_fb_fm{_FMX.group(1)}"), \
    "T2-D591 / D655: an fm45 / fm40 root is read only with --idea w20_fb_fm45 / w20_fb_fm40, and that idea only on its own root (the teacher and the mix move together)"
assert MODE != "mix" or not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d594b_") or ARGS.get("--idea") == "w20_fb", \
    "T2-D594b: a t2_d594b_ root is read only with mix --idea w20_fb (W21's form; only the P1 students swapped)"
assert MODE != "mix" or not _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d589_") or ARGS.get("--idea") == ("w20_fb_fm45" if "fm45" in _os0.environ.get("NI26_D559_ROOT", "") else "w20_fb"), \
    "T2-D589: a t2_d589_ root is read only with mix --idea w20_fb (W21's form; only the P1 students swapped)"
assert (_os0.environ.get("NI26_LAB_D584") == "1") == (ARGS.get("--idea") == "w20_fb_wref"), \
    "T2-D584: NI26_LAB_D584=1 (D539b's weighted side heads) and --idea w20_fb_wref (the FM part's weighted refits) go together: one arm"
_OK571 = {"fb": ("w20_fb",), "lp": ("w20_fb", "w20_fb_zlp", "w20_fb_fmlp", "w20_fb_softproto", "w20_fb_wref", "w20_fb_mapproto", "w20_fb_fm45", "w20_fb_fm40", "w20_fb_fm55"), "un": ("w20_st",)}   # T2-D579 / D581: their arms on W21's lp roots
assert not _R571.startswith("t2_d571_") or ARGS.get("--idea") in _OK571[_R571[len("t2_d571_"):][:2]], (_R571, ARGS.get("--idea"))
assert not _R571.startswith("t2_d638_") or (MODE == "fbteach" and not _R571.startswith(("t2_d638_fb", "t2_d638_lp"))) or (MODE == "mix" and _R571.startswith("t2_d638_lp") and ARGS.get("--idea") == "w20_fb"), \
    ("T2-D638: a base root t2_d638_<cell>_860 feeds fbteach only; an lp root t2_d638_lp<cell>_860 is read only as mix --idea w20_fb", _R571, MODE)
assert not _R571.startswith("t2_d642_") or (MODE == "mix" and _R571.startswith("t2_d642_lp") and ARGS.get("--idea") == ("w20_fb_fm40" if "fm40" in _R571 else "w20_fb_fm55" if "fm55" in _R571 else "w20_fb")), \
    ("T2-D642: an lp root t2_d642_lp<arm><root>_860 is read only as mix --idea w20_fb (its fbteach runs on the D638 cell's base root)", _R571, MODE)
_ST = _os0.environ.get("NI26_LAB_STARTS", "")             # T2-D570: the SHIPPED decoder starts ('shipped', or 'shipped_but_<fam>' = that family recomputed)
STARTS = None
if _ST:
    import json as _json0
    _SJ = _json0.load(open("scripts/local/t2_shipped_starts_w20d557.json"))
    _keep = {"fbc", "sh", "ifn", "fm", "dc"}
    if _ST != "shipped":
        assert _ST.startswith("shipped_but_") and _ST[len("shipped_but_"):] in _keep, _ST
        _keep -= {_ST[len("shipped_but_"):]}
    STARTS = {k_: [float(x) / sum(_SJ[k_]) for x in _SJ[k_]] for k_ in sorted(_keep)}
sys.argv = [sys.argv[0], "--draws", str(DSET), "--items", ITEMS]
SRC = open("scripts/local/t2_d540a_stack_read.py").read()
CUT = "\nBASE = read([])\n"
assert SRC.count(CUT) == 1, "the stack reader's tail moved"
exec(compile(SRC.split(CUT)[0], f"t2_d540a_stack_read_head(lab,{DSET})", "exec"))
sys.path.insert(0, "scripts/local")
import t2_lab_ideas as IDEAS  # noqa: E402

LAB = O / "t2_lab"
LAB.mkdir(exist_ok=True)
ON = W19C + ["D553"] + (["D557"] if LAB_D557 else []) + (["D567"] if LAB_D567 else []) + (["D568"] if LAB_D568 else []) + (["D568B"] if LAB_D568B else []) + (["D582"] if LAB_D582 else []) + (["D582B"] if LAB_D582B else []) + (["D583"] if LAB_D583 else []) + (["D586"] if LAB_D586 else []) + (["D583B"] if LAB_D583B else []) + (["D586B"] if LAB_D586B else []) + (["D588"] if LAB_D588 else []) + (["D588B"] if LAB_D588B else []) + (["D596"] if LAB_D596 else []) + (["D596B"] if LAB_D596B else []) + (["D611"] if LAB_D611 else []) + (["D611B"] if LAB_D611B else []) + (["D618"] if LAB_D618 else []) + (["D618B"] if LAB_D618B else [])
# ---------------- memoized I/O ----------------
_load_raw, _sp_raw = load, side_prob
_MEMO = {}


def load(path):
    k = ("load", str(path))
    if k not in _MEMO:
        _MEMO[k] = _load_raw(path)
    return _MEMO[k]


def side_prob(path, f, mode):
    k = ("side", str(path), f, mode, H_ON)
    if k not in _MEMO:
        _MEMO[k] = _sp_raw(path, f, mode)
    return _MEMO[k]


def dumps(path):
    """one FM member's dumps on IX[f]'s order: L (pre-side logits from the main head on the features), z, side head, features, main head."""
    k = ("dump", str(path))
    if k not in _MEMO:
        z = np.load(need(path))
        refuse_oracle(path, z)
        f = int(__import__("re").search(r"/lc9_(\d)__", str(path)).group(1))
        te = z["test_idx"].astype(np.int64)
        pos = {int(r): i for i, r in enumerate(te)}
        sel = np.array([pos[int(r)] for r in IX[f]])
        d = {"f": f, "rows": IX[f]}
        if "fm_feat_te" in z.files:
            Fm, Wm, bm = (z[k_].astype(np.float64) for k_ in ("fm_feat_te", "fm_out_w", "fm_out_b"))
            d.update(feat=Fm[sel], Wm=Wm, bm=bm, L=(Fm @ Wm.T + bm)[sel])
        elif "side_logit_last" in z.files:                 # D553's members: pre-side logits only (no features)
            d.update(feat=None, L=z["side_logit_last"].astype(np.float64)[sel])
        else:                                            # a member without dumps (set A0's MI s832): fixed probabilities
            p_ = z["prob_last"].astype(np.float64)[sel]
            d.update(feat=None, L=None, p=p_ / p_.sum(1, keepdims=True))
        if "side_w" in z.files:
            d.update(z=z["side_te"].astype(np.float64)[sel], Ws=z["side_w"].astype(np.float64), bs=z["side_b"].astype(np.float64))
        _MEMO[k] = d
    return _MEMO[k]


# ---------------- the lab's mix = the reader's mix (asserted below) ----------------
def hsh(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).hexdigest()


_DEC = {}
D606 = _os0.environ.get("NI26_LAB_D606", "") == "1"   # T2-D606 (registered 9daab875): past-only OUTPUT calibration, the mix branch only
assert _os0.environ.get("NI26_LAB_D606", "") in ("", "0", "1"), _os0.environ.get("NI26_LAB_D606")
_D606_ON = False                                        # switched on in the mix branch, after every startup identity check
D607 = _os0.environ.get("NI26_LAB_D607", "") == "1"   # T2-D607 (registered 0e7feec8): FM output history re-scored, the mix branch only
assert _os0.environ.get("NI26_LAB_D607", "") in ("", "0", "1"), _os0.environ.get("NI26_LAB_D607")


def sk_mult(M, iters, lam):
    """T2-D606: the cumulative column multiplier of the corrector's Sinkhorn on M (rows oldest first; = causal_sinkhorn's iteration, the
    row normalisations being per-row scalars): every row of the iterated matrix == normalize(M_i * mult)."""
    K, n = M.shape[1], M.shape[0]
    c = np.ones(K)
    if n == 0:
        return c
    w = None if lam == 1.0 else float(lam) ** np.arange(n - 1, -1, -1, dtype=np.float64)
    for _ in range(int(iters)):
        s = (n / K) / M.sum(0) if w is None else (w.sum() / K) / (M * w[:, None]).sum(0)
        M = M * s
        c = c * s
        M = M / M.sum(1, keepdims=True)
    return c


def causal_sinkhorn_past(prob, order_keys, iters=10, n0=3, lam=1.0, start=None):
    """T2-D606: = t2p_report.causal_sinkhorn but row t = normalize(p_t * b_{t-1}), b_{t-1} = sk_mult of the PRECEDING rows [n0 start rows;
    p_1 .. p_{t-1}] (the current row never estimates its own correction)."""
    out = np.empty_like(prob, dtype=np.float64)
    K = prob.shape[1]
    st0 = np.full(K, 1.0 / K) if start is None else np.asarray(start, np.float64)
    streams, order = order_keys
    for s_ in np.unique(streams):
        rows = np.flatnonzero(streams == s_)
        rows = rows[np.argsort(order[rows], kind="stable")]
        P = prob[rows].astype(np.float64)
        for t in range(len(rows)):
            b = sk_mult(np.vstack([np.tile(st0, (int(n0), 1)), P[:t]]), iters, lam)
            q = P[t] * b
            out[rows[t]] = q / q.sum()
    return out


def decode(p, key, start, args):
    if _LAMX[0] is not None and _FMST_FINAL[0]:          # T2-D656: NI26_LAB_LAM = the lambda of every decode of the FINAL saved mix
        args = (args[0], args[1], _LAMX[0])
        _LAMN[0] += 1
    k = (hsh(p), hsh(np.asarray(start)), args, _D606_ON)
    if k not in _DEC:
        _DEC[k] = (causal_sinkhorn_past if _D606_ON else causal_sinkhorn)(p, key, args[0], args[1], args[2], start=start)
    return _DEC[k]


_LAMX = [None]                                           # T2-D656: NI26_LAB_LAM (unset = None = every decode keeps its own lambda, bitwise)
_LAMN = [0]                                              # T2-D656: decodes run with the override (asserted > 0 when NI26_LAB_LAM)
_ORDN = [0]                                              # T2-D638: folds decoded with the reordered key (the final mix); asserted > 0 when NI26_LAB_ORDER
_FMST_FINAL = [False]                                    # T2-D642 / D638: True only around the mix branch's final lab_mix (the saved one): the FM-start hook + NI26_LAB_ORDER act there only
assert not _os0.environ.get("NI26_LAB_FMSTART") or (MODE in ("mix", "fbteach") and _os0.environ.get("NI26_LAB_FMSTART").split(":", 1)[0] in ("dump", "freeze")), \
    "T2-D642: NI26_LAB_FMSTART only in mix / fbteach, dump:<npz> or freeze:<npz>"
LAST_VEC = {}                                            # T2-D570: the recomputed (cross-fitted) starts of the last lab_mix, for the printout
if _os0.environ.get("NI26_LAB_LAM"):                       # T2-D656 (registered 544853f8): the lambda override
    assert MODE == "mix" and _os0.environ.get("NI26_D559_ROOT", "").startswith("t2_d642_lp"), "T2-D656: NI26_LAB_LAM only on a D642 lp root's mix"
    assert _os0.environ.get("NI26_LAB_D607", "") != "1", "T2-D656: not with D607 (its post hook decodes outside decode())"
    _LAMX[0] = float(_os0.environ["NI26_LAB_LAM"])
    assert 0.5 <= _LAMX[0] <= 1.0, _LAMX[0]
    print(f"[lab] T2-D656 decode lambda override {_LAMX[0]} on the FINAL saved mix (PARTIAL: refit labels keep the shipped lambda)", flush=True)
_ORDER = _os0.environ.get("NI26_LAB_ORDER", "")         # T2-D638: a DECODE-ONLY reorder '<runs_perm|in_run>:<seed>' of lab_mix's final key
if _ORDER:                                               # (members, students, teacher and refits stay in recording order: PARTIAL order evidence)
    assert MODE == "mix" and _os0.environ.get("NI26_D559_ROOT", "").startswith(("t2_d638_lp", "t2_d642_lp")), "T2-D638 / D642: NI26_LAB_ORDER only on a D638 / D642 lp root's mix"
    _ok_, _osd_ = _ORDER.split(":")
    assert _ok_ in ("runs_perm", "in_run", "in_sess"), _ORDER        # T2-D656: + in_sess (= D448's whole-session shuffle)
    _rng_, _RUN_ = np.random.default_rng(int(_osd_)), pd.read_csv(EXP / "index.csv")["run"].to_numpy().astype(int)
    ORDK = np.empty(len(ORD))
    for _s_ in np.unique(STREAM):                        # = scripts/local/t2_d448_orderfit.keys over the whole export (as t2_d631 / d632)
        _m_ = np.flatnonzero(STREAM == _s_)
        if _ok_ == "in_sess":                           # T2-D656: every window of the stream (session) shuffled, runs ignored
            ORDK[_m_] = _rng_.permutation(len(_m_))
            continue
        _runs_ = np.unique(_RUN_[_m_])
        _pos_ = dict(zip(_runs_, _rng_.permutation(len(_runs_)) if _ok_ == "runs_perm" else np.arange(len(_runs_))))
        for _r_ in _runs_:
            _mr_ = _m_[_RUN_[_m_] == _r_]
            _inner_ = _rng_.permutation(len(_mr_)) if _ok_ == "in_run" else np.argsort(np.argsort(ORD[_mr_]))
            ORDK[_mr_] = _pos_[_r_] * 1000 + _inner_
    print(f"[lab] T2-D638 decode-only reorder {_ORDER}: lab_mix's final key permuted (PARTIAL)", flush=True)
else:
    ORDK = ORD


def lab_mix(P, Wt, dec=None, dc_args=(10, 3, 0.95), post=None):
    """P[(f, d)] = {family: raw part on IX[f]}; Wt = family weights; dec = {family: (iters, n0, lam)} (default the shipped 10 / 3 / 0.95)."""
    dec = dec or {}
    vec = {}
    for d in DRAWS:
        for fam in Wt:
            per = {f: P[(f, d)][fam][U[IX[f]] == TRU[f]].mean(0) for f in range(10)}
            for f in range(10):
                v = np.mean([per[g] for g in range(10) if g != f], 0)
                vec[(f, d, fam)] = v / v.sum()
    for k_ in [k_ for k_ in vec if k_[2] == "fm" and _FMST_FINAL[0]]:   # T2-D642: the FINAL saved mix's FM start, recorded / frozen per NI26_LAB_FMSTART
        vec[k_] = IDEAS.fmstart_hook("mix_final", k_[0], k_[1], vec[k_])
    LAST_VEC.clear()
    LAST_VEC.update(vec)
    if STARTS:                                           # T2-D570: the shipped constants (one vector per family, every fold and draw)
        assert _ST != "shipped" or (set(Wt) <= set(STARTS) and "dc" in STARTS), (sorted(Wt), sorted(STARTS))   # F2 (iii): no silent fallback
        for k_ in list(vec):
            if k_[2] in STARTS:
                vec[k_] = np.asarray(STARTS[k_[2]], np.float64)
    out = {}
    for f in range(10):
        ix = IX[f]
        _ORDN[0] += int(bool(_ORDER) and _FMST_FINAL[0])
        key = (STREAM[ix], ORDK[ix] if _FMST_FINAL[0] else ORD[ix])   # T2-D638: the decode-only reorder on the FINAL saved mix only (the startup
                                                 # identity check and REF stay chronological; ORDK is ORD itself unless NI26_LAB_ORDER)
        te, p = DC[f]
        pos = {int(r): k for k, r in enumerate(te)}
        q_ = p[np.array([pos[int(r)] for r in ix])]
        dcd = decode(q_ / q_.sum(1, keepdims=True), key, np.asarray(STARTS["dc"], np.float64) if STARTS and "dc" in STARTS else DCVEC[f], dc_args)
        for d in DRAWS:
            if post is None:
                mx = sum(Wt[fam] * decode(P[(f, d)][fam], key, vec[(f, d, fam)], dec.get(fam, (10, 3, 0.95))) for fam in Wt)
                out[(f, d)] = (ix, (1 - W_DC) * mx + W_DC * dcd)
            else:                                        # an idea's post-decode hook: (f, d, key, {family: decoded}, Wt, dc decoded) -> final
                D_ = {fam: decode(P[(f, d)][fam], key, vec[(f, d, fam)], dec.get(fam, (10, 3, 0.95))) for fam in Wt}
                out[(f, d)] = (ix, post(f, d, key, D_, Wt, dcd))
    return out


configure(ON)
W_ON, FM_FORMS = dict(W), list(fm_forms(W18))
P_ON = {(f, d): parts_of(f, d, W18) for f in range(10) for d in DRAWS}
configure([])
REF = read(ON)
_STARTS_RUN, STARTS = STARTS, None                      # T2-D570: the identity check runs on the lab convention (the reader recomputes its starts)
LAB_REF = lab_mix(P_ON, W_ON)
dmax = max(float(np.abs(LAB_REF[k][1] - REF[k][1]).max()) for k in REF)
assert dmax == 0.0, f"the lab's mix != the reader's read(W19): {dmax}"
STARTS = _STARTS_RUN
FM_PATHS = {f: [fm_member(s, f, form) for s, form in FM_FORMS] for f in range(10)}
if MODE == "fmpaths":                                     # T2-D642: the FM member files the read loads (per fold, in slot order) + the forms -> json
    _os0.makedirs(_os0.path.dirname(ARGS["--out"]), exist_ok=True)
    json.dump(dict(forms=[list(x) for x in FM_FORMS], paths={str(f): [str(x) for x in FM_PATHS[f]] for f in range(10)}, fmsplice=_os0.environ.get("NI26_LAB_FMSPLICE", "")),
              open(ARGS["--out"], "w"), indent=1)
    print(f"[lab] T2-D642 fmpaths: {ARGS['--out']} ({len(FM_FORMS)} slots x 10 folds)", flush=True)
    raise SystemExit(0)
print(f"[lab] set {SET} = draws {DRAWS}: W19 level {level(REF)}; the lab's mix == read(W19) exactly; {len(_MEMO)} files cached "
      f"({time.time() - T0:.0f} s)", flush=True)


class Ctx:
    """what an idea sees: the folds, draws, labels-free stream keys, W19's parts and weights, the FM members' dumps, the teacher posteriors."""
    folds, draws = range(10), DRAWS
    P, W, fm_paths, dumps = P_ON, W_ON, FM_PATHS, staticmethod(dumps)
    teacher, conf_half, fit_head, softmax = staticmethod(teacher), staticmethod(conf_half), staticmethod(fit_head), staticmethod(softmax)
    d584_weights = staticmethod(d584_weights)                   # T2-D584
    stream, order, IX, CKS, W_DC = STREAM, ORD, IX, CKS, W_DC
    # T2-D562 (cross-fitted ideas only): labels / users, used ONLY on OTHER folds' stranger rows (the idea must exclude the scored fold's users)
    Y, U, TRU = Y, U, TRU
    TEACH, S_DIR, STREAMS_ALL = TEACH, S, STREAM                # T2-D566: the teacher files' dir, sofia_s1 (D498's members), every row's stream
    STARTS = STARTS                                             # T2-D570: None = the lab's cross-fitted starts; else the shipped constants
    MODE = MODE                                                 # T2-D642: the lab mode keys the FM-start records (fbteach_fb / mix_fb)
    lab_mix = staticmethod(lambda P, Wt, dec=None, post=None: lab_mix(P, Wt, dec, post=post))


if MODE == "lpteach":                                     # T2-D569 (registered 2026-11-18 before any number): the label-propagation teacher files
    # = the draw set's shipped teacher_p1 files with post_r<c> replaced, per stream and refit point c, by Zhou et al. 2004's propagation of the
    # prefix posterior over a kNN graph of the prefix windows' FM pooled features (the six members' L2-normalised features concatenated = D563's
    # fm space = the template's per-member buffers; centred by the prefix mean, L2-normalised; cosine; k nearest (self excluded), weights
    # max(cos, 0), symmetrised by max, S = D^-1/2 A D^-1/2, F = (I - alpha S)^-1 T0, clipped at 1e-9, rows renormalised) = the reviewer's
    # construction (T2_FRESH_IDEAS.md idea 4). Every other key is copied verbatim (R_*, vec_*, post, D_*: the FM refits' teacher is unchanged).
    # alpha 0 = the identity root: the files are copied byte for byte.
    import shutil as _sh
    out_ = Path(ARGS["--out"])
    alpha_, kk_ = float(ARGS.get("--alpha", "0.5")), int(ARGS.get("--k", "10"))
    assert out_.name == "teacher_p1" and out_.parent.name.startswith(("t2_d569_", "t2_d571_", "t2_d638_lp", "t2_d642_lp")) and out_.parent.name.endswith(f"_{DSET}"), out_   # T2-D638 / D642
    SRC_T = Path(ARGS["--src"]) if ARGS.get("--src") else TEACH   # T2-D569 amended: LP of the arm's REAL teacher (a t2_d571_ fb root's files)
    assert SRC_T == TEACH or (SRC_T.name == "teacher_p1" and SRC_T.parent.name.startswith(("t2_d571_fb", "t2_d638_fb", "t2_d642_fb")) and SRC_T.parent.name.endswith(f"_{DSET}")), SRC_T   # T2-D638 / D642
    assert not out_.exists() and tuple(CKS) == (60, 100, 150) and 0 <= alpha_ < 1 and kk_ >= 1, (out_, CKS, alpha_, kk_)
    out_.mkdir(parents=True)
    acc_ = {"shipped": [], "lp": []}
    for f in range(10):
        Fz = IDEAS._knn_feats(Ctx, f, "fm") if alpha_ > 0 else None
        posF = {int(r): i for i, r in enumerate(IX[f])}
        for d in DRAWS:
            src_ = SRC_T / f"lc9_{f}__s{d}.npz"
            if alpha_ == 0:
                _sh.copyfile(src_, out_ / src_.name)
                continue
            z = np.load(src_)
            new = {k: z[k] for k in z.files}
            rows = z["rows"].astype(np.int64)
            Fr = Fz[np.array([posF[int(r)] for r in rows])]
            st_, od_, held_ = STREAM[rows], ORD[rows], z["held"].astype(bool)
            for c in CKS:
                PR = z[f"post_r{c}"].astype(np.float64)
                PN = PR.copy()
                for s_ in np.unique(st_):
                    j = np.flatnonzero(st_ == s_)
                    j = j[np.argsort(od_[j], kind="stable")]
                    if len(j) <= c:
                        continue
                    a = j[:c]
                    T0 = PR[a]
                    assert np.isfinite(T0).all() and not np.isfinite(PR[j[c:]]).any(), (f, d, c, s_)
                    X = Fr[a] - Fr[a].mean(0)
                    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                    Sm = X @ X.T
                    np.fill_diagonal(Sm, -np.inf)
                    nn = np.argsort(-Sm, 1, kind="stable")[:, :kk_]
                    A_ = np.zeros((c, c))
                    rr = np.repeat(np.arange(c), kk_)
                    A_[rr, nn.ravel()] = np.maximum(Sm[rr, nn.ravel()], 0)
                    A_ = np.maximum(A_, A_.T)
                    dg = A_.sum(1) + 1e-9
                    Sn = A_ / np.sqrt(dg[:, None] * dg[None, :])
                    Fp = np.clip(np.linalg.solve(np.eye(c) - alpha_ * Sn, T0), 1e-9, None)
                    PN[a] = Fp / Fp.sum(1, keepdims=True)
                    if held_[a[0]]:                      # descriptive only (registered): the confident half's label accuracy, strangers
                        for nm, P_ in (("shipped", T0), ("lp", PN[a])):
                            ya, keep = conf_half(P_)
                            acc_[nm].append((c, float((ya[keep] == Y[rows][a][keep]).mean())))
                new[f"post_r{c}"] = PN.astype(np.float32)
            np.savez_compressed(out_ / src_.name, **new)
    n_ = len(list(out_.glob("lc9_*__s*.npz")))
    assert n_ == 30, n_
    if alpha_ > 0:
        for c in CKS:
            s0 = np.mean([v for c_, v in acc_["shipped"] if c_ == c])
            s1 = np.mean([v for c_, v in acc_["lp"] if c_ == c])
            print(f"[lab] D569 descriptive: strangers' confident-half label accuracy at c {c}: shipped {s0:.4f} -> LP {s1:.4f}", flush=True)
    print(f"[lab] D569 teacher files written: {out_} ({n_} files; alpha {alpha_}, k {kk_}; {'identity copy' if alpha_ == 0 else 'LP'})", flush=True)
    raise SystemExit(0)


if MODE == "lpgolden":                                    # T2-D569 W21 option: the LAB-vs-TEMPLATE golden test on real data (set A or B)
    # For every (fold, draw, stream, refit c) of the set: the template's StreamModel._cnn_lp_post (tracks/t2_bci/submission/
    # t2p_submission_w21lp.py) on a stub holding fp_feat = each member's RAW pooled features (the dumps, member order = the lab's FM slots) and
    # post = the fb root's teacher posterior of windows 0 .. c-1, vs the saved LP root's post_r<c> (float32) -> the max |diff|.
    import importlib.util as _ilu
    import types as _types
    sys.path.insert(0, "src")
    from ni26 import kit as _kit
    _kit.use_track("bci_decoding")
    _sp = _ilu.spec_from_file_location("t2p_w21lp", "tracks/t2_bci/submission/t2p_submission_w21lp.py")
    TM = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(TM)
    fbr, lpr = Path(ARGS["--fb"]), Path(ARGS["--lp"])
    dmax_, nblk_ = 0.0, 0
    for f in range(10):
        feats = [ctx_d["feat"] for ctx_d in (dumps(p_) for p_ in FM_PATHS[f])]
        posF = {int(r): i for i, r in enumerate(IX[f])}
        for d in DRAWS:
            zf, zl = np.load(fbr / f"lc9_{f}__s{d}.npz"), np.load(lpr / f"lc9_{f}__s{d}.npz")
            rows = zf["rows"].astype(np.int64)
            st_, od_ = STREAM[rows], ORD[rows]
            for c in CKS:
                for s_ in np.unique(st_):
                    j = np.flatnonzero(st_ == s_)
                    j = j[np.argsort(od_[j], kind="stable")]
                    if len(j) <= c:
                        continue
                    a = j[:c]
                    gi = np.array([posF[int(r)] for r in rows[a]])
                    stub = _types.SimpleNamespace(cnn_lp=dict(alpha=0.5, k=10), cnn_lp_stats=dict(refits=0, smoothed=0, skipped=0, errors=0),
                                                  fp_feat={i: [x[g] for x in feats] for i, g in enumerate(gi)})
                    out = TM.StreamModel._cnn_lp_post(stub, zf[f"post_r{c}"].astype(np.float64)[a], c)
                    assert stub.cnn_lp_stats["smoothed"] == 1 and stub.cnn_lp_stats["errors"] == 0, stub.cnn_lp_stats
                    dmax_ = max(dmax_, float(np.abs(out - zl[f"post_r{c}"].astype(np.float64)[a]).max()))
                    nblk_ += 1
    print(f"[lab] D569 GOLDEN (lab vs template on real data): {nblk_} stream x refit blocks; max |template - lab LP file| {dmax_:.2e}", flush=True)
    assert dmax_ < 1e-6, dmax_
    raise SystemExit(0)


if MODE == "fbteach":                                     # T2-D571 / D572 (registered 2026-11-18 before any number): the arm's REAL teacher as P1 teacher files
    # = the chain's teacher_p1 files with post_r<c> replaced by fm_feedback's sequential teacher at each refit c (CNN rows = the raw routed parts,
    # FM rows = the arm's CURRENT adapted FM part ('own') or the current members' unadapted outputs ('base', D572); starts per the lab convention,
    # or NI26_LAB_STARTS); the finite pattern of every post_r<c> is asserted equal to the chain file's. Every other key verbatim.
    out_ = Path(ARGS["--out"])
    rows_ = ARGS.get("--fm-rows", "own")
    assert out_.name == "teacher_p1" and out_.parent.name.startswith(("t2_d571_", "t2_d638_fb", "t2_d642_fb")) and out_.parent.name.endswith(f"_{DSET}") and not out_.exists(), out_   # T2-D638 / D642
    assert rows_ in ("own", "base") and tuple(CKS) == (60, 100, 150), (rows_, CKS)
    rec_ = {}
    twx_ = ARGS.get("--tw-fm")                            # T2-D591 / D655: the teacher's FM share (None = W21's 0.35); only for an fm45 / fm40 root
    _mx = __import__("re").search(r"fm(4[05]|55)", out_.parent.name)   # T2-D658: + fm55
    assert (twx_ is not None) == bool(_mx) and (not _mx or abs(float(twx_) - int(_mx.group(1)) / 100) < 1e-12), \
        ("T2-D591 / D655: --tw-fm goes with an fm45 / fm40 root, at its own share, and only there", out_, twx_)
    IDEAS.fm_feedback(Ctx, proto=float(ARGS.get("--proto", "0.5")), fm_rows=rows_, record=rec_, tw=IDEAS.tw_fm(float(twx_)) if twx_ else None)
    out_.mkdir(parents=True)
    dmax_, nrep_ = 0.0, 0
    for f in range(10):
        for d in DRAWS:
            z = np.load(TEACH / f"lc9_{f}__s{d}.npz")
            new = {k: z[k] for k in z.files}
            pos_ = {int(r): k for k, r in enumerate(z["rows"])}
            for c in CKS:
                PN = np.full(z[f"post_r{c}"].shape, np.nan)
                for c_, a, q in rec_[(f, d)]:
                    if c_ == c:
                        PN[np.array([pos_[int(r)] for r in IX[f][a]])] = q
                        nrep_ += 1
                assert np.array_equal(np.isfinite(PN).all(1), np.isfinite(z[f"post_r{c}"]).all(1)), (f, d, c)
                fin = np.isfinite(PN).all(1)
                dmax_ = max(dmax_, float(np.abs(PN[fin] - z[f"post_r{c}"][fin]).max()))
                new[f"post_r{c}"] = PN.astype(np.float32)
            np.savez_compressed(out_ / f"lc9_{f}__s{d}.npz", **new)
    n_ = len(list(out_.glob("lc9_*__s*.npz")))
    assert n_ == 30, n_
    print(f"[lab] D571 real-teacher files written: {out_} ({n_} files; fm_rows {rows_}; {nrep_} stream x refit blocks; "
          f"starts {'shipped ' + _ST if STARTS else 'lab'}; max |teacher - fixed teacher_p1| {dmax_:.4f})", flush=True)
    raise SystemExit(0)


BASE_W20 = ARGS.get("--base", "w19") == "w20"           # T2-D560 / W21: ideas judged ON TOP OF W20 (W19 + proto 0.5 = the owner's W20 line)
if BASE_W20:
    P_ON = IDEAS._fm_part(Ctx, proto=0.5)
    Ctx.P = P_ON
    REF = lab_mix(P_ON, W_ON)
    print(f"[lab] BASE = W20 (W19 + proto 0.5): level {level(REF)}; every idea is judged against W20", flush=True)


if MODE == "mix":                                         # T2-D564: save this process's W20 (or W20 + one registered idea) mixes for a 2-process read
    import os as _os
    r_ = {}
    if ARGS.get("--idea"):
        reg_ = [(n, fn, p) for n, fn, p in IDEAS.registry() if n == ARGS["--idea"]]
        assert reg_, ARGS["--idea"]
        r_ = reg_[0][1](Ctx, **reg_[0][2]) or {}
        print(f"[lab] mix with idea {reg_[0][0]} {reg_[0][2]}", flush=True)
    if ARGS.get("--base") == "w18":                      # T2-D573 (P3, the forecast anchor): W18's form = the reader's BASE = read([]); no idea
        assert not ARGS.get("--idea"), "w18: the reader's own BASE only"
        configure([])
        P18, W18w = {(f, d): parts_of(f, d, W18) for f in range(10) for d in DRAWS}, dict(W)
        _S18, STARTS = STARTS, None
        M_ = lab_mix(P18, W18w)
        R18 = read([])
        d18 = max(float(np.abs(M_[k][1] - R18[k][1]).max()) for k in R18)
        assert d18 == 0.0, f"the lab's W18 mix != read([]): {d18}"
        STARTS = _S18
        if STARTS:                                       # the shipped-starts convention (T2-D573 amended): the same parts, the shipped priors
            M_ = lab_mix(P18, W18w)
        print(f"[lab] W18 form: lab mix == read([]) exactly; starts {'shipped' if STARTS else 'recomputed'}", flush=True)
    else:
        if D606:                                         # T2-D606: the OUTPUT decoders (families + DC) past-only; every part / teacher as computed above
            _D606_ON = True
            print("[lab] T2-D606: past-only output calibration ON (q_t = normalize(p_t * b_{t-1}))", flush=True)
        if D607:                                         # T2-D607 (registered 0e7feec8): the FM output history re-scored under each installed head
            assert ARGS.get("--idea") == "w20_fb" and not D606 and not STARTS and not r_.get("post"), "T2-D607: W21's form (w20_fb), lab starts"
            assert not _os.environ.get("NI26_LAB_FMSTART"), "T2-D642: the FM-start hook is not wired into D607's mixes"
            RS = {}
            r7 = IDEAS.fm_feedback(Ctx, proto=0.5, rescore=RS)    # = the registry's w20_fb + the rescore read
            d7 = max(float(np.abs(r7["P"][k]["fm"] - r_["P"][k]["fm"]).max()) for k in r_["P"])
            assert d7 == 0.0, f"T2-D607: the instrumented fm_feedback changed the FM part ({d7})"
            CHK7 = dict(own=0.0, segs=0, rows=0)
            args7 = (r_.get("dec") or {}).get("fm", (10, 3, 0.95))

            def post607(f, d, key, D_, Wt, dcd, rs=None):
                fm = D_["fm"].copy()
                for (s_, c, e, jj, prob) in (RS if rs is None else rs).get((f, d), []):
                    CHK7["own"] = max(CHK7["own"], float(np.abs(prob[c:e] - r_["P"][(f, d)]["fm"][jj[c:e]]).max()))
                    o_ = causal_sinkhorn(prob, (np.zeros(len(jj), int), np.arange(len(jj))), args7[0], args7[1], args7[2], start=LAST_VEC[(f, d, "fm")])
                    fm[jj[c:e]] = o_[c:e]
                    CHK7["segs"] += 1
                    CHK7["rows"] += e - c
                D2 = dict(D_, fm=fm)
                return (1 - W_DC) * sum(Wt[fam] * D2[fam] for fam in Wt) + W_DC * dcd

            M0 = lab_mix(r_.get("P", P_ON), r_.get("W", W_ON), r_.get("dec"))
            M0p = lab_mix(r_.get("P", P_ON), r_.get("W", W_ON), r_.get("dec"), post=lambda f, d, key, D_, Wt, dcd: post607(f, d, key, D_, Wt, dcd, rs={}))
            assert all(np.array_equal(M0[k][1], M0p[k][1]) for k in M0), "T2-D607: the post hook without rescoring != lab_mix"
            M_ = lab_mix(r_.get("P", P_ON), r_.get("W", W_ON), r_.get("dec"), post=post607)
            pre7 = max(float(np.abs(M_[k][1] - M0[k][1])[ORD[M_[k][0]] < min(CKS)].max()) for k in M_)
            assert CHK7["own"] <= 1e-12 and pre7 == 0.0, (CHK7, pre7)
            print(f"[lab] T2-D607: FM history re-scored at {CHK7['segs']} installations ({CHK7['rows']} decoded rows changed source); own-window "
                  f"rows == the regular FM part (max |d| {CHK7['own']:.1e}); pre-refit == W21 (0.0); the post hook without rescoring == lab_mix", flush=True)
        else:
            _FMST_FINAL[0] = True                        # T2-D642: the FM-start hook acts on this (saved) mix only
            M_ = lab_mix(r_.get("P", P_ON), r_.get("W", W_ON), r_.get("dec"), post=r_.get("post"))
            _FMST_FINAL[0] = False
    assert not D606 or _D606_ON, "T2-D606: not with --base w18"
    assert not _ORDER or _ORDN[0] == 10, ("T2-D638: NI26_LAB_ORDER set but the saved mix was not decoded with the reordered key", _ORDN[0])
    assert _LAMX[0] is None or _LAMN[0] > 0, "T2-D656: NI26_LAB_LAM set but no decode of the saved mix used it"
    np.savez(ARGS["--out"], **{f"ix_{f}_{d}": ix for (f, d), (ix, mx) in M_.items()}, **{f"mx_{f}_{d}": mx for (f, d), (ix, mx) in M_.items()},
             teach=_os.environ.get("NI26_TEACH_DIR", ""), base=ARGS.get("--base", "w19"), idea=ARGS.get("--idea", ""), order=_ORDER,
             lam=-1.0 if _LAMX[0] is None else _LAMX[0])   # T2-D638; T2-D656: lam -1 = no override
    print(f"[lab] mix saved: {ARGS['--out']} (teacher {_os.environ.get('NI26_TEACH_DIR', 'shipped')}; level {level(M_)})", flush=True)
    for fam_ in ("fbc", "sh", "ifn", "fm"):                # T2-D570 (descriptive): the recomputed cross-fitted starts (mean over folds x draws)
        v_ = [v for k, v in LAST_VEC.items() if k[2] == fam_]
        if v_:
            print(f"[lab] starts {fam_}: recomputed mean {np.round(np.mean(v_, 0), 4).tolist()} (sd {np.round(np.std(v_, 0), 4).tolist()})"
                  + (f"; used {np.round(STARTS[fam_], 4).tolist()} (shipped)" if STARTS and fam_ in STARTS else "; used the recomputed"), flush=True)
    print(f"[lab] starts dc: recomputed mean {np.round(np.mean([DCVEC[f] for f in range(10)], 0), 4).tolist()}"
          + (f"; used {np.round(STARTS['dc'], 4).tolist()} (shipped)" if STARTS and 'dc' in STARTS else "; used the recomputed"), flush=True)
    raise SystemExit(0)


def run_idea(name, fn, params):
    t0 = time.time()
    r = fn(Ctx, **params) or {}
    P = r.get("P", P_ON)
    Wt = r.get("W", W_ON)
    A = lab_mix(P, Wt, r.get("dec"), post=r.get("post"))
    sc = judge(A, REF)
    per = np.array(list(sc["per_stranger"].values()))
    sc["se_ten"] = round(float(per.std(ddof=1) / np.sqrt(len(per))), 4)
    sc["guards"] = bool(sc["ten"] > 0 and sum(x > 0 for x in sc["draws"]) >= 2 and sc["min_user"] >= -0.020 and sc["returning"] >= -0.003)
    sc["stack_bar"] = bool(sc["ten"] >= 0.001 and sc["guards"] and sc["sealed_mix"] > 0)
    rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), set=SET, base="w20" if BASE_W20 else "w19", draw_set=list(DRAWS), idea=name, params=params, secs=round(time.time() - t0, 1), **sc)
    return rec


if MODE == "screen":
    batch = getattr(IDEAS, ARGS.get("--batch", "batch1"))()
    out = LAB / ARGS.get("--log", f"screen_{SET}.jsonl")   # --log: a separate file for descriptive runs (never a confirm source)
    recs = []
    for name, fn, params in batch:
        rec = run_idea(name, fn, params)
        recs.append(rec)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"[lab] {name} {params}: ten {rec['ten']:+.4f} (SE {rec['se_ten']:.4f}) draws {rec['draws']} min {rec['min_user']:+.4f} "
              f"ret {rec['returning']:+.4f} sealed {rec['sealed_mix']:+.4f} guards {rec['guards']} ({rec['secs']} s)", flush=True)
    ok = sorted([r for r in recs if r["guards"]], key=lambda r: -r["ten"])
    print(f"[lab] screened {len(recs)} configs in {time.time() - T0:.0f} s ({3600 * len(recs) / max(time.time() - T0, 1):.0f} / hour incl. "
          f"start-up); passing the guards: {len(ok)}; the pre-declared top {TOP_K}: {[(r['idea'], r['params'], r['ten']) for r in ok[:TOP_K]]}", flush=True)
elif MODE == "confirm":
    src = [json.loads(l_) for l_ in open(LAB / ARGS["--from"])]
    ok = sorted([r for r in src if r["guards"]], key=lambda r: -r["ten"])[:TOP_K]
    reg = {n: (fn, p) for n, fn, p in getattr(IDEAS, "registry")()}
    print(f"[lab] CONFIRM on set {SET}: the top {TOP_K} of {len(src)} screened configs ({sum(r['guards'] for r in src)} passed the guards)", flush=True)
    for r in ok:
        fn, _ = reg[r["idea"]]
        rec = run_idea(r["idea"], fn, r["params"])
        rec["screened_with"] = len(src)
        with open(LAB / f"confirm_{SET}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"[lab] CONFIRM {r['idea']} {r['params']}: ten {rec['ten']:+.4f} draws {rec['draws']} min {rec['min_user']:+.4f} ret {rec['returning']:+.4f} "
              f"sealed {rec['sealed_mix']:+.4f} -> {'PASS (stack bar)' if rec['stack_bar'] else 'FAIL'}", flush=True)
