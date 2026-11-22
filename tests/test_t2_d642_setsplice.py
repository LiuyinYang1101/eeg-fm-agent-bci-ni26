"""T2-D642 set-read splicer (scripts/local/t2_d642_setsplice.py), on set E's real lc9 FM files, in a temp dir (no cell pinned, nothing written
to the outputs tree):
  1. the NO-OP id splice: every array of every spliced file == set E's original (only 'res' gains the provenance);
  2. a synthetic B source (set E's rows perturbed + a different head) for each root kind s / p04: the scored group's per-row arrays == the
     source's rows (by row id, not position), every unscored row == set E's, the heads == the source's; the source-row-set mismatch is refused.
  python tests/test_t2_d642_setsplice.py   (needs outputs/t2_d642/setread/fmpaths_setE.json = `t2_lab.py fmpaths --set E --base w20`)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/local"))
import t2_d642_setsplice as S  # noqa: E402


def main():
    src_json = S.SR0 / "fmpaths_setE.json"
    assert src_json.exists(), f"run t2_lab.py fmpaths first: {src_json}"
    paths = S.fm_paths()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        S.SR0, S.D642 = td / "setread", td
        S.SR0.mkdir()
        (S.SR0 / "fmpaths_setE.json").write_text(src_json.read_text())
        S.pin_cell("bw21")                               # -> S.SR = td / setread / bw21
        S.pin_cell = lambda cell: None
        # 1. the no-op
        S.cmd_fmspl("id", "bw21")
        n = 0
        for f in S.SPLICED:
            for k in range(6):
                a, b = np.load(paths[f][k]), np.load(S.SR / "fmspl_id" / Path(paths[f][k]).name)
                assert sorted(set(b.files) - {"res"}) == sorted(set(a.files) - {"res"}), (f, k)
                for key in set(a.files) - {"res"}:
                    assert np.array_equal(a[key], b[key]), (f, k, key)
                assert json.loads(str(b["res"]))["d642_splice"]["tag"] == "id"
                n += 1
        print(f"1. no-op id splice: {n} files array-identical to set E's (res + provenance only)")
        # 2. synthetic sources: slots 1 (headed) + 0 (head-less MI) enter; the source = set E's scored rows, shuffled and perturbed, new heads
        (td / "members.json").write_text(json.dumps(dict(nominated=True, entering=["s0_mi", "s1_stlsd"])))
        rng = np.random.default_rng(0)
        fake = {}

        def make(slot, arm, fold, seed):
            k = S.SLOTS.index(slot)
            f = S.SPLICED[0] if fold.startswith("rt1_") else S.USERS.index(fold[4:])
            z = np.load(paths[f][k])
            te = z["test_idx"].astype(np.int64)
            m, src = S.scored("bs" if fold.startswith("ss8_") else "b" + fold[4:].lower(), f, te)
            assert src == fold, (src, fold)
            perm = rng.permutation(int(m.sum()))
            new = {"test_idx": te[m][perm], "val_idx": z["val_idx"], "prob_val_last": z["prob_val_last"]}
            for key in S.ROW_KEYS:
                if key in z.files:
                    new[key] = (z[key][m][perm] + 1.0).astype(z[key].dtype)
            for key in S.HEAD_KEYS:
                if key in z.files:
                    new[key] = (z[key] * 2.0 + 0.5).astype(z[key].dtype)
            p = td / "src" / f"{fold}__{slot}__{arm}__s{seed}.npz"
            p.parent.mkdir(exist_ok=True)
            np.savez(p, **new)
            fake[(slot, fold)] = p
            return p
        S.job_dir = lambda slot, arm: (slot, arm)
        S.member_file = lambda ds, slot, arm, fold, seed: fake.get((slot, fold)) or make(slot, arm, fold, seed)
        for tag in ("bs", "bp04"):
            S.cmd_fmspl(tag, "bw21")
            for f in S.SPLICED:
                for k in (0, 1):
                    a, b = np.load(paths[f][k]), np.load(S.SR / f"fmspl_{tag}" / Path(paths[f][k]).name)
                    te = a["test_idx"].astype(np.int64)
                    m, src = S.scored(tag, f, te)
                    zs = np.load(fake[(S.SLOTS[k], src)])
                    pos = {int(r): i for i, r in enumerate(zs["test_idx"])}
                    sel = np.array([pos[int(r)] for r in te[m]])
                    assert np.array_equal(b["test_idx"], a["test_idx"])
                    for key in S.ROW_KEYS:
                        if key in a.files:
                            assert np.array_equal(b[key][m], zs[key][sel]), (tag, f, k, key, "scored")
                            assert np.array_equal(b[key][~m], a[key][~m]), (tag, f, k, key, "unscored")
                    for key in S.HEAD_KEYS:
                        if key in a.files:
                            assert np.array_equal(b[key], zs[key]) and not np.array_equal(b[key], a[key]), (tag, f, k, key, "head")
                assert not any((S.SR / f"fmspl_{tag}" / Path(paths[f][k]).name).exists() for k in (2, 3, 4, 5)), "a non-entering slot was spliced"
            print(f"2. {tag}: scored rows <- the source (by row id), unscored rows = set E's, heads = the source's; slots 2-5 untouched")
        # 2b. W26's replicate tags: d = C' (arm dir Cp), t = T, both at seed 950 + slot; b / c at 900 + slot
        calls = []

        def rec(ds, slot, arm, fold, seed):
            calls.append((arm, seed - S.SLOTS.index(slot)))
            return fake.get((slot, fold)) or make(slot, arm, fold, seed)
        S.member_file = rec
        (td / "members_w27.json").write_text(json.dumps(dict(nominated=True, entering=["s1_stlsd"])))   # T2-D644: W27's own entering slots
        for tag, want in (("ds", ("Cp", 950)), ("tp04", ("T", 950)), ("cp10", ("C", 900)), ("es", ("E", 900)), ("gp04", ("B", 900))):
            calls.clear()
            S.cmd_fmspl(tag, "bw21")
            assert calls and set(calls) == {want}, (tag, set(calls))
            if tag[0] in "ge":
                assert json.loads((S.SR / f"fmspl_{tag}" / "manifest.json").read_text())["slots"] == ["s1_stlsd"], tag
        print("2b. tags ds / tp04 / cp10 / es / gp04 request arm Cp / T / C / E / B at seed 950 / 950 / 900 / 900 / 900 + slot; g / e splice W27's slots only")
        # 3. a source with the wrong rows is refused
        bad = td / "src" / "bad.npz"
        z = dict(np.load(fake[("s1_stlsd", "ss8_P01")]))
        z["test_idx"] = z["test_idx"] + 1
        np.savez(bad, **z)
        S.member_file = lambda ds, slot, arm, fold, seed: bad
        try:
            S.cmd_fmspl("cs", "bw21")
            raise SystemExit("FAIL: a source with the wrong rows was accepted")
        except AssertionError as e:
            assert "source rows" in str(e), e
        print("3. a source whose rows != the scored group's is refused")
    print("OK")


if __name__ == "__main__":
    main()
