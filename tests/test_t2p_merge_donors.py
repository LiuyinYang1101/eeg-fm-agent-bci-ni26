#!/usr/bin/env python
"""T2-D546b: tracks/t2_bci/scripts/t2p_merge_donors.py on small fake donor zips (CPU, no export, no torch):
  1. default (no --keep-side) = the earlier merge: a head-less swap member replaces a headed base slot, the base head file is dropped, the
     merged slot has no 'side'; a headed swap member is refused;
  2. --keep-side: the swap member's side file enters as fm_<slot>_swap_side.pt byte for byte (CRC), the config entry's side.file points at
     it, the base member's weights and head are dropped, every other base entry is copied byte for byte, 'cont' is an allowed difference;
  3. --keep-side refuses a head-less base slot (a head-less slot stays head-less).
  python tests/test_t2p_merge_donors.py  ->  "OK"
"""
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tracks/t2_bci/scripts/t2p_merge_donors.py"


def member(i, seed, side=True, **kw):
    m = dict(family="st", fm="st_large", file=f"fm_{i}.pt", seed=seed, epochs=100, mixup=0.4, llrd=0.75, rc=0.75, calib="gain",
             align=dict(mode="sz", scale=14.74, shrink=0.05), **kw)
    m.setdefault("cpt", dict(sha256="cpt-sha-same"))          # T2-D642: every member records its CPT start (the merge asserts the slot's)
    if side:
        m["side"] = dict(kind="dcmean", n0=10.0, file=f"fm_{i}_side.pt")
    return m


def make(path, members, extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps(dict(fm_members=members, other=1)))
        for m in members:
            z.writestr(m["file"], f"W{m['seed']}".encode() * 50)
            if m.get("side"):
                z.writestr(m["side"]["file"], f"H{m['seed']}".encode() * 10)
        for k, v in (extra or {}).items():
            z.writestr(k, v)


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = td / "base.zip"
        make(base, [member(0, 620, side=False), member(1, 621), member(2, 622), member(3, 623), member(4, 624), member(5, 625)],
             extra={"weights.pt": b"cnn" * 100})
        swap_h = td / "swap_h.zip"
        make(swap_h, [member(0, 622, cont=dict(epochs=25)), member(1, 624, cont=dict(epochs=25))])
        swap_n = td / "swap_n.zip"
        make(swap_n, [member(0, 622, side=False)])
        # 1. default: head-less swap merges as before; a headed swap is refused
        r = run("--base", str(base), "--swap", str(swap_n), "--slots", "2", "--out", str(td / "m0.zip"))
        assert r.returncode == 0, r.stderr
        z = zipfile.ZipFile(td / "m0.zip")
        c = json.loads(z.read("config.json"))
        assert "side" not in c["fm_members"][2] and "fm_2.pt" not in z.namelist() and "fm_2_side.pt" not in z.namelist()
        assert "fm_2_swap.pt" in z.namelist() and "fm_1_side.pt" in z.namelist()
        r = run("--base", str(base), "--swap", str(swap_h), "--slots", "2", "--out", str(td / "m1.zip"))
        assert r.returncode != 0 and "head-less" in (r.stderr + r.stdout), "default must refuse a headed swap"
        print("default: head-less swap merged as before; headed swap refused")
        # 2. --keep-side
        r = run("--base", str(base), "--swap", str(swap_h), "--slots", "2,4", "--swap-index", "0,1", "--keep-side", "--out", str(td / "m2.zip"))
        assert r.returncode == 0, r.stderr
        z, zs, zb = zipfile.ZipFile(td / "m2.zip"), zipfile.ZipFile(swap_h), zipfile.ZipFile(base)
        c = json.loads(z.read("config.json"))
        for k, j in ((2, 0), (4, 1)):
            m = c["fm_members"][k]
            assert m["file"] == f"fm_{k}_swap.pt" and m["side"]["file"] == f"fm_{k}_swap_side.pt" and m["cont"] == dict(epochs=25), m
            assert z.getinfo(f"fm_{k}_swap.pt").CRC == zs.getinfo(f"fm_{j}.pt").CRC
            assert z.getinfo(f"fm_{k}_swap_side.pt").CRC == zs.getinfo(f"fm_{j}_side.pt").CRC
            assert f"fm_{k}.pt" not in z.namelist() and f"fm_{k}_side.pt" not in z.namelist()
        for n in ("fm_0.pt", "fm_1.pt", "fm_1_side.pt", "fm_3.pt", "fm_3_side.pt", "fm_5.pt", "fm_5_side.pt", "weights.pt"):
            assert z.getinfo(n).CRC == zb.getinfo(n).CRC, n
        assert c["other"] == 1 and [m["seed"] for m in c["fm_members"]] == [620, 621, 622, 623, 624, 625]
        print("--keep-side: side heads follow the swap byte for byte; the rest of the base unchanged")
        # 3. --keep-side refuses a head-less base slot
        r = run("--base", str(base), "--swap", str(swap_h), "--slots", "0", "--keep-side", "--out", str(td / "m3.zip"))
        assert r.returncode != 0, "--keep-side must refuse a head-less base slot"
        print("--keep-side: a head-less base slot refused")
        # T2-D557: an augmented swap (mixup 0.8 + aug, training-only) merges with --keep-side and keeps its own training record
        swap_a = td / "swap_a.zip"
        ma = member(0, 621)
        ma.update(mixup=0.8, aug="0.2:30:0.1")
        make(swap_a, [ma])
        r = run("--base", str(base), "--swap", str(swap_a), "--slots", "1", "--swap-index", "0", "--keep-side", "--out", str(td / "ma.zip"))
        assert r.returncode == 0, r.stderr
        ca = json.loads(zipfile.ZipFile(td / "ma.zip").read("config.json"))["fm_members"][1]
        assert ca["mixup"] == 0.8 and ca["aug"] == "0.2:30:0.1" and ca["file"] == "fm_1_swap.pt" and ca["side"]["file"] == "fm_1_swap_side.pt", ca
        mb = member(0, 621)
        mb.update(rc=0.5)                                  # an inference-relevant difference is still refused
        make(td / "swap_b.zip", [mb])
        r = run("--base", str(base), "--swap", str(td / "swap_b.zip"), "--slots", "1", "--swap-index", "0", "--keep-side", "--out", str(td / "mb.zip"))
        assert r.returncode != 0, "a non-training difference must be refused"
        print("5. T2-D557: an augmented swap (mixup / aug differ) merges; an rc difference is still refused", flush=True)
        # 6. T2-D557: re-swapping a slot that was swapped before (fm_2_swap.pt in the base) -> fm_2_swap2.pt (+ its side), no name reused
        r = run("--base", str(td / "ma.zip"), "--swap", str(swap_a), "--slots", "1", "--swap-index", "0", "--keep-side", "--out", str(td / "mc.zip"))
        assert r.returncode == 0, r.stderr
        zc = zipfile.ZipFile(td / "mc.zip")
        cc = json.loads(zc.read("config.json"))["fm_members"][1]
        assert cc["file"] == "fm_1_swap2.pt" and cc["side"]["file"] == "fm_1_swap2_side.pt" and "fm_1_swap.pt" not in zc.namelist(), (cc, zc.namelist())
        print("6. T2-D557: a re-swapped slot gets fm_<k>_swap2.pt (+ side); the dropped name is not reused", flush=True)
        # 7. the night plan 2026-11-18 (the Fable review's blocker): the W22 donors' training-only keys vp (D582) / recomb (D583) / phase (D586)
        #    / cons (D588), each alone and all together, merge with --keep-side and keep their record; an inference key is still refused
        for k_, v_ in (("vp", 2), ("recomb", "4:0.5"), ("phase", "0.25:0.7854"), ("cons", 1.0), ("wmargin", 0.3), (None, None)):
            mk = member(0, 621)
            mk.update(mixup=0.8, aug="0.2:30:0.1")
            mk.update({k_: v_} if k_ else dict(vp=2, recomb="4:0.5", phase="0.25:0.7854", cons=1.0, wmargin=0.3))
            make(td / f"swap_{k_}.zip", [mk])
            r = run("--base", str(base), "--swap", str(td / f"swap_{k_}.zip"), "--slots", "1", "--swap-index", "0", "--keep-side",
                    "--out", str(td / f"m_{k_}.zip"))
            assert r.returncode == 0, (k_, r.stderr[-400:])
            ck = json.loads(zipfile.ZipFile(td / f"m_{k_}.zip").read("config.json"))["fm_members"][1]
            assert all(ck.get(a_) == mk.get(a_) for a_ in ("vp", "recomb", "phase", "cons", "wmargin")), (k_, ck)
        mk = member(0, 621)
        mk.update(recomb="4:0.5", llrd=0.5)                # recomb allowed, llrd (inference-relevant) still refused
        make(td / "swap_bad.zip", [mk])
        r = run("--base", str(base), "--swap", str(td / "swap_bad.zip"), "--slots", "1", "--swap-index", "0", "--keep-side", "--out", str(td / "m_bad.zip"))
        assert r.returncode != 0, "a non-training difference next to a new training key must still be refused"
        print("7. W22 donors: vp / recomb / phase / cons (alone and together) merge as training-only; llrd is still refused", flush=True)
    src = (ROOT / "tracks/t2_bci/scripts/t2p_repack.py").read_text()
    tl = src.split("train_only = (")[1].split(")")[0]
    assert all(f'"{k_}"' in tl for k_ in ("vp", "recomb", "phase", "cons", "wmargin")), tl
    import re
    pat = re.search(r"assert not re\.search\(r'(.+?)', src\)", src).group(1)
    for bad in ('m.get("recomb")', 'm_["phase"]', '"cons" in m', 'm["vp"]', 'm.get("wmargin")'):
        assert re.search(pat, bad), ("the template-read guard misses", bad)
    for t_ in ("t2p_submission_w21lp.py", "t2p_submission_w19.py"):
        assert not re.search(pat, (ROOT / "tracks/t2_bci/submission" / t_).read_text()), t_
    print("8. t2p_repack: train_only holds vp / recomb / phase / cons; its template-read guard catches each key and passes W19 / W21lp")
    # T2-D642 (Track3 b): a swap member from ANOTHER CPT checkpoint is refused unless --allow-cpt-change
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        make(td / "b.zip", [member(0, 620, side=False), member(1, 621)])
        make(td / "s.zip", [member(0, 721, cpt=dict(sha256="cpt-sha-OTHER"))])
        r = run("--base", str(td / "b.zip"), "--swap", str(td / "s.zip"), "--slots", "1", "--keep-side", "--out", str(td / "m.zip"))
        assert r.returncode != 0 and "CPT" in (r.stderr + r.stdout), (r.returncode, r.stderr[-300:])
        r = run("--base", str(td / "b.zip"), "--swap", str(td / "s.zip"), "--slots", "1", "--keep-side", "--allow-cpt-change", "--out", str(td / "m2.zip"))
        assert r.returncode == 0, r.stderr[-500:]
    print("9. T2-D642: a swap member from another CPT checkpoint is refused; --allow-cpt-change admits it")
    print("OK")


if __name__ == "__main__":
    main()
