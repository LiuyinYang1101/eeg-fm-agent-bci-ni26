#!/usr/bin/env python
"""T2-D529 / D532 (W18 build input; not a candidate by itself): one FM donor zip = a BASE donor (t2p_final --fm-donor, e.g. the SD donor with
window-mean side heads) with chosen FM member slots replaced by members of a SWAP donor (e.g. the head-less MI-expert member), for
t2p_repack --fm-from. Byte for byte: every base file except the replaced members' weights (and their side heads) is copied unchanged; the swap
member's weights enter under a new name fm_<slot>_swap.pt; its config entry is the swap donor's (no 'side': a head-less member stays head-less)
plus source = {zip, file}. Checks: the replaced and the incoming member agree on every inference key except the training-only / provenance ones
(t2p_repack's own rule), and on the input reference (plain / CAR); the merged config lists exactly the files the zip holds.
    python t2p_merge_donors.py --base SD.zip --swap MI.zip --slots 0 [--swap-index 0] --out MERGED.zip
T2-D546b (--keep-side): the swap members KEEP their window-mean side heads (fm_<slot>_swap_side.pt, the swap's side file byte for byte; its
config side entry with the new file name); only where the base member has a head too (a head-less slot stays head-less). Off = the earlier merge.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

SKIP = ("provenance", "cpt", "source", "file", "ref", "car", "seed", "tilt", "side", "cont", "mixup", "aug", "vp", "recomb", "phase", "cons", "wmargin", "user_balance", "user_balance_pow", "grl")   # t2p_repack --fm-from's allowed differences (T2-D642: + user_balance)
# (+ side; T2-D546b: + cont, a training-only record; T2-D557: + mixup / aug, training-only like t2p_repack's train_only, whose regex guard they share;
#  the night plan 2026-11-18 (the Fable review's blocker): + vp (D582) / recomb (D583) / phase (D586) / cons (D588) / wmargin (D596), training-only too)


def eff_ref(m):
    return m.get("ref") or ("car" if m.get("car") else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--swap", required=True)
    ap.add_argument("--slots", required=True, help="comma list of base fm_members indices to replace")
    ap.add_argument("--swap-index", default="", help="comma list of swap fm_members indices, one per slot (default 0, 1, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-side", action="store_true", help="T2-D546b: keep the swap members' side heads (default off = head-less swaps)")
    ap.add_argument("--allow-cpt-change", action="store_true", help="T2-D642 (Track3): allow a swap member fine-tuned from ANOTHER CPT checkpoint "
                    "than the slot it replaces (default: refused; 'cpt' is in SKIP, so it is checked here explicitly)")
    a = ap.parse_args()
    slots = [int(v) for v in a.slots.split(",")]
    sidx = [int(v) for v in a.swap_index.split(",")] if a.swap_index else list(range(len(slots)))
    assert len(sidx) == len(slots) and len(set(slots)) == len(slots), (slots, sidx)
    zb, zs = zipfile.ZipFile(a.base), zipfile.ZipFile(a.swap)
    cb, cs = json.loads(zb.read("config.json")), json.loads(zs.read("config.json"))
    mb, ms = cb["fm_members"], cs["fm_members"]
    assert all(m.get("weight") is None for m in mb + ms), "equal-weight FM slots only"
    drop, add, new_m = set(), {}, list(mb)
    for k, j in zip(slots, sidx):
        b, s = mb[k], ms[j]
        assert eff_ref(b) == eff_ref(s), f"slot {k}: input reference {eff_ref(b)} != the swap member's {eff_ref(s)}"
        diff = sorted(x for x in set(b) | set(s) if x not in SKIP and b.get(x) != s.get(x))
        assert not diff, f"slot {k}: the swap member differs in inference keys {diff}"
        cb_, cs_ = (b.get("cpt") or {}).get("sha256"), (s.get("cpt") or {}).get("sha256")   # T2-D642 (Track3 b): the CPT start must be the slot's
        assert a.allow_cpt_change or (cb_ and cs_ and cb_ == cs_), f"slot {k}: the swap member's CPT {cs_} != the slot's {cb_} (--allow-cpt-change to override)"
        if a.keep_side:              # T2-D546b: a headed swap replaces a headed base member (same kind / prior n0); the head file follows it
            assert s.get("side") and b.get("side") and s["side"]["kind"] == b["side"]["kind"] and s["side"]["n0"] == b["side"]["n0"], \
                f"slot {k}: --keep-side needs a side head on both members of the same kind / n0"
        else:
            assert not s.get("side"), "the swap member carries a side head: this tool keeps swap members head-less"
        drop.add(b["file"])
        if b.get("side"):
            drop.add(b["side"]["file"])
        def fresh(stem, ext):                         # T2-D557: a slot swapped before (fm_<k>_swap.pt) gets fm_<k>_swap2.pt, ...: a name
            j = 1                                     # never reused (the dropped base file's name stays out of the merged zip)
            while True:
                nm = f"{stem}{'' if j == 1 else j}{ext}"
                if nm not in zb.namelist() and nm not in add:
                    return nm
                j += 1
        nf = fresh(f"fm_{k}_swap", ".pt")
        assert nf not in zb.namelist(), nf
        add[nf] = s["file"]
        new_m[k] = {x: v for x, v in s.items() if x != "side"}
        new_m[k].update(file=nf, source=dict(zip=str(a.swap), file=s["file"], seed=s.get("seed")))
        if a.keep_side:
            nsf = nf.replace(".pt", "_side.pt")
            assert nsf not in zb.namelist(), nsf
            add[nsf] = s["side"]["file"]
            new_m[k]["side"] = dict(s["side"], file=nsf)
        print(f"[merge] slot {k}: base {b['file']} (seed {b.get('seed')}{', head ' + b['side']['file'] if b.get('side') else ''}) -> swap "
              f"{s['file']} (seed {s.get('seed')}) as {nf}; keys differing (allowed): "
              f"{sorted(x for x in set(b) | set(s) if b.get(x) != s.get(x))}", flush=True)
    cm = dict(cb, fm_members=new_m)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zo:
        for i in zb.infolist():
            if i.filename in drop:
                continue
            if i.filename == "config.json":
                zo.writestr("config.json", json.dumps(cm, indent=1))
                continue
            with zb.open(i) as r, zo.open(i.filename, "w", force_zip64=True) as w:
                shutil.copyfileobj(r, w, 1 << 24)
        for nf, sf in add.items():
            with zs.open(sf) as r, zo.open(nf, "w", force_zip64=True) as w:
                shutil.copyfileobj(r, w, 1 << 24)
    zm = zipfile.ZipFile(out)
    names = set(zm.namelist())
    want = {m["file"] for m in new_m} | {m["side"]["file"] for m in new_m if m.get("side")}
    assert want <= names, sorted(want - names)
    assert not (drop & names), sorted(drop & names)
    for i in zb.infolist():                         # every kept base entry byte for byte (CRC), the swapped-in weights too
        if i.filename not in drop and i.filename != "config.json":
            assert zm.getinfo(i.filename).CRC == i.CRC, i.filename
    for nf, sf in add.items():
        assert zm.getinfo(nf).CRC == zs.getinfo(sf).CRC, nf
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[merge] {out} ({out.stat().st_size / 1e9:.2f} GB, sha256 {sha[:16]}): {len(new_m)} FM members, heads on "
          f"{sum(bool(m.get('side')) for m in new_m)}; dropped {sorted(drop)}; CRCs equal to the sources", flush=True)


if __name__ == "__main__":
    main()
