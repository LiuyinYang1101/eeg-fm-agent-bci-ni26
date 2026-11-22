#!/usr/bin/env python
"""T2-D642 x9 -> W25-ALL6: the STRUCTURAL W21-vs-W25 check, a HARD gate before any promotion (Track3's checklist): W25 must be provably "W21
with its six FM members swapped".
  1. every non-FM file of the zip is byte-identical to W21's (submission.py, weights*.pt, *_pu.pt, tta_vp.pt, dcfam.npz, side.pt, ...): the
     same name set, the same sha256;
  2. config.json equal to W21's outside fm_members, except an explicit allow-list (repo_commit: the build snapshot);
  3. per FM slot k: the member entry equal to W21's except file / source / seed / user_balance / provenance / cont (and the side entry's file);
     the CPT sha256 == W21's slot k; user_balance present (all six balanced); a member has a side head iff W21's slot k has one;
  4. with --donor (the merged donor zip): every W25 side-head file is byte-identical to a side file of the donor (the head follows its member);
  4b. (Track3's MUST-ADD) with --donor, SLOT-ALIGNED: sha(W25 slot k member) == sha(donor slot k member), and the same for each side file
     (t2p_repack --fm-from copies members byte for byte, so this holds exactly; a stale / wrong donor fails here);
  5. (optional) with --d642 MI,A,B,C,K2,K4 (the D642 donor zips): the donor slot k member == that zip's index-0 member (the chain back to the
     trained balanced members), slots 0 / 1 / 3 / 5 / 2 / 4 = MI / A / B / C / cont-622 / cont-624;
  5-lite (Track3; no download) with --d642-ids MI,A,B,C,K2,K4 (the donor JOB ids): per slot k the merged donor's member records source.zip
     inside that job's dir, source.file == 'fm_0.pt' (the swap's index-0 member) and seed == 620 + k.
  python t2p_w25_zipcheck.py --base W21.zip --cand W25.zip [--donor MERGED_DONOR.zip] [--out RESULT.json]   -> exit 0 = PASS, 1 = FAIL
  Self-test (W21 vs W21): fails ONLY 'no user_balance key' + 'the member weights are W21's own' on all six slots; every other check passes.
"""
import argparse
import hashlib
import json
import sys
import zipfile

CFG_ALLOW = {"repo_commit"}                                         # build-snapshot provenance only
MEMBER_ALLOW = {"file", "source", "seed", "user_balance", "user_balance_pow", "grl", "vp", "provenance", "cont"}   # + grl (D649) / vp (D652): training-only
SIDE_ALLOW = {"file"}


_SHA = {}


def sha(z, n):
    """sha256 of member n of zip z (cached: each ~200 MB FM file is decompressed once)."""
    k = (z.filename, n)
    if k not in _SHA:
        h = hashlib.sha256()
        with z.open(n) as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        _SHA[k] = h.hexdigest()
    return _SHA[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--donor", default="")
    ap.add_argument("--d642", default="", help="comma list MI,A,B,C,K2,K4 of the D642 donor zips (optional check 5)")
    ap.add_argument("--d642-ids", default="", help="comma list MI,A,B,C,K2,K4 of the D642 donor JOB ids (check 5-lite, metadata only)")
    ap.add_argument("--out", default="")
    ap.add_argument("--keep-slots", default="", help="T2-D652 one-hop (W30 vs W25-ALL6): comma list of slots that must be byte-identical to the BASE (W30 keeps slot 0 = the MI donor); every other slot must differ")
    a = ap.parse_args()
    keep = {int(x) for x in a.keep_slots.split(",") if x != ""}
    zb, zc = zipfile.ZipFile(a.base), zipfile.ZipFile(a.cand)
    cb, cc = json.loads(zb.read("config.json")), json.loads(zc.read("config.json"))
    fails, notes = [], []
    fm_b = {m["file"] for m in cb["fm_members"]} | {m["side"]["file"] for m in cb["fm_members"] if m.get("side")}
    fm_c = {m["file"] for m in cc["fm_members"]} | {m["side"]["file"] for m in cc["fm_members"] if m.get("side")}
    nb, nc = set(zb.namelist()) - fm_b - {"config.json"}, set(zc.namelist()) - fm_c - {"config.json"}
    if nb != nc:
        fails.append(f"1. non-FM file sets differ: only base {sorted(nb - nc)}, only cand {sorted(nc - nb)}")
    for n in sorted(nb & nc):
        if sha(zb, n) != sha(zc, n):
            fails.append(f"1. non-FM file {n} differs")
    notes.append(f"1. {len(nb & nc)} non-FM files compared byte for byte")
    miss_c = fm_c - set(zc.namelist())                     # Track3's nit: every FM / side file the cand config names exists (no stray files: check 1)
    if miss_c:
        fails.append(f"1. files named in cand's config but missing from the zip: {sorted(miss_c)}")
    keys = (set(cb) | set(cc)) - {"fm_members"}
    for k in sorted(keys):
        if cb.get(k) != cc.get(k):
            (notes if k in CFG_ALLOW else fails).append(f"2. config['{k}'] differs{' (allowed)' if k in CFG_ALLOW else ''}")
    if len(cb["fm_members"]) != len(cc["fm_members"]):
        fails.append(f"3. fm_members count {len(cb['fm_members'])} vs {len(cc['fm_members'])}")
    for k, (mb, mc) in enumerate(zip(cb["fm_members"], cc["fm_members"])):
        d = sorted(x for x in set(mb) | set(mc) if x not in MEMBER_ALLOW | {"side", "cpt"} and mb.get(x) != mc.get(x))
        if d:
            fails.append(f"3. slot {k}: member keys differ {d}")
        sb, sc_ = (mb.get("cpt") or {}).get("sha256"), (mc.get("cpt") or {}).get("sha256")
        if not (sb and sc_ and sb == sc_):
            fails.append(f"3. slot {k}: CPT sha {sc_} != W21's {sb}")
        if not mc.get("user_balance"):
            fails.append(f"3. slot {k}: no user_balance key (not a balanced member)")
        if bool(mb.get("side")) != bool(mc.get("side")):
            fails.append(f"3. slot {k}: side head presence differs (base {bool(mb.get('side'))}, cand {bool(mc.get('side'))})")
        elif mb.get("side"):
            ds = sorted(x for x in set(mb["side"]) | set(mc["side"]) if x not in SIDE_ALLOW and mb["side"].get(x) != mc["side"].get(x))
            if ds:
                fails.append(f"3. slot {k}: side entry keys differ {ds} (the z-prior / kind / n0 must match)")
        notes.append(f"3. slot {k}: seed {mb.get('seed')} -> {mc.get('seed')}, file {mb['file']} -> {mc['file']}, "
                     f"weights sha {sha(zb, mb['file'])[:12]} -> {sha(zc, mc['file'])[:12]}, user_balance {mc.get('user_balance')!s:.60}")
        if k in keep:                                    # T2-D652 one-hop: a kept slot is the base's own member, byte for byte
            if sha(zb, mb["file"]) != sha(zc, mc["file"]):
                fails.append(f"3. slot {k}: a --keep-slots member differs from the base's")
        elif sha(zb, mb["file"]) == sha(zc, mc["file"]):
            fails.append(f"3. slot {k}: the member weights are W21's own (not swapped)")
        if mc.get("grl") is not None:                   # T2-D649 (pinned): the adversary is training-only: no adversary key in the shipped member
            import io
            import torch
            st = torch.load(io.BytesIO(zc.read(mc["file"])), map_location="cpu", weights_only=False)
            def keys_(o, pre=""):
                if isinstance(o, dict):
                    for k_, v_ in o.items():
                        yield f"{pre}{k_}"
                        yield from keys_(v_, f"{pre}{k_}.")
            bad = [k_ for k_ in keys_(st) if "adv" in k_.lower().split(".")[-1] or k_.lower().startswith("adv") or ".adv" in k_.lower()]
            if bad:
                fails.append(f"3. slot {k}: adversary keys in the shipped member {mc['file']}: {bad[:3]}")
            notes.append(f"3. slot {k}: grl member {mc['file']}: no adversary key ({sum(1 for _ in keys_(st))} keys scanned)")
    if a.donor:
        zd = zipfile.ZipFile(a.donor)
        dside = {sha(zd, n) for n in zd.namelist() if n.endswith("_side.pt")}
        for k, mc in enumerate(cc["fm_members"]):
            if mc.get("side") and sha(zc, mc["side"]["file"]) not in dside:
                fails.append(f"4. slot {k}: side head {mc['side']['file']} is not byte-identical to any donor side file")
        notes.append(f"4. side heads checked against the donor's {len(dside)} side files")
        cd = json.loads(zd.read("config.json"))                 # 4b: slot-aligned byte equality with the merged donor
        assert len(cd["fm_members"]) == len(cc["fm_members"]), (len(cd["fm_members"]), len(cc["fm_members"]))
        for k, (md, mc) in enumerate(zip(cd["fm_members"], cc["fm_members"])):
            if sha(zc, mc["file"]) != sha(zd, md["file"]):
                fails.append(f"4b. slot {k}: the member {mc['file']} != the donor's slot-{k} member {md['file']}")
            if bool(md.get("side")) != bool(mc.get("side")):
                fails.append(f"4b. slot {k}: side presence differs from the donor's")
            elif mc.get("side") and sha(zc, mc["side"]["file"]) != sha(zd, md["side"]["file"]):
                fails.append(f"4b. slot {k}: the side head != the donor's slot-{k} side head")
        notes.append("4b. every slot's member (+ side head) compared byte for byte with the donor's slot-aligned files")
        DCFG_ALLOW = {"file", "source", "provenance"}      # T2-D652 one-hop (the manager's (b)): the config block follows the donor's, key by key
        for k, (md, mc) in enumerate(zip(cd["fm_members"], cc["fm_members"])):
            dk = sorted(x for x in set(md) | set(mc) if x not in DCFG_ALLOW | {"side"} and md.get(x) != mc.get(x))
            if dk:
                fails.append(f"4c. slot {k}: the fm_members entry differs from the donor's in {dk}")
            if md.get("side") or mc.get("side"):
                ds_ = sorted(x for x in set(md.get("side") or {}) | set(mc.get("side") or {}) if x != "file" and (md.get("side") or {}).get(x) != (mc.get("side") or {}).get(x))
                if ds_:
                    fails.append(f"4c. slot {k}: the side entry differs from the donor's in {ds_}")
        notes.append("4c. every slot's fm_members entry == the donor's key by key (file / source / provenance aside; side entries too)")
        if a.d642:
            zs = [zipfile.ZipFile(x) for x in a.d642.split(",")]
            assert len(zs) == 6, "--d642 MI,A,B,C,K2,K4"
            for k, zi in zip((0, 1, 3, 5, 2, 4), zs):           # 5: the donor slot k member == the D642 donor's index-0 member
                m0 = json.loads(zi.read("config.json"))["fm_members"][0]
                if sha(zd, cd["fm_members"][k]["file"]) != sha(zi, m0["file"]):
                    fails.append(f"5. slot {k}: the donor member != {zi.filename}'s index-0 member")
            notes.append("5. the donor's six members traced byte for byte to the D642 donor zips' index-0 members")
        if a.d642_ids:                                          # 5-lite: the merged donor's own provenance, per slot
            ids = a.d642_ids.split(",")
            assert len(ids) == 6, "--d642-ids MI,A,B,C,K2,K4"
            for k, jid in zip((0, 1, 3, 5, 2, 4), ids):
                md, src = cd["fm_members"][k], cd["fm_members"][k].get("source") or {}
                if f"-{jid}/" not in str(src.get("zip", "")) or src.get("file") != "fm_0.pt" or md.get("seed") != 620 + k:
                    fails.append(f"5-lite. slot {k}: source {src} / seed {md.get('seed')} != the D642 donor job {jid}'s fm_0.pt, seed {620 + k}")
            notes.append(f"5-lite. the donor's six members record the D642 donor jobs {ids} (fm_0.pt, seeds 620-625)")
    def fsha(path):                                             # the whole zip's sha256 (the promotion ties this check to the exact bytes)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        return h.hexdigest()
    res = dict(base=a.base, cand=a.cand, donor=a.donor, base_sha256=fsha(a.base), cand_sha256=fsha(a.cand),
               donor_sha256=fsha(a.donor) if a.donor else "", passes=not fails, fails=fails, notes=notes)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
    print("\n".join(notes))
    print(("PASS" if not fails else "FAIL") + "".join(f"\n  !! {f}" for f in fails))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
