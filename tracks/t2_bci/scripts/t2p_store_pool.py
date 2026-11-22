"""T2-D356 / T2-FM-D2: a store-backed CPT pool for t2p_cpt.py: the in-house FM project's consolidated continuous stores, read through the in-house
FM project's OWN loader (ContinuousWindowDataset + splits.pool_paths from the fork snapshot on PYTHONPATH), with C's frozen split as-is.

What it does, and nothing else:
  * train pool = pool_paths(split) (the trainer's in_pretrain pool), val pool = pool_paths(split, tier=1, qc=True) (its val
    tier): the split decides every recording (the in-house FM project's holdouts, the ni26 holdouts of data/fm/ni26_only); the masks file
    (docs/data/pool_masks.json of the snapshot) drops bad channels / spans exactly as the in-house FM project's trainer does;
  * windows: 4 s at hop --store-hop on the 256 Hz store, anti-alias decimated to 128 Hz by the loader -> (C, 512), ST's input;
  * channels: each stored name -> an ST-EEGFormer table name (rule `st_name`): exact (case-insensitive) after dropping an
    'EEG ' prefix and a '-REF' / '-LE' / '-AVG' suffix; 'X-M1' / 'X_M2' / 'X-A1' (a mastoid / ear reference) -> X; EGI
    HydroCel 'E<n>' through tracks/t2_bci/fm/egi_to_st.json (mutual nearest neighbours <= 6 deg); every other name (bipolar
    derivations such as 'Fpz-Cz' / 'FP1-F7' / 'F3_F4', EGI's reference Cz / VREF, unknown names) is dropped; a recording keeps
    only if >= --store-min-ch channels map; recordings are grouped by (source, mapped channel set), one group per micro-batch;
  * draws: source = the split's `_source` (dataset[:accession]); p(source) ∝ sqrt(train windows) (the in-house FM project's dataset-temp
    0.5); within a source p(group) ∝ windows; the sleep-PSG sources (the in-house FM project's weight_policy group) are damped only if their
    DELIVERED TOKEN share would exceed --store-sleep-share (the in-house FM project's 0.25 rule, solved in closed form);
  * PROTEUS streams: ni26_proteus rows (patient 'ni26_proteus:P01', session 'Dreyer2026Proteus-P01-1-...') -> 'P01|1'; an
    unparseable row or a J-CPT holdout stream (P04 / P05 / P08 / P09, P10 s2-3) refuses the pool;
  * record: split + masks + map sha256, the snapshot's SYNC_VERSION, per-source split hours vs indexed hours (windows x hop),
    every opened shard (path, size, mtime): the read-in-place manifest the in-house FM project asked for.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EGI_MAP = REPO / "tracks/t2_bci/fm/egi_to_st.json"
ST_TABLE = REPO / "tracks/t1_image/fm/steeg_channels.json"
# the in-house FM project's weight policy group (fork configs/ni26fm/weight_policy.json, 42f1594): sleep PSG incl. the ni26 PSG-Muse views
SLEEP_PSG = ("physionet2018", "sleepedf_ext", "hmc_sleep", "dreem_sleep", "ni26_pn18_muse", "ni26_hmc_muse", "ni26_dod_muse")
JCPT_HOLDOUT_USERS = {"P04", "P05", "P08", "P09"}
# T2-FM-D2 amendment (2026-11-05, before any number; the in-house FM project's C treats HBN as ONE family, 'fam:hbn'): the HBN-EEG releases are one
# dataset split into 11 OpenNeuro accessions; each would otherwise draw its own sqrt-share (11x HBN). They form one source 'hbn'.
HBN_RELEASES = {f"ds0055{k:02d}" for k in range(5, 17)} - {"ds005513"}
FAMILY = {f"scalp_bids:{a}": "hbn" for a in HBN_RELEASES}
VAL_SEED = 20261013
WIN_S, FS = 4.0, 128
REF_SUFFIX = re.compile(r"[-_ ](REF|LE|AVG|AR)$", re.I)
MASTOID = re.compile(r"^(?P<a>[A-Za-z0-9]+)[-_](M1|M2|A1|A2)$", re.I)
PROTEUS_SES = re.compile(r"^Dreyer2026Proteus-(P\d\d)-([0-9]+)-")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def egi_net(names):
    """'GSN-HydroCel-129' | 'GSN-HydroCel-257' | None, from the stored names (every name E<n>, Cz or VREF)."""
    e = [n for n in names if re.fullmatch(r"E\d+", n)]
    if not e or set(names) - set(e) - {"Cz", "VREF"}:
        return None
    return "GSN-HydroCel-129" if max(int(n[1:]) for n in e) <= 129 else "GSN-HydroCel-257"


def st_name(raw, low, egi=None):
    """(ST table name or None, rule) for one stored channel name; `egi` = that recording's EGI map (or None)."""
    if egi is not None:
        return (egi.get(raw), "egi") if re.fullmatch(r"E\d+", raw) else (None, "egi_ref")
    n = raw.strip()
    n = n[4:].strip() if n.upper().startswith("EEG ") else n
    n = REF_SUFFIX.sub("", n)
    if n.lower() in low:
        return low[n.lower()], "direct"
    m = MASTOID.match(n)
    if m and m.group("a").lower() in low:
        return low[m.group("a").lower()], "mastoid_ref"
    return None, "dropped"


def proteus_stream(patient_id, session_id):
    """'P01|1' for an ni26_proteus row; raises on anything else (never guess a stream)."""
    m = PROTEUS_SES.match(str(session_id))
    u = str(patient_id).split(":", 1)[-1]
    if not m or m.group(1) != u:
        raise SystemExit(f"store pool: ni26_proteus row ({patient_id}, {session_id}) has no parseable PROTEUS stream")
    return m.group(1), m.group(2)


class StorePool:
    def __init__(self, split, roots, src, log, masks=None, hop_s=4.0, min_ch=4, val_per_source=128, sleep_share=0.25,
                 batch=64, token_scale=True, index_workers=None, sources=None):
        t0 = time.time()
        src = Path(src)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from inhouse_fm.data.continuous_dataset import ContinuousWindowDataset, portable_key
        from inhouse_fm.data.splits import pool_paths
        import inhouse_fm
        mod = Path(inhouse_fm.__file__).resolve()
        assert mod.is_relative_to(src.resolve()), f"inhouse_fm imported from {mod}, not the snapshot {src}"
        snap = src.resolve().parent
        masks = Path(masks) if masks else snap / "docs/data/pool_masks.json"
        assert masks.exists(), f"store pool: masks file {masks} missing (the in-house FM project's loader would train UNMASKED)"
        roots = [r for r in str(roots).split(",") if r]
        assert len({Path(r).name for r in roots}) == len(roots), f"root basenames must differ: {roots}"
        sp = pd.read_parquet(split)
        keep, vkeep = pool_paths(sp), pool_paths(sp, tier=1, qc=True)
        sp = sp.assign(_pkx=sp["out_path"].astype(str).map(portable_key))
        srccol = sp["_source"].astype(str) if "_source" in sp else sp["dataset"].astype(str)
        sp = sp.assign(_src=np.where(srccol.isin(["", "nan", "None"]), sp["dataset"].astype(str), srccol))
        sp = sp.assign(_src=sp["_src"].map(lambda v: FAMILY.get(v, v)))     # the HBN releases -> 'hbn' (one source)
        pk_src = dict(zip(sp["_pkx"], sp["_src"]))
        pk_ds = dict(zip(sp["_pkx"], sp["dataset"].astype(str)))
        if sources:                                       # T2-FM-D6: only these sources (the split's _source / dataset)
            allowed = {pk for pk, s_ in pk_src.items() if s_ in set(sources) or pk_ds.get(pk) in set(sources)}
            assert allowed, f"store pool: no split row of the sources {sources}"
            keep = {k for k in keep if k in allowed or portable_key(k) in allowed}
            vkeep = {k for k in vkeep if k in allowed or portable_key(k) in allowed}
            log(f"store pool restricted to the sources {sources}: {len(allowed)} recordings in the split")
        trk = {k for k in keep if "/" in k and not k.startswith("/")}        # portable keys of the train pool
        vak = {k for k in vkeep if "/" in k and not k.startswith("/")}
        hours = sp[sp["_pkx"].isin(trk)].groupby("_src")["duration_s"].sum() / 3600
        # PROTEUS streams from the split rows of the TRAIN pool (the val tier is never trained on, but is recorded too)
        pr = sp[(sp["dataset"].astype(str) == "ni26_proteus") & sp["_pkx"].isin(trk | vak)]
        self.proteus_streams = {proteus_stream(p, s) for p, s in zip(pr["patient_id"], pr["session_id"])}
        bad = {f"{u}|{s}" for u, s in self.proteus_streams if u in JCPT_HOLDOUT_USERS or (u == "P10" and s != "1")}
        assert not bad, f"store pool: J-CPT holdout PROTEUS streams in the split: {sorted(bad)}"
        log(f"store split {split}: {len(trk)} train / {len(vak)} val recordings, {hours.sum():.0f} train h in "
            f"{len(hours)} sources; PROTEUS streams {sorted(f'{u}|{s}' for u, s in self.proteus_streams)}")

        table = json.loads(ST_TABLE.read_text())
        low = {k.lower(): k for k in table}
        egi_all = json.loads(EGI_MAP.read_text())
        kw = dict(window_s=WIN_S, hop_s=hop_s, channels="all", target_sfreq=FS, masks=str(masks), index_workers=index_workers)
        self.ds = ContinuousWindowDataset(roots, keep_paths=keep, **kw)
        vds = ContinuousWindowDataset(roots, keep_paths=vkeep, **kw)
        self.batch, self.token_scale, self.hop_s = batch, token_scale, hop_s

        def plan(d, which):
            """per recording: (source, sorted ST names, stored-row order) or None; + the mapping tallies"""
            out, tally = [], {}
            for m in d._meta:
                s = pk_src.get(m["pk"])
                if s is None:
                    raise SystemExit(f"store pool: recording {m['pk']} ({which}) is in no split row (portable_key mismatch)")
                if pk_ds.get(m["pk"]) in ("tuh_abnormal", "tuh_events", "seed_vig"):
                    raise SystemExit(f"store pool: a non-redistributable source {pk_ds[m['pk']]} is in the pool ({m['pk']})")
                net = egi_net(m["sel_names"])
                t = tally.setdefault(s, dict(recordings=0, kept=0, rules={}, dropped_names=set()))
                t["recordings"] += 1
                pairs = {}
                for r, n in enumerate(m["sel_names"]):
                    st, rule = st_name(n, low, egi_all[net] if net else None)
                    if st is not None and st in pairs:
                        rule = "duplicate"                    # a second stored channel on the same ST name: first one kept
                    t["rules"][rule] = t["rules"].get(rule, 0) + 1
                    if st is None or rule == "duplicate":
                        t["dropped_names"].add(n)
                    else:
                        pairs[st] = r
                if len(pairs) < min_ch:
                    out.append(None)
                    continue
                t["kept"] += 1
                names = tuple(sorted(pairs))
                out.append((s, names, np.array([pairs[n] for n in names], np.intp)))
            return out, tally

        rec, tally = plan(self.ds, "train")
        vrec, _ = plan(vds, "val")
        idx = np.asarray(self.ds._index, np.int64).reshape(-1, 2)
        self.ds._index = []                               # the (ri, start) tuples -> numpy (fork-shared, no per-worker copy)
        self.rec_rows = [None if r is None else r[2] for r in rec]
        gkey = {}
        rec_g = np.full(len(rec), -1, np.int64)
        for ri, r in enumerate(rec):
            if r is not None:
                rec_g[ri] = gkey.setdefault((r[0], r[1]), len(gkey))
        self.groups = [dict(source=s, ch=list(n)) for (s, n) in gkey]
        wg = rec_g[idx[:, 0]]
        ok = wg >= 0
        self.win_ri, self.win_st = idx[ok, 0], idx[ok, 1]
        wg = wg[ok]
        order = np.argsort(wg, kind="stable")
        self.win_ri, self.win_st, wg = self.win_ri[order], self.win_st[order], wg[order]
        cuts = np.searchsorted(wg, np.arange(len(gkey) + 1))
        self.g_slice = [(int(cuts[g]), int(cuts[g + 1])) for g in range(len(gkey))]
        nwin = np.array([b - a for a, b in self.g_slice], float)
        for g, n in zip(self.groups, nwin):
            g["n_windows"] = int(n)
        sources = sorted({g["source"] for g in self.groups})
        s_n = {s: sum(n for g, n in zip(self.groups, nwin) if g["source"] == s) for s in sources}
        self.mb = [min(batch, max(8, round(batch * 41 / len(g["ch"])))) if token_scale else batch for g in self.groups]
        p_s = {s: np.sqrt(s_n[s]) for s in sources}
        tot = sum(p_s.values())
        p_s = {s: v / tot for s, v in p_s.items()}
        p_g = np.array([p_s.get(g["source"], 0.0) * n / max(s_n.get(g["source"], 1), 1) for g, n in zip(self.groups, nwin)])
        tok = np.array([b * len(g["ch"]) for g, b in zip(self.groups, self.mb)], float)   # channel-windows per draw (x 32 patches)
        sl = np.array([g["source"].split(":")[0] in SLEEP_PSG for g in self.groups])
        A, B = float((p_g * tok)[sl].sum()), float((p_g * tok)[~sl].sum())
        self.sleep_mult = 1.0
        if A > 0 and B > 0 and A / (A + B) > sleep_share:
            self.sleep_mult = sleep_share * B / ((1 - sleep_share) * A)
            p_g = np.where(sl, p_g * self.sleep_mult, p_g)
        self.p_draw = p_g / p_g.sum()
        self.sleep_token_share = float((self.p_draw * tok)[sl].sum() / (self.p_draw * tok).sum()) if sl.any() else 0.0
        self.token_share = {}                             # delivered channel-patch tokens per source (reported beside C's)
        for g, q, t in zip(self.groups, self.p_draw, tok):
            self.token_share[g["source"]] = self.token_share.get(g["source"], 0.0) + float(q * t)
        tt = sum(self.token_share.values()) or 1.0
        self.token_share = {k: v / tt for k, v in sorted(self.token_share.items(), key=lambda kv: -kv[1])}
        for g, q, b in zip(self.groups, self.p_draw, self.mb):
            g["p_draw"], g["micro_batch"] = float(q), int(b)

        # val: up to val_per_source windows of each source's tier-1 recordings, read ONCE (fixed rng), cached in RAM
        vidx = np.asarray(vds._index, np.int64).reshape(-1, 2)
        vds._index = []
        vr = np.random.default_rng(VAL_SEED)
        self.val = {}
        sx = {s: k for k, s in enumerate(sources)}
        vs = np.array([-1 if r is None else sx.get(r[0], -1) for r in vrec] or [-1], np.int64)
        vwin_src = vs[vidx[:, 0]] if len(vidx) else np.zeros(0, np.int64)
        for s in sources:
            cand = np.flatnonzero(vwin_src == sx[s])
            if not len(cand):
                continue
            pick = sorted(vr.choice(len(cand), size=min(val_per_source, len(cand)), replace=False))
            by_ch = {}
            for j in pick:
                ri, st = vidx[cand[j]]
                _, names, rows = vrec[ri]
                by_ch.setdefault(names, []).append(vds._raw_window(int(ri), int(st), rows).astype(np.float16))
            self.val[s] = [(list(n), np.stack(w)) for n, w in by_ch.items()]
        vds.close()

        shards = sorted({m["path"] for m in self.ds._meta})
        self.record = dict(
            split=str(split), split_sha256=sha256(split), roots=roots, snapshot=str(snap),
            snapshot_sync_version=(snap / "SYNC_VERSION").read_text().strip() if (snap / "SYNC_VERSION").exists() else None,
            masks=str(masks), masks_sha256=sha256(masks) if masks.exists() else None, egi_map_sha256=sha256(EGI_MAP),
            st_table_sha256=sha256(ST_TABLE), hop_s=hop_s, window_s=WIN_S, fs=FS, min_ch=min_ch, sleep_share_cap=sleep_share,
            sleep_mult=self.sleep_mult, sleep_token_share=self.sleep_token_share, token_share=self.token_share,
            families={k: v for k, v in FAMILY.items()}, n_groups=len(self.groups),
            n_train_windows=int(nwin.sum()), shards=[dict(path=p, size=Path(p).stat().st_size, mtime=Path(p).stat().st_mtime)
                                                     for p in shards],
            sources={s: dict(split_train_h=round(float(hours.get(s, 0.0)), 6),
                             indexed_h=round(float(s_n.get(s, 0.0) * hop_s / 3600), 6),
                             recordings=tally.get(s, {}).get("recordings", 0), recordings_kept=tally.get(s, {}).get("kept", 0),
                             channel_rules=tally.get(s, {}).get("rules", {}),
                             dropped_names=sorted(tally.get(s, {}).get("dropped_names", set()))[:40],
                             p_draw=float(sum(g["p_draw"] for g in self.groups if g["source"] == s)),
                             n_val=int(sum(len(w) for _, w in self.val.get(s, []))))
                     for s in sorted(set(hours.index) | set(tally))})
        missing = [s for s, v in self.record["sources"].items() if v["split_train_h"] > 0 and v["recordings"] == 0]
        if missing:                                       # a split source the loader never served: a store / key mismatch
            raise SystemExit(f"store pool: split sources with train hours but 0 loaded recordings: {missing}")
        for s, v in self.record["sources"].items():
            log(f"store source {s}: split {v['split_train_h']:.1f} h, indexed {v['indexed_h']:.1f} h, recordings "
                f"{v['recordings_kept']}/{v['recordings']} kept, rules {v['channel_rules']}, p_draw {v['p_draw']:.3f}, val {v['n_val']}")
        log("store token shares: " + ", ".join(f"{k} {v:.3f}" for k, v in list(self.token_share.items())[:12]))
        log(f"store pool: {len(self.groups)} groups, {int(nwin.sum())} train windows, sleep token share "
            f"{self.sleep_token_share:.3f} (mult {self.sleep_mult:.3f}), {len(shards)} shards, {time.time() - t0:.0f} s")

    def read(self, g, rng):
        """one micro-batch of group g: (mb, C, 512) float32 at 128 Hz, channels in self.groups[g]['ch'] order."""
        a, b = self.g_slice[g]
        pick = rng.integers(a, b, size=self.mb[g])
        pick = pick[np.argsort(self.win_ri[pick], kind="stable")]       # same-recording reads together
        return np.stack([self.ds._raw_window(int(self.win_ri[i]), int(self.win_st[i]), self.rec_rows[self.win_ri[i]])
                         for i in pick])

    def stream(self, seed, workers, prefetch=4):
        """an endless iterator of (group, x) micro-batches from `workers` DataLoader workers (one rng per worker)."""
        import torch
        pool = self

        class _It(torch.utils.data.IterableDataset):
            def __iter__(self):
                info = torch.utils.data.get_worker_info()
                rng = np.random.default_rng([*np.atleast_1d(seed).astype(np.int64).tolist(), info.id if info else 0])
                while True:
                    g = int(rng.choice(len(pool.groups), p=pool.p_draw))
                    yield g, torch.from_numpy(pool.read(g, rng))

        dl = torch.utils.data.DataLoader(_It(), batch_size=None, num_workers=workers, persistent_workers=workers > 0,
                                         prefetch_factor=prefetch if workers > 0 else None,
                                         multiprocessing_context="fork" if workers > 0 else None)
        return iter(dl)
