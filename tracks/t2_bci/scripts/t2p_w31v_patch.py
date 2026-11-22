#!/usr/bin/env python
"""T2-D654 (FM view-TTA) the SPLIT-GRAPH template = the W21lp template + cfg["fm_views"] (absent = the W21lp code path, BITWISE: every edit is
behind `self.views is not None`). With fm_views the graph path runs, per window:
  (1) a CNN graph ('cnn': the main / cnn2 / cnn3 pooled families, as in 'main'), then the host reads the gate margin = top1 - top2 of the CNN
      blend with W21's family weights renormalised over the CNN share ((1 - fm - w2 - w3) : w2 : w3) = viewmix's 'c' gate;
  (2) the causal gate (viewmix.causal_gate, per stream: gated iff m_t <= quantile_q(m_0..m_t) AND n_gated + 1 <= q (t + 1) + 1; reset_state
      clears it), or "gate": "off" = never gated (the split graphs only: the gate-OFF golden check vs W25);
  (3) FM@1 ('fm1': the six members at B = 1, as in 'main') or, on a gated window, FM@K ('fmK': per member ONE forward of [identity, view 1 ..
      K - 1] (t2p_fm.fm_view_spec: gain exp(U(+-0.2)) from torch.Generator(65400 + v), then the circular shift 15 / -15 / 30 at 120 Hz, on the
      aligned FM input), the pooled features averaged over the K views, then the member's head: net.out(F) under the same autocast, so the
      fm_proto / cnn_lp pre-hook on net.out records the FINAL (view-mean) feature = the consistency pin). Side / DC vectors: the identity window's.
  cfg["fm_views"] = {"K": 2, "q": 0.1, "gate": "cnn" | "prob" | "off"}; refused with routed (st_lora) members or a MOR member.
  python t2p_w31v_patch.py template                      -> tracks/t2_bci/submission/t2p_submission_w31v.py (from t2p_submission_w21lp.py)
  python t2p_w31v_patch.py zip IN.zip OUT.zip '<fm_views JSON>'   -> OUT.zip = IN.zip with its submission.py patched + config fm_views
  python t2p_w31v_patch.py check BASE.zip CAND.zip                -> the structural check (exit 0 = PASS)
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EDITS = [
    ("class StreamModel(torch.nn.Module):\n",
     "def _d654_gate_step(st, m, q, gate):\n"
     "    \"\"\"T2-D654: one window of the per-stream causal gate (st = dict(m, n, g, gen), cleared in reset_state) -> True iff this window gets the views.\n"
     "    'cnn' = viewmix.causal_gate (m_t <= quantile_q(m_0..m_t), linear); 'prob' = viewmix.prob_gate (ONE draw per window from the stream's\n"
     "    Generator(65460), r_t = #{s < t: m_s < m_t} / t, p_t = min(1, 2q (1 - r_t)), viewed iff u_t < p_t); both behind the token bucket\n"
     "    n_gated + 1 <= q (t + 1) + 1; 'off' = never (the split graphs only).\"\"\"\n"
     "    t = st[\"n\"]\n"
     "    st[\"n\"] += 1\n"
     "    if gate == \"off\":\n"
     "        return False\n"
     "    if gate == \"prob\":\n"
     "        u = float(torch.rand(1, generator=st[\"gen\"])[0])\n"
     "        rk = float(np.sum(np.asarray(st[\"m\"], np.float64) < m)) / t if t > 0 else 0.0\n"
     "        st[\"m\"].append(m)\n"
     "        ok = u < min(1.0, 2.0 * q * (1.0 - rk))\n"
     "    else:\n"
     "        st[\"m\"].append(m)\n"
     "        ok = m <= float(np.quantile(np.asarray(st[\"m\"], np.float64), q))\n"
     "    if ok and st[\"g\"] + 1 <= q * (t + 1) + 1:\n"
     "        st[\"g\"] += 1\n"
     "        return True\n"
     "    return False\n"
     "\n"
     "\n"
     "class StreamModel(torch.nn.Module):\n"),
    ("        self.gshared = {\"g\": {}}\n",
     "        self.gshared = {\"g\": {}}\n"
     "        # T2-D654 (W31v): FM view-TTA through split CUDA graphs (cnn -> the host gate -> fm1 | fmK); None = the W21lp path (bitwise)\n"
     "        self.views = cfg.get(\"fm_views\")\n"
     "        if self.views is not None:\n"
     "            vk, vq, vg = int(self.views[\"K\"]), float(self.views[\"q\"]), str(self.views.get(\"gate\", \"cnn\"))\n"
     "            assert 2 <= vk <= 4 and 0.0 <= vq <= 1.0 and vg in (\"cnn\", \"prob\", \"off\"), self.views\n"
     "            assert len(self.fm_nets) and not any(isinstance(n, STRouted) for n in self.fm_nets) and \\\n"
     "                not any(m.get(\"family\") == \"mor\" for m in cfg.get(\"fm_members\", [])), \"D654: views need plain ST members\"\n"
     "            c_ = len(self.idx)\n"
     "            self.view_spec = []\n"
     "            for v in range(1, vk):                       # = t2p_fm.fm_view_spec (VIEW_SEED0 65400, VIEW_GAIN 0.2, VIEW_SHIFTS 15 / -15 / 30)\n"
     "                g_ = torch.Generator().manual_seed(65400 + v)\n"
     "                self.view_spec.append((torch.exp((torch.rand(c_, generator=g_) * 2 - 1) * 0.2), (15, -15, 30)[v - 1]))\n"
     "            self.views = dict(K=vk, q=vq, gate=vg)\n"),
    ("    def reset_state(self):\n        self.aligner.reset()\n",
     "    def reset_state(self):\n        self.aligner.reset()\n"
     "        if getattr(self, \"views\", None) is not None:     # T2-D654: the per-stream causal gate state\n"
     "            self.vg = dict(m=[], n=0, g=0, gen=torch.Generator().manual_seed(65460))   # 'prob': re-seeded per stream\n"),
    ("    def _capture(self, fn, zs, slot):\n",
     "    def _gpu_cnn(self, zs, cache=True):\n"
     "        \"\"\"T2-D654: = _gpu_eager's CNN part (the main / cnn2 / cnn3 pooled families) -> their probability tensors.\"\"\"\n"
     "        out = [self._fam_t(self.nets, self.stk[0], self._v(zs[0]))]\n"
     "        if len(self.nets2):\n"
     "            out.append(self._fam_t(self.nets2, self.stk[1], self._v(zs[1])))\n"
     "        if len(self.nets3):\n"
     "            out.append(self._fam_t(self.nets3, self.stk[2], self._v(zs[2])))\n"
     "        return out\n"
     "\n"
     "    def _gpu_fm1(self, zs, cache=True):\n"
     "        \"\"\"T2-D654: = _gpu_eager's FM part at B = 1 (zs = the members' aligned inputs).\"\"\"\n"
     "        out = []\n"
     "        for net, zf in zip(self.fm_nets, zs):\n"
     "            with torch.autocast(\"cuda\", dtype=torch.bfloat16, enabled=zf.is_cuda, cache_enabled=cache):\n"
     "                out.append(F.softmax(net(zf).float(), dim=1)[0])\n"
     "        return out\n"
     "\n"
     "    def _gpu_fmk(self, zs, cache=True):\n"
     "        \"\"\"T2-D654: the FM members on [identity, view 1 .. K - 1] in ONE forward each (zs = the members' (K, C, T) batches): the pooled\n"
     "        features averaged over the K views, then the member's head (net.out, under the same autocast: its fm_proto pre-hook records the\n"
     "        view-mean feature, the window's FINAL feature).\"\"\"\n"
     "        out = []\n"
     "        for net, zk in zip(self.fm_nets, zs):\n"
     "            with torch.autocast(\"cuda\", dtype=torch.bfloat16, enabled=zk.is_cuda, cache_enabled=cache):\n"
     "                f_ = net.features(zk).float().mean(0, keepdim=True)\n"
     "                out.append(F.softmax(net.out(f_).float(), dim=1)[0])\n"
     "        return out\n"
     "\n"
     "    def _view_batch(self, zf):\n"
     "        \"\"\"T2-D654: (1, C, T) aligned FM input -> (K, C, T) = [identity, the fixed views] (t2p_fm.fm_apply_view: gain, then the roll).\"\"\"\n"
     "        L_ = zf.shape[-1]\n"
     "        vs = [zf]\n"
     "        for gain, shift in self.view_spec:\n"
     "            t_ = (torch.arange(L_, device=zf.device) - int(shift)) % L_\n"
     "            vs.append((zf * gain.to(device=zf.device, dtype=zf.dtype).view(1, -1, 1))[..., t_])\n"
     "        return torch.cat(vs, 0)\n"
     "\n"
     "    def _view_gate(self, pc):\n"
     "        \"\"\"T2-D654: this window's gate decision from the CNN families' probabilities pc (numpy, main / cnn2 / cnn3): the margin = top1 - top2\n"
     "        of their blend at W21's weights renormalised over the CNN share (viewmix's 'c' margin), then _d654_gate_step (cnn / prob / off).\"\"\"\n"
     "        w_ = [1.0 - self.fm_weight - self.w2 - self.w3] + ([self.w2] if len(self.nets2) else []) + ([self.w3] if len(self.nets3) else [])\n"
     "        p_ = sum(wi * np.asarray(pi, np.float64) / max(float(np.sum(pi)), 1e-30) for wi, pi in zip(w_, pc)) / sum(w_)\n"
     "        s_ = np.sort(p_)\n"
     "        return _d654_gate_step(self.vg, float(s_[-1] - s_[-2]), self.views[\"q\"], self.views[\"gate\"])\n"
     "\n"
     "    def _capture(self, fn, zs, slot):\n"),
    ("        outs = self._run(self._gpu_eager, [z, z2] + z3 + zfs + rd, \"main\")\n",
     "        if self.views is None:\n"
     "            outs = self._run(self._gpu_eager, [z, z2] + z3 + zfs + rd, \"main\")\n"
     "        else:                                            # T2-D654: cnn graph -> the host gate -> fm1 | fmK\n"
     "            assert not rd, \"D654: no routed members\"\n"
     "            oc = self._run(self._gpu_cnn, [z, z2] + z3, \"cnn\")\n"
     "            if self._view_gate([q.cpu().numpy() for q in oc]):\n"
     "                outs = oc + self._run(self._gpu_fmk, [self._view_batch(zf) for zf in zfs], \"fmK\")\n"
     "            else:\n"
     "                outs = oc + self._run(self._gpu_fm1, zfs, \"fm1\")\n"),
]


def patch(text):
    for old, new in EDITS:
        assert text.count(old) == 1, ("anchor", old[:60], text.count(old))
        text = text.replace(old, new)
    return text


def main():
    a = sys.argv[1:]
    if a[0] == "template":
        src = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"
        out = ROOT / "tracks/t2_bci/submission/t2p_submission_w31v.py"
        out.write_text(patch(src.read_text()))
        print(f"wrote {out} from {src.name}")
        return
    if a[0] == "check":                              # the structural check: CAND == BASE except submission.py (== patch(BASE's)) and config fm_views
        zb, zc = zipfile.ZipFile(a[1]), zipfile.ZipFile(a[2])
        nb, nc = set(zb.namelist()), set(zc.namelist())
        assert nb == nc, ("file sets differ", sorted(nb ^ nc))
        sh = lambda z, n: hashlib.sha256(z.read(n)).hexdigest()   # noqa: E731
        diff = sorted(n for n in nb if n not in ("submission.py", "config.json") and sh(zb, n) != sh(zc, n))
        assert not diff, ("non-template files differ", diff)
        assert zc.read("submission.py").decode() == patch(zb.read("submission.py").decode()), "submission.py != patch(base)"
        cb, cc = json.loads(zb.read("config.json")), json.loads(zc.read("config.json"))
        v = cc.pop("fm_views")
        assert "fm_views" not in cb and cb == cc, "config differs outside fm_views"
        print(f"STRUCT PASS: {len(nb) - 2} files byte-identical, submission.py == patch(base), config == base + fm_views {v}")
        return
    assert a[0] == "zip" and len(a) == 4, __doc__
    zin, zout, views = a[1], a[2], json.loads(a[3])
    assert views.get("gate", "cnn") in ("cnn", "prob", "off") and 2 <= int(views["K"]) <= 4
    with zipfile.ZipFile(zin) as zi, zipfile.ZipFile(zout, "w", zipfile.ZIP_DEFLATED) as zo:
        cfg = json.loads(zi.read("config.json"))
        assert "fm_views" not in cfg and cfg.get("gpu_graph"), "D654: a W25-style (gpu_graph) package without views"
        cfg["fm_views"] = views
        for info in zi.infolist():
            if info.filename == "submission.py":
                zo.writestr(info, patch(zi.read(info).decode()))
            elif info.filename == "config.json":
                zo.writestr(info, json.dumps(cfg, indent=1))
            else:
                with zi.open(info) as src, zo.open(zipfile.ZipInfo(info.filename, date_time=info.date_time), "w") as dst:
                    for blk in iter(lambda: src.read(1 << 24), b""):
                        dst.write(blk)
    h = hashlib.sha256(Path(zout).read_bytes()).hexdigest()
    Path(zout + ".sha256").write_text(f"{h}  {Path(zout).name}\n")
    print(f"wrote {zout} (fm_views {views}) sha256 {h}")


if __name__ == "__main__":
    main()
