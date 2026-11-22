#!/usr/bin/env python
"""T2-D654 (W31v) promotion: the MECHANICAL gates of the view package, each on a fetched job dir with S1_DONE status=OK (Track3's ask: the owner's
D654 timing bar as an explicit gate line, as t2p_w25_gates does for W25's 3,300 / 3,400):
  timing: 3 reps on the registered affine D517 line (s = 62.0 ms - 163); THE OWNER'S D654 BAR (16:4x, in writing): MEAN <= 3,480 s AND every rep
  <= 3,550 s (W25's own 3,300 / 3,400 reported beside it);
  K2: the kit harness's predictions == the build's saved predictions (n_differ 0), 0 errors / kept-trained fallbacks in the side / FM / proto refits;
  build: the golden's view checks (a) spec bitwise, (b) view batch bitwise, (c) fmK graph vs the eager reference <= 5e-3; the realised view fraction
  within the token bucket (gated <= q t + 1 per stream); the build job itself ran t2p_w31v_patch.py check (a failure stops it before the golden).
  python t2p_w31v_gates.py <TIME_DIR> <KIT_DIR> <BUILD_DIR>   -> prints the gate lines; exit 0 = PASS
"""
import glob
import json
import sys

td, kd, bd = sys.argv[1], sys.argv[2], sys.argv[3]
LINE_A, LINE_B = 62.0, -163.0
MEAN_BAR, REP_BAR = 3480.0, 3550.0                            # the owner's D654 bar (2026-11-20 16:4x, in writing)
t = [p for p in glob.glob(td + "/*.json") if not p.endswith((".kit_sim.json", ".kit_warmup.json"))]
assert len(t) == 1, ("timing json", t)
reps = [r["ms_per_window"] for r in json.load(open(t[0]))["reps"]]
secs = [round(LINE_A * m + LINE_B, 1) for m in reps]
mean_s = round(sum(secs) / len(secs), 1)
tok = len(reps) == 3 and mean_s <= MEAN_BAR and max(secs) <= REP_BAR
print(f"timing: {len(reps)} reps ms_per_window {reps} -> D517-line s {secs}; mean {mean_s} s (the owner's D654 bar {MEAN_BAR:,.0f}), max {max(secs)} s "
      f"(bar {REP_BAR:,.0f}) -> {tok}  [W25's own bars 3,300 / 3,400: {mean_s <= 3300 and max(secs) <= 3400}]")
kf = glob.glob(kd + "/*.kitfull.json"); assert len(kf) == 1, kf
K = json.load(open(kf[0]))
kok = K["preds_equal_pkgcheck"] and K["n_differ"] == 0 and all(K[s].get("errors", 0) == 0 and K[s].get("kept_trained", 0) == 0
                                                               for s in ("refit_stats", "fm_refit_stats", "fm_proto_stats"))
print(f"K2: preds == the build's {K['preds_equal_pkgcheck']}, n_differ {K['n_differ']}, cell BA {K['cell_mean_ba']:.4f}, refits {K['refit_stats']}, "
      f"fm {K['fm_refit_stats']}, proto {K['fm_proto_stats']} -> {kok}")
gf = glob.glob(bd + "/*.golden.json"); assert len(gf) == 1, gf
B = json.load(open(gf[0]))["packages"]["B"]
vc, vf = B.get("view_checks") or {}, B.get("view_frac") or {}
bok = bool(vc.get("a_spec_bitwise")) and bool(vc.get("b_view_batch_bitwise")) and float(vc.get("c_fmk_vs_eager_max_abs", 1.0)) <= 5e-3 and bool(vf.get("bucket_ok"))
print(f"build: view checks {vc}; view fraction {vf}; differ vs W25-ALL6 {B.get('differ_vs_base')} (reported) -> {bok}")
ok = tok and kok and bok
print("GATES " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
