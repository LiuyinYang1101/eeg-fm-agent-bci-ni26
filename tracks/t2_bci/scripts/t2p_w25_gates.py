#!/usr/bin/env python
"""T2 W25-ALL6 promotion: the MECHANICAL gates (Track3 / the manager, pin 2), each on a fetched job dir with S1_DONE status=OK:
  timing: 3 reps; the registered D517 line is AFFINE: seconds = 62.0 x ms_per_window - 163 (DECISIONS: "affine 62.0 s/ms - 163 s, residuals
  ≈ ±1.3 %", the W18 gate's table; conservative: the W18 worker ran 12.4 % faster than it); MEAN <= 3,300 s AND every rep <= 3,400 s;
  K2: preds == the pkgcheck's (n_differ 0), 0 errors / kept-trained fallbacks in the side / FM / proto refits;
  pkgcheck: finite, every gate true (the W23-style 'gates' dict, or the W21lp pkgcheck's gate_* 'ok' fields + per-stream finite + 0 public
  fallbacks + cnn_lp active).
  python t2p_w25_gates.py <TIME_DIR> <KIT_DIR> <PKG_DIR>   -> prints the gate lines; exit 0 = PASS
"""
import glob, json, sys
td, kd, pd_ = sys.argv[1], sys.argv[2], sys.argv[3]
LINE_A, LINE_B = 62.0, -163.0                                  # the registered affine D517 line (s = 62.0 ms - 163)
t = [p for p in glob.glob(td + "/*.json") if not p.endswith((".kit_sim.json", ".kit_warmup.json"))]
assert len(t) == 1, ("timing json", t)
reps = [r["ms_per_window"] for r in json.load(open(t[0]))["reps"]]
secs = [round(LINE_A * m + LINE_B, 1) for m in reps]
mean_s = round(sum(secs) / len(secs), 1)
print(f"timing: {len(reps)} reps ms_per_window {reps} -> D517-line s {secs} (62.0 ms - 163); mean {mean_s} s (bar 3,300), max {max(secs)} s (bar 3,400)")
kf = glob.glob(kd + "/*.kitfull.json"); assert len(kf) == 1, kf
K = json.load(open(kf[0]))
kok = K["preds_equal_pkgcheck"] and K["n_differ"] == 0 and all(K[s].get("errors", 0) == 0 and K[s].get("kept_trained", 0) == 0 for s in ("refit_stats", "fm_refit_stats", "fm_proto_stats"))
print(f"K2: preds == pkgcheck {K['preds_equal_pkgcheck']}, n_differ {K['n_differ']}, cell BA {K['cell_mean_ba']:.4f}, refits {K['refit_stats']}, fm {K['fm_refit_stats']}, proto {K['fm_proto_stats']} -> {kok}")
pf = glob.glob(pd_ + "/*.pkgcheck.json"); assert len(pf) == 1, pf
Pk = json.load(open(pf[0]))
if "gates" in Pk:                                              # the W23-style pkgcheck (t2p_w23_pkgcheck): finite + a gates dict
    pok = Pk["finite"] is True and all(Pk["gates"].values())
    print(f"pkgcheck: finite {Pk['finite']}, gates {Pk['gates']} -> {pok}")
else:                                                          # the W21lp pkgcheck (t2p_w21lp_pkgcheck; the W22 / W25 chains' step 4)
    oks = {g: Pk[g]["ok"] for g in ("gate_IL", "gate_B", "gate_S", "gate_FM")}
    oks.update({f"{g}[{st}]": v["ok"] for g in ("gate_H", "gate_HF", "gate_HP") for st, v in Pk[g].items()})
    fin = {st: Pk[st]["finite"] for st in Pk if "|" in st}
    fb = {k: (Pk[k]["errors"], Pk[k]["kept_trained"] if k != "cnn_lp_stats_public" else 0)   # Track3: counters MANDATORY (a missing key fails closed;
          for k in ("refit_stats_public", "fm_refit_stats_public", "fm_proto_stats_public", "cnn_lp_stats_public")}   # cnn_lp has no kept_trained field)
    lp_on = Pk["cnn_lp_stats_public"].get("smoothed", 0) > 0
    pok = bool(oks) and all(oks.values()) and bool(fin) and all(fin.values()) and all(e == 0 and k_ == 0 for e, k_ in fb.values()) and lp_on
    print(f"pkgcheck (w21lp): gates {oks}; finite {fin}; (errors, kept_trained) {fb}; cnn_lp smoothed {Pk['cnn_lp_stats_public'].get('smoothed')} -> {pok}")
ok = len(reps) == 3 and mean_s <= 3300 and max(secs) <= 3400 and kok and pok
print("GATES " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
