#!/usr/bin/env bash
# T2-D561 STAGE 1 read (registered 2026-11-17 before any number; INSIGHT ONLY). The arms routed as shipped (t2p_fpcalib blend, tv8c; FBCNet
# r 1/3 tag 3 = fb3, ShallowNet / IFNet r 0.5 tag 2 = fb2), then D413c's ALONE read exactly: one family alone, KIT protocol test cells,
# causal SK-lambda (10 it, n0 3, lambda 0.95), cell-mean BA, per seed 340-345, PER FAMILY (FBCNet tilt; ShallowNet and IFNet at align none)
# and POOLED (the per-seed deltas averaged over the three families): d20 = mf20 - control, d50 = mf50 - control, d2050 = mf20 - mf50
# (seen1 the same, descriptive).
# STAGE 2 TRIGGER (the manager's, amended before any number): in >= 2 of 3 families, mf20 > control AND mf20 > mf50 on >= 5 / 6 kit seeds.
# ST-L stays out (no synthetic path in t2p_fm; post-competition).
#   mem-guard bash scripts/local/t2_d561_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d561/read
mkdir -p "$R"
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@" > /dev/null; }
one() { local d=( "$S"/ni26-t2p-d561-$1-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need one dir for d561-$1: ${d[*]}" >&2; exit 2; }; echo "${d[0]}runs"; }
SD=340,341,342,343,344,345
for p in kit seen1; do
  for a in mf20:fbc_tl15mf20_s340:r1_pulxx_vp2_e180_vr_rc75_tl15_mf20 mf50:fbc_tl15mf_s340:r1_pulxx_vp2_e180_vr_rc75_tl15_mf50 ctl:fbc_tl15_s340:r1_pulxx_vp2_e180_vr_rc75_tl15; do
    IFS=: read -r k arm rec <<< "$a"
    blend --runs "$(one $arm)" --protocols $p --out "$R/route_fbc_$k" --model fbcnet_bboth --align none --recipe "${rec}tv8c" --seeds $SD
  done
  for fam in sh:shallow ifn:ifnet; do
    IFS=: read -r f mod <<< "$fam"
    for a in mf20:${f}_mf20_s340:r1_pulx_vp2_e180_vr_mf20 mf50:${f}_mf50_s340:r1_pulx_vp2_e180_vr_mf50 ctl:${f}_ctl_s340:r1_pulx_vp2_e180_vr; do
      IFS=: read -r k arm rec <<< "$a"
      blend --runs "$(one $arm)" --protocols $p --out "$R/route_${f}_$k" --model $mod --align none --recipe "${rec}tv8c" --seeds $SD --r 0.5 --tag 2
    done
  done
done
$PY - "$O" "$E" <<'PYEOF'
import json, sys
import numpy as np, pandas as pd
O, E = sys.argv[1], sys.argv[2]; R = f"{O}/t2_d561/read"
sys.path.insert(0, "tracks/t2_bci/scripts")
from t2p_report import causal_sinkhorn
ix_ = pd.read_csv(f"{E}/index.csv"); strm, ordr, y, cell = ix_.stream.to_numpy(), ix_.order.to_numpy(), ix_.y.to_numpy(), ix_.cell.to_numpy()
FAM = {"fbc": ("fbcnet_bboth", {"mf20": "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20", "mf50": "r1_pulxx_vp2_e180_vr_rc75_tl15_mf50", "ctl": "r1_pulxx_vp2_e180_vr_rc75_tl15"}, "fb3"),
       "sh": ("shallow", {"mf20": "r1_pulx_vp2_e180_vr_mf20", "mf50": "r1_pulx_vp2_e180_vr_mf50", "ctl": "r1_pulx_vp2_e180_vr"}, "fb2"),
       "ifn": ("ifnet", {"mf20": "r1_pulx_vp2_e180_vr_mf20", "mf50": "r1_pulx_vp2_e180_vr_mf50", "ctl": "r1_pulx_vp2_e180_vr"}, "fb2")}


def cba(f):
    z = np.load(f); ix = z["test_idx"]; P = causal_sinkhorn(z["prob_last"].astype(np.float64), (strm[ix], ordr[ix]), 10, 3, 0.95); pr = P.argmax(1)
    return np.mean([np.mean([np.mean(pr[(cell[ix] == c) & (y[ix] == k)] == k) for k in np.unique(y[ix][cell[ix] == c])]) for c in np.unique(cell[ix])])


def dd(v):
    return dict(mean=round(float(v.mean()), 4), pos=int((v > 0).sum()), per_seed=np.round(v, 4).tolist())


res, D = {}, {}
for fam, (mod, rec, tg) in FAM.items():
    for p in ("kit", "seen1"):
        L = {a: np.array([cba(f"{R}/route_{fam}_{a}/{p}__{mod}__none__n00__{rec[a]}tv8c{tg}__s{s}.npz") for s in range(340, 346)]) for a in rec}
        D[(fam, p)] = (L["mf20"] - L["ctl"], L["mf50"] - L["ctl"], L["mf20"] - L["mf50"])
        res[f"{fam}/{p}"] = dict(level={a: round(float(v.mean()), 4) for a, v in L.items()}, d20=dd(D[(fam, p)][0]), d50=dd(D[(fam, p)][1]),
                                 d2050=dd(D[(fam, p)][2]))
        print(f"D561 {fam} {p}: {res[f'{fam}/{p}']}", flush=True)
for p in ("kit", "seen1"):                                # pooled: the per-seed deltas averaged over the three families
    res[f"pooled/{p}"] = {k: dd(np.mean([D[(f, p)][i] for f in FAM], 0)) for i, k in enumerate(("d20", "d50", "d2050"))}
    print(f"D561 pooled {p}: {res[f'pooled/{p}']}", flush=True)
fam_ok = {f: bool(res[f"{f}/kit"]["d20"]["pos"] >= 5 and res[f"{f}/kit"]["d2050"]["pos"] >= 5) for f in FAM}
res["family_trigger"] = fam_ok
res["stage2_trigger"] = bool(sum(fam_ok.values()) >= 2)
print(f"D561 STAGE 1 verdict: {'STAGE 2 TRIGGERED' if res['stage2_trigger'] else 'stage 2 NOT triggered'} (kit, per family mf20 > control "
      f"AND > mf50 on >= 5 / 6 seeds: {fam_ok}; >= 2 of 3 families needed). ST-L stays out (no synthetic path).", flush=True)
json.dump(res, open(f"{R}/d561_read.json", "w"), indent=1)
PYEOF
