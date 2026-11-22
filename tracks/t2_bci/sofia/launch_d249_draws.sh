#!/usr/bin/env bash
# T2-D249 (GATED; do not run before the gates): the vp2 last-five draws with every adopted re-tune INCLUDING the adopted ST-L fine-tune
# (T2-D247 / D247r: --fm-llrd 0.75 --fm-rc 0.75) = launch_d217_draws.sh with launch_d248.sh in place of launch_d232.sh: each draw retrains its own
# ST-L (and MOR) at its D205 draw's FM seed 20k and keeps them (the plain repack; no --fm-from donor, whose route needs every FM member config to match),
# instead of taking its D210 vp draw's FM files. CNN recipes as T2-D232 / D248 (FBCNet r1_pulxx_vp2_e180_vr_rc75, ShallowNet r1_pulxx_vp2_e180_vr).
# Same builds as launch_d217_draws.sh (those also train an ST-L, then swap in the donor's), so the same cost (≈ 0.7 GPU-h per draw, T2-D248).
# Use this one iff upload 5 = the D248 candidate 5f538614 (the adopted ST-L); if the user keeps 5816509e, launch_d217_draws.sh applies.
# Gates (PLAN §6b item 8, user 2026-10-23 16:57 / 23:19): upload 5 keeps the re-tunes (`t2_official_read.py guard5`) AND the 3-way rule chose the
# vp2 set or the mixed case. <n seeds> = 15 iff upload 4 completed on the worker with `[ni26] cuda graph main: used`, else 10.
# <set>: vp2 = draws k = 1-4; mixed = draws k = 1-2. vp2 ZIP 1 with the re-tunes = upload 5's 5f538614 (T2-D248) at 10 seeds; at 15 seeds the
# line is rebuilt too (seeds 0-14, FM seed 0).
#   bash tracks/t2_bci/sofia/launch_d249_draws.sh GO <10|15> <vp2|mixed>
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ "${1:-}" = GO ] || { echo "gated: pass GO once upload 5's guard and the 3-way rule allow it"; exit 2; }
n=${2:?10 or 15}; [ "$n" = 10 ] || [ "$n" = 15 ] || exit 2
case ${3:?vp2 or mixed} in vp2) ks="1 2 3 4";; mixed) ks="1 2";; *) exit 2;; esac
if [ "$n" = 15 ]; then
  ROW=d249 bash tracks/t2_bci/sofia/launch_d248.sh stline15 "$(seq -s, 0 14)" 0 all r1_pulxx_vp2_e180_vr_rc75 r1_pulxx_vp2_e180_vr 180 0.75 0.75
fi
for k in $ks; do
  b=$((20 * k)); S=$(seq -s, $b $((b + n - 1)))
  ROW=d249 bash tracks/t2_bci/sofia/launch_d248.sh std${k}n$n "$S" $b all r1_pulxx_vp2_e180_vr_rc75 r1_pulxx_vp2_e180_vr 180 0.75 0.75
done
