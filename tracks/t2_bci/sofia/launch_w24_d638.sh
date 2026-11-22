#!/usr/bin/env bash
# T2-D638 -> W24' (PREPARED; launched ONLY after rule (b) passes, on the manager's GO): the base (ROUTER=w21: W21; ROUTER=w23: the gated W23
# PREBUILD, i.e. rule (b)'s router) with its CNN members replaced by user-balanced ones. Every job on sofia (W21's training env, torch 2.9.1).
#   PREBUILD (the default after a rule (b) pass: no owner line exists for W24' yet): N / zip / jobs carry _PREBUILD, S1_PURPOSE 'not approved';
#   GO needs W24_APPROVED = the owner's typed W24' line (naming W24 and D638).
# Chain (= W23's partitions / walltimes; zen5_vis = 1 GPU per user, the three zen5_vis steps chain by afterok):
#   (all five submitted at once, chained by afterok; the build records the donor zip's sha)
#   (1) the donor: t2p_final.slurm (t2p_final CNN-only, W21's CNN specs: fbc / sh / ifn x seeds 200-214, the copies at r 0.3333 / 0.5 / 0.5 with
#       W21's pu recipes, --train all, --user-balance; template W11 = three families; its own golden + kit replay) [zen5_vis, 2 h]
#   (2) t2p_w24_build.slurm: swap -> inert repack (W21's template) -> K1 [zen5_vis, 2 h]
#   (3) t2p_pkgcheck.slurm w24 (t2p_w24_pkgcheck.py vs the base) [zen5_vis, 2 h] -> (4) K2 [zen5_vis, 2 h]; (5) the H200 timing afterok (2) [1 h]
#   DONOR_LOCAL=<outputs/t2_w24_donor/t2p_w24donor_PREBUILD.zip> (the manager's speed option): skip (1); `s1 put` the zip + .sha256 to
#   runs/ni26/ni26-t2p-pkg-<DN>-local/; (2) checks the PINNED sha first. GO / PREBUILD refuse unless outputs/t2_d638/result.json's rule (b)
#   picked c<ROUTER>.
#   ROUTER=w21|w23 [DONOR_LOCAL=...] [W24_APPROVED="..."] PREBUILD_CHOICE_AT="YYYY-MM-DD HH:MM" mem-guard bash tracks/t2_bci/sofia/launch_w24_d638.sh dry|PREBUILDdry|PREBUILD|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILDdry|PREBUILD|GO}; [[ "$mode" =~ ^(dry|PREBUILDdry|PREBUILD|GO)$ ]] || exit 2
pb=0; [[ "$mode" == PREBUILD* ]] && pb=1
case "${ROUTER:-}" in w21|w23) ;; *) echo "ROUTER=w21|w23 (rule (b)'s router)"; exit 2 ;; esac
if [ $pb = 1 ]; then
  [[ "${PREBUILD_CHOICE_AT:-}" =~ ^2026-11-[0-9]{2}\ [0-9]{2}:[0-9x]{2}$ ]] || { echo "set PREBUILD_CHOICE_AT='YYYY-MM-DD HH:MM' (the manager's GO after rule (b))"; exit 3; }
  [ -z "${W24_APPROVED:-}" ] || { echo "W24_APPROVED is set: an approved build is GO, not PREBUILD"; exit 3; }
  [ "$mode" == PREBUILD ] && mode=GO || mode=dry
fi
if [ "$mode" == GO ] && [ $pb = 0 ]; then
  [[ "${W24_APPROVED:-}" == *"W24"* && "${W24_APPROVED:-}" == *"D638"* ]] || { echo "set W24_APPROVED to the owner's typed W24' line (naming W24 and D638)"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/t2p_w24_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_w24_d638.sh)" ] || { echo "commit first"; exit 2; }
if [ "$mode" == GO ]; then                       # rule (b) must have PASSED and picked THIS router (outputs/t2_d638/result.json, the reader's verdict)
  b_=$(python3 -c "import json; v = json.load(open('$HOME/workspace/outputs/neural_interface_challenge/t2_d638/result.json'))['rules']['verdict']; print(v['b_best'] or '')" 2>/dev/null || true)
  [ -n "$b_" ] || { echo "rule (b) has not passed (outputs/t2_d638/result.json: no b_best)"; exit 3; }
  [ "$b_" = "c$ROUTER" ] || { echo "rule (b)'s best cell is $b_, not c$ROUTER: ROUTER must be ${b_#c}"; exit 3; }
fi
if [ "$ROUTER" = w21 ]; then
  BASE_REL=ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip; BASE_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc
else
  BASE_REL=ni26-t2p-w23d632-t2p_fin2_w23d632_PREBUILD-1643078/t2p_fin2_w23d632_PREBUILD.zip; BASE_SHA=28c32c5d73af70ce8627872da54858b81cce9e4bfc16c08a27a3eece18764873
fi
LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/$BASE_REL
h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of the base ($ROUTER): $h"; [ "$h" = "$BASE_SHA" ] || { echo "!! the local base sha256 != $BASE_SHA"; exit 4; }
R=$S1_ROOT/runs/ni26
SEEDS=200,201,202,203,204,205,206,207,208,209,210,211,212,213,214
N=t2p_fin2_w24${ROUTER}d638; DN=t2p_w24donor
P="T2-D638 W24' = the $ROUTER base + user-balanced CNN members (the owner's typed W24' line) for gating"
PBX=""
if [ $pb = 1 ]; then
  N=${N}_PREBUILD; DN=${DN}_PREBUILD
  P="T2-D638 W24' PREBUILD = the $ROUTER base + user-balanced CNN members, after rule (b) passed (the manager's GO, $PREBUILD_CHOICE_AT); not approved"
  PBX=",W24_PREBUILD=1"
fi
V=(--partition=zen5_vis --account=vsc)
VH=(--partition=zen4_h200)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
DONOR_ARGS=(--model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75 --epochs 180 --seeds $SEEDS --train all --user-blend 0.3333
  --model2 shallow --align2 ea --recipe2 r1_pulxx_vp2_e180_vr --epochs2 180 --seeds2 $SEEDS --weight2 0.23571428571428574 --user-blend2 0.5 --recipe2-pu r1_pulx
  --model3 ifnet --align3 ea --recipe3 r1_pulx_vp2_e180_vr --epochs3 180 --seeds3 $SEEDS --weight3 0.1 --user-blend3 0.5 --recipe3-pu r1_pulx_vp2_e180_vr
  --template tracks/t2_bci/submission/t2p_submission_w11.py --user-balance)
echo "### base $BASE_REL (sha $BASE_SHA) -> $N; donor $DN"
if [ -n "${DONOR_LOCAL:-}" ]; then                   # the manager's speed option: the donor built on the LOCAL 5090 (scripts/local/t2_w24_donor_local.sh)
  [ -f "$DONOR_LOCAL" ] && [ -f "$DONOR_LOCAL.sha256" ] && [ "$(basename "$DONOR_LOCAL")" = "$DN.zip" ] || { echo "DONOR_LOCAL must be the local $DN.zip with its .sha256"; exit 2; }
  DSHA=$(sha256sum "$DONOR_LOCAL" | cut -d' ' -f1); [ "$DSHA" = "$(cut -d' ' -f1 "$DONOR_LOCAL.sha256")" ] || { echo "!! the local donor's sha256 != its .sha256"; exit 4; }
  D_DIR="ni26-t2p-pkg-$DN-local"; D_REL="$D_DIR/$DN.zip"; J1=local; DEP=()
  echo "### the LOCAL donor $DONOR_LOCAL sha256 $DSHA -> runs/ni26/$D_REL (pinned for (2))"
  if [ "$mode" == GO ]; then
    s1 put "$DONOR_LOCAL" "runs/ni26/$D_DIR/" && s1 put "$DONOR_LOCAL.sha256" "runs/ni26/$D_DIR/" || { echo "!! s1 put failed"; exit 4; }
  fi
  DSE=$DSHA
else
  J1=$(sub S1_PURPOSE="$P: (1) the user-balanced CNN donor (t2p_final, zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm "${V[@]}" --time=02:00:00 \
    --job-name="ni26-t2p-pkg-$DN" -- "$DN" "${DONOR_ARGS[@]}")
  [ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
  [ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
  D_REL="ni26-t2p-pkg-$DN-$J1/$DN.zip"; DEP=(--dependency="afterok:$J1"); DSE=record
fi
J2=$(sub S1_EXPORT="BASE_SHA_EXPECT=$BASE_SHA,DONOR_SHA_EXPECT=$DSE$PBX" S1_PURPOSE="$P: (2) swap + inert repack + K1 (zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_w24_build.slurm "${V[@]}" --time=02:00:00 "${DEP[@]}" --job-name="ni26-t2p-w24-$N" -- "$R/$BASE_REL" "$R/$D_REL" "$N")
[ "$mode" == dry ] && { echo "dry: (2) validated (DONOR_LOCAL); the rest need its id"; exit 0; }
Z_REL="ni26-t2p-w24-$N-$J2/$N.zip"
J5=$(sub S1_PURPOSE="$P: (5) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  "${VH[@]}" --time=01:00:00 --dependency="afterok:$J2" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)      # first in the H200 queue
J3=$(sub S1_EXPORT="OLD_SHA_EXPECT=$BASE_SHA" S1_PURPOSE="$P: (3) the package check vs the base (files / mechanism / routes; zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=02:00:00 --dependency="afterok:$J2" --job-name="ni26-t2p-pkgcheck-$N" -- w24 "$BASE_REL" "$Z_REL" "$N")
J4=$(sub S1_PURPOSE="$P: (4) condition K2 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=02:00:00 \
  --dependency="afterok:$J3" --job-name="ni26-t2p-w24kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J3/$N.pkgcheck.pred.npz" "$N")
echo "W24' ($ROUTER base$([ $pb = 1 ] && echo ', PREBUILD: NOT APPROVED')): (1) donor $J1 -> (2) build $J2 -> (3) check $J3 -> (4) K2 $J4; (5) timing $J5; the zip runs/ni26/$Z_REL"
