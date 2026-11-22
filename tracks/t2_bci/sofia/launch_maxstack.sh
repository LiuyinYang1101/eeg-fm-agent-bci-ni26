#!/usr/bin/env bash
# T2 max-stack builds (the user's goal T2-D324; GO 430): ONE train-all t2p_final build of A1's MOR-free L_V recipe with the passed
# swings, then two chained repacks: (1) the SKλ decoder (tta sinkhorn 10 / n0 3 / λ 0.95) + the CUDA-graph path, (2) tv8c
# (--tta-vp 8 --tta-copies through the FBCNet vp bank). Every part is this build's own (no donor). Short ZIP names (< 64 chars).
#   bash tracks/t2_bci/sofia/launch_maxstack.sh dry|GO <name> <cnn2: shallow|shallow_w2> <fm align: sz|ea> <fm seeds, e.g. 0,1> <fm weight> [fm seed knobs JSON]
#   e.g. M1 = two ST-L seeds at fm 0.3:  bash tracks/t2_bci/sofia/launch_maxstack.sh GO M1 shallow sz 0,1 0.3
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; name=${2:?name}; cnn2=${3:?cnn2}; fal=${4:?fm align}; fseeds=${5:?fm seeds}; fw=${6:?fm weight}; knobs=${7:-}
kn=(); [ -n "$knobs" ] && kn=(--fm-seed-knobs "$knobs")       # T2-D333: per-ST-seed vp / lp_epochs
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "$cnn2" == shallow || "$cnn2" == shallow_w2 ]] && [[ "$fal" == sz || "$fal" == ea ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/submission/t2p_submission.py src/ni26/train.py tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the build code first"; exit 2; }
R=$S1_ROOT/runs/ni26
CK=${CK:-$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth}   # backlog 90 / T2-D392: CK=<the all-user P1 ckpt> rebuilds the ST-L slot from it (unset = d111-all, as every build)
S=${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}            # A1's CNN seeds (L_V d0); T2-D444: SEEDS=<15 fresh seeds> = a redraw (unset = every earlier build)
REC=${REC:-r1_pulxx_vp2_e180_vr_rc75}; REC2PU=${REC2PU:-r1_pulx}   # T2-D334: S1's pv10 copies = REC=..._rc75_pv10 REC2PU=r1_pulx_vp2_e180_vr_pv10
W3=${W3:-0}; X3=${X3:-}; TPL=${TEMPLATE:-}                     # T2-D352 (M5): a third CNN family's share / its t2p_final args / the W11 template
cnn2w=$(python3 -c "print(repr((1 - $fw - $W3) * 3 / 7))")    # CNNs 4 : 3 inside 1 - fm (- w3); W3 0 = the earlier value exactly
tp=(); [ -n "$TPL" ] && tp=(--template "$TPL")
TTA='{"kind":"sinkhorn","iters":10,"n0":3,"lam":0.95}'           # the sk leg's decoder; T2-D471: TTA_START=<[mi, sub, word]> adds the start vector (W12 template; unset = unchanged)
[ -n "${TTA_START:-}" ] && TTA="{\"kind\":\"sinkhorn\",\"iters\":10,\"n0\":3,\"lam\":0.95,\"start\":$TTA_START}"
x3=(); [ -n "$X3" ] && read -r -a x3 <<< "$X3"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh   # the file cap: a new snapshot only when T2 code changed
sub() {   # job name, dependency ('' or id), slurm, args...
  local jn="$1" dep="$2" sl="$3"; shift 3
  local depflag=(); [ -n "$dep" ] && [ "$dep" != 0 ] && depflag=(--dependency="afterok:$dep")
  [ -n "${EXCLUDE:-}" ] && depflag+=(--exclude="$EXCLUDE")          # bad nodes (SOFIA_OPS §6), e.g. EXCLUDE=acc005,acc006,acc012
  [ -n "${NICE:-}" ] && depflag+=(--nice="$NICE")                    # T2-D366: a non-critical build behind the manager's re-nice plan
  local out
  out=$(env "${dry[@]}" S1_PURPOSE="T2 max-stack $name (goal T2-D324; GO 430): cnn2 $cnn2, ST-L align $fal, ST-L seeds $fseeds, fm weight $fw, recipe $REC / $REC2PU; step $jn" \
    s1 submit ni26 "$sl" --time="$TIME" "${depflag[@]}" --job-name="ni26-t2p-ms-$jn" -- "$@" 2>&1)
  echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|error|ERROR" | cut -c1-200 >&2
  local j; j=$(sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"); echo "${j:-0}"
}
TIME=06:00:00 j=$(sub "$name" "" tracks/t2_bci/sofia/t2p_final.slurm \
  "t2p_ms_$name" --model fbcnet_bboth --align none --epochs 180 --seeds "$S" --tta prior:1:3 \
  --model2 "$cnn2" --align2 ea --n02 0 --epochs2 180 --seeds2 "$S" --weight2 "$cnn2w" \
  --fm st_large --fm-align "$fal" --fm-calib gain --fm-epochs 30 --fm-weight "$fw" --fm-mixup 0.4 --fm-seeds "$fseeds" \
  --fm-llrd 0.75 --fm-rc 0.75 "${kn[@]}" --fm-ckpt "$CK" --user-blend 0.3333 --train all \
  --recipe "$REC" --recipe2 r1_pulxx_vp2_e180_vr --user-blend2 0.5 --recipe2-pu "$REC2PU" "${x3[@]}" "${tp[@]}")
TIME=01:30:00 k=$(sub "$name-sk" "$j" tracks/t2_bci/sofia/t2p_repack.slurm \
  "$R/ni26-t2p-ms-$name-$j/t2p_ms_$name.zip" "t2p_ms_${name}_sk" --allow-differ 0 --gpu-graph \
  --set "tta=$TTA" "${tp[@]}")
TIME=01:30:00 sub "$name-tv" "$k" tracks/t2_bci/sofia/t2p_repack.slurm \
  "$R/ni26-t2p-ms-$name-sk-$k/t2p_ms_${name}_sk.zip" "t2p_ms_$name" --tta-vp 8 --tta-copies --tta-vp-recipe r1_pulxx_vp2_e180_vr_rc75 "${tp[@]}" >/dev/null
