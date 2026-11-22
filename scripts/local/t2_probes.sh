#!/usr/bin/env bash
# T2 config-only repack variants and the S1 build (rows 79 / 89; approved by the user 2026-11-01 ≈ 16:40, R1–R3).
# Config repacks of the Nov 5 winner W* (members byte for byte) + the S1 donor build. Nothing here runs before the Nov 5 read names W*;
# GO only on the user's word. Each repack = t2p_repack.slurm's checks (≈ 0.16 GPU-h).
#   bash scripts/local/t2_probes.sh dry T   <base.zip>                 tv8: --tta-vp 8, copies off (vp bank of the FBCNet recipe)
#   bash scripts/local/t2_probes.sh dry U   <base.zip>                 copy odds x 2: user_blend.r 0.5, user_blend2.r 0.6667
#   bash scripts/local/t2_probes.sh dry F   <base.zip> mf|mor          fm_weight up, the FM members' and the CNNs' ratios kept (mf 2/9 -> 0.3 = 4:3:3; mor 0.3 -> 0.4)
#   bash scripts/local/t2_probes.sh dry set <name> <base.zip> <--set k=v ...>    later variants (2nd step, reversal, C) by explicit edits
#   bash scripts/local/t2_probes.sh dry tv  <name> <base.zip> copies|nocopies     the tv repack alone (C's second step)
#   bash scripts/local/t2_probes.sh dry S1d <tag> <cnn seeds, e.g. 0-14>          the S1 donor: CNN-only t2p_final, pv10 copies (≈ 0.5 GPU-h)
#   bash scripts/local/t2_probes.sh dry S1  <base.zip> <donor.zip>                copies swap, then (afterok) tv8c on its output
# <base.zip> / <donor.zip> = absolute sofia paths ($R/...). After each job: s1 fetch, re-hash, scripts/local/t2_probes.sh check <zip>,
# then a docs/SUBMISSIONS.md row before the user's upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=$S1_ROOT/runs/ni26
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
mode="${1:-}"; rung="${2:-}"; shift 2 || true
usage() { sed -n '4,10p' "$0"; exit 2; }
if [ "$mode" == check ]; then         # local: the template sha must be a warm-up-Finished one; config summary
  z="$rung"; [ -f "$z" ] || { echo "no file $z"; exit 2; }
  mem-guard $PY -I - "$z" <<'EOF'
import hashlib, json, sys, zipfile
OK = {"570b7bd4": "W9 (MOR-free + SKλ)", "b7ccc187": "W10 (MOR + SKλ)", "883fb3a2": "W8 (MOR-free, row 75 twin)"}
z = zipfile.ZipFile(sys.argv[1]); s = hashlib.sha256(z.read("submission.py")).hexdigest()[:8]; c = json.loads(z.read("config.json"))
print("zip sha256", hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
print("submission.py", s, "->", OK.get(s, "NOT A WARM-UP-FINISHED TEMPLATE"))
print("tta", c.get("tta"), "| cnn_tta_vp", {k: v for k, v in (c.get("cnn_tta_vp") or {}).items() if k != "file"} or None)
print("fm_weight", c.get("fm_weight"), "| members", [(m.get("family", "st"), m.get("weight")) for m in c.get("fm_members", [])],
      "| cnn2.weight", c["cnn2"]["weight"], "| r", c["user_blend"]["r"], c["user_blend2"]["r"],
      "| copies", c["user_blend"].get("copies_from_recipe", "base"), c["user_blend2"].get("recipe_pu"))
sys.exit(0 if s in OK else 4)
EOF
  exit $?
fi
[[ "$mode" == dry || "$mode" == GO ]] || usage
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_final.slurm src/ni26/train.py)" ] \
  || { echo "commit the build code first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26
VPREC=r1_pulxx_vp2_e180_vr_rc75          # the FBCNet pooled recipe whose vp bank the L_V nets trained through (as the W8 / W9 twins)
rep() {   # name, dependency ('' or afterok:<id>), base zip (sofia abs), t2p_repack options...
  local name="$1" dep="$2" zip="$3"; shift 3
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="$dep")
  env "${dry[@]}" S1_PURPOSE="T2 config variants (rows 79 / 89; R1–R3 approved 2026-11-01): $name, a config repack of the Nov 5 winner (~0.16 GPU-h)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 "${depflag[@]}" --job-name="ni26-t2p-repack-pr-$name" \
    -- "$zip" "t2p_pr_$name" "$@" 2>&1 | grep -E "SUBMITTED|DRY RUN|DRYRUN|to start|sbatch:|refus|error|ERROR" | cut -c1-240 || true
}
jid() { sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$1"; }
case "$rung" in
  T)  [ $# -eq 1 ] || usage; rep T "" "$1" --tta-vp 8 --tta-vp-recipe "$VPREC" ;;
  U)  [ $# -eq 1 ] || usage; rep U "" "$1" --set user_blend.r=0.5 --set user_blend2.r=0.6667 ;;
  F)  [ $# -eq 2 ] || usage
      if [ "$2" == mf ]; then rep F-mf "" "$1" --set fm_weight=0.3 --set fm_members.0.weight=0.3 --set cnn2.weight=0.3
      elif [ "$2" == mor ]; then
        read -r a b c d < <($PY -I -c "import json; print(*(json.dumps(x) for x in (0.4, 0.4 * 2 / 3, 0.4 / 3, 0.6 * 3 / 7)))")   # fm_weight 0.3 -> 0.4, ratios kept (amendment 11-01 19:1x)
        rep F-mor "" "$1" --set "fm_weight=$a" --set "fm_members.0.weight=$b" --set "fm_members.1.weight=$c" --set "cnn2.weight=$d"
      else usage; fi ;;
  set) [ $# -ge 3 ] || usage; nm="$1"; z="$2"; shift 2; rep "$nm" "" "$z" "$@" ;;
  tv) [ $# -eq 3 ] || usage; cp=(); [ "$3" == copies ] && cp=(--tta-copies); rep "$1" "" "$2" --tta-vp 8 "${cp[@]}" --tta-vp-recipe "$VPREC" ;;
  S1d) [ $# -eq 2 ] || usage; tag="$1"; S=$($PY -I -c "import sys; a,b=map(int,sys.argv[1].split('-')); print(','.join(map(str,range(a,b+1))))" "$2")
      env "${dry[@]}" S1_PURPOSE="T2 R3 / row 89: S1 donor $tag = a CNN-only build of the leading draw (CNN seeds $2) with pv10 copies (pu_vp 1.0; pooled nets as the draw's), ~0.5 GPU-h; feeds the S1 package" \
        s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=02:00:00 --job-name="ni26-t2p-pkg-s1d-$tag" \
        -- "t2p_s1d_$tag" --model fbcnet_bboth --align none --epochs 180 --seeds "$S" --tta prior:1:3 \
           --model2 shallow --align2 ea --n02 0 --epochs2 180 --seeds2 "$S" --weight2 0.3 \
           --user-blend 0.3333 --train all --recipe r1_pulxx_vp2_e180_vr_rc75_pv10 --recipe2 r1_pulxx_vp2_e180_vr \
           --user-blend2 0.5 --recipe2-pu r1_pulx_vp2_e180_vr_pv10 2>&1 | grep -E "SUBMITTED|DRY RUN|DRYRUN|to start|sbatch:|refus|error|ERROR" | cut -c1-240 || true ;;
  S1) [ $# -eq 2 ] || usage
      out=$(rep S1-cp "" "$1" --copies-from "$2"); echo "$out"
      if [ "$mode" == GO ]; then
        j=$(jid "$out"); [ -n "$j" ] || { echo "S1 step 1 not submitted"; exit 2; }
        rep S1 "afterok:$j" "$R/ni26-t2p-repack-pr-S1-cp-$j/t2p_pr_S1-cp.zip" --tta-vp 8 --tta-copies --tta-vp-recipe "$VPREC"
      else
        echo "  (GO would then submit S1 afterok:<step-1 id> on \$R/ni26-t2p-repack-pr-S1-cp-<id>/t2p_pr_S1-cp.zip --tta-vp 8 --tta-copies --tta-vp-recipe $VPREC)"
      fi ;;
  *) usage ;;
esac
