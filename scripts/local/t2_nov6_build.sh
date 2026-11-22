#!/usr/bin/env bash
# T2 Nov 6 risk-on builds (T2-D320 reads; the user's decisions T2-D321: bar 40 ms, fm_weight free on typical-user reads; GO 430).
# Config repacks only (t2p_repack.slurm's checks: parts, clean equality / NEW0 = REF, robustness, kit replay), chained afterok:
#   X2 = A4 (W10, MOR) + fm 0.4 (ST-L : MOR 2 : 1, CNNs 4 : 3) + tv8c        typical +0.0133 (12 / 12), ≈ 20 ms
#   X1 = A4 + S1's pv10 copies (donor 1552311) + fm 0.4 + tv8c               typical +0.0116 (11 / 12), ≈ 20 ms
#   X3 = S1 (W9, MOR-free) + fm 0.3 (4 : 3 : 3)                              typical +0.0111 (12 / 12), 16.65 ms
#   bash scripts/local/t2_nov6_build.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; [[ "$mode" == dry || "$mode" == GO ]] || { sed -n 2,7p "$0"; exit 2; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the build code first"; exit 2; }
R=$S1_ROOT/runs/ni26
A4=$R/ni26-t2p-repack-lam-a4-lv-d0-mor-1550060/t2p_lam_a4-lv-d0-mor.zip
S1=$R/ni26-t2p-repack-pr-S1-1552338/t2p_pr_S1.zip
DON=$R/ni26-t2p-pkg-s1d-lvd0-1552311/t2p_s1d_lvd0.zip
VPREC=r1_pulxx_vp2_e180_vr_rc75
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
read -r FW B C D < <($PY -I -c "import json; print(*(json.dumps(x) for x in (0.4, 0.4 * 2 / 3, 0.4 / 3, 0.6 * 3 / 7)))")
FMOR=(--set "fm_weight=$FW" --set "fm_members.0.weight=$B" --set "fm_members.1.weight=$C" --set "cnn2.weight=$D")
FMF=(--set fm_weight=0.3 --set fm_members.0.weight=0.3 --set cnn2.weight=0.3)
TV=(--tta-vp 8 --tta-copies --tta-vp-recipe "$VPREC")
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26
rep() {   # name, dependency ('' or a job id), base zip (sofia abs), t2p_repack options... ; echoes the job id (GO) or 0 (dry)
  local name="$1" dep="$2" zip="$3"; shift 3
  local depflag=(); [ -n "$dep" ] && [ "$dep" != 0 ] && depflag=(--dependency="afterok:$dep")
  local out
  out=$(env "${dry[@]}" S1_PURPOSE="T2 Nov 6 risk-on build $name (T2-D320 typical-user reads; T2-D321 bar 40 ms / fm_weight free; GO 430): a config repack (~0.16 GPU-h); feeds the Nov 6 candidates" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 "${depflag[@]}" --job-name="ni26-t2p-repack-n6-$name" \
    -- "$zip" "t2p_n6_$name" "$@" 2>&1)
  echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|error|ERROR" | cut -c1-200 >&2
  local j; j=$(sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"); echo "${j:-0}"
}
p() { echo "$R/ni26-t2p-repack-n6-$1-$2/t2p_n6_$1.zip"; }
# MOR-free rows added 2026-11-05 after the MOR upload Finished 0.806 (R2 keeps the MOR out unless MOR - A1 >= +0.004):
#   X4 = A1 (W9) + fm 0.3 (4 : 3 : 3) + tv8c (A1's copies)     typical +0.0122 (12 / 12; P04 -0.0102)
#   X5 = A1 + fm 0.35 (CNNs 4 : 3) + tv8c                       typical +0.0134 (12 / 12; P04 -0.0171)
A1=$R/ni26-t2p-repack-lam-p1-lv-d0-1550056/t2p_lam_p1-lv-d0.zip
read -r G H < <($PY -I -c "import json; print(*(json.dumps(x) for x in (0.35, 0.65 * 3 / 7)))")
F35=(--set "fm_weight=$G" --set "fm_members.0.weight=$G" --set "cnn2.weight=$H")
for row in ${ROWS:-X1 X2 X3}; do
  case "$row" in
    X2) j=$(rep X2-F "" "$A4" "${FMOR[@]}");          rep X2 "$j" "$(p X2-F "$j")" "${TV[@]}" >/dev/null ;;
    X1) j=$(rep X1-cp "" "$A4" --copies-from "$DON"); k=$(rep X1-F "$j" "$(p X1-cp "$j")" "${FMOR[@]}"); rep X1 "$k" "$(p X1-F "$k")" "${TV[@]}" >/dev/null ;;
    X3) rep X3 "" "$S1" "${FMF[@]}" >/dev/null ;;
    X4) j=$(rep X4-F "" "$A1" "${FMF[@]}");           rep X4 "$j" "$(p X4-F "$j")" "${TV[@]}" >/dev/null ;;
    X5) j=$(rep X5-F "" "$A1" "${F35[@]}");           rep X5 "$j" "$(p X5-F "$j")" "${TV[@]}" >/dev/null ;;
    *) echo "unknown row $row"; exit 2 ;;
  esac
done
