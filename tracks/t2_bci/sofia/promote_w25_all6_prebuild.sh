#!/usr/bin/env bash
# T2-D642 x9 (W25-ALL6) PREBUILD -> the approved W25 name: sha-pinned, NO rebuild (= promote_w23_d632_prebuild.sh for W25-ALL6).
# The ONLY way a launch_w25.sh PREBUILD zip becomes an upload candidate. Local, after the PREBUILD chain is gated and fetched:
#   W25_APPROVED="<the owner's typed line in T2's shell = USER_LINES item 167>" PB_SHA=<the 64-hex sha256 of the gated PREBUILD zip> \
#     TIME_DIR=<the fetched timing job dir (1644832)> KIT_DIR=<K2's (1644831)> PKG_DIR=<the pkgcheck's (1644830)> \
#     bash tracks/t2_bci/sofia/promote_w25_all6_prebuild.sh <local dir of the fetched W21lp build job> dry|GO
# Checks (dry and GO): the W25_APPROVED test == launch_w25.sh GO's ("T2 W25 approved" + "D642"); D642's --combine --set all6 PASS
#   (setread/bw21/set_combined_all6.json); the dir holds exactly one t2p_fin2_w25all6_PREBUILD.zip and its .zip.sha256 (the build job's);
#   sha256(zip) == PB_SHA == that .zip.sha256; the HARD structural check <dir>/zipcheck_w21_vs_w25.json passes AND ran on these bytes
#   (its cand_sha256 == PB_SHA) with --donor (4 / 4b) and --d642-ids (5-lite), its base_sha256 == the SHIPPED W21 zip's (94281e72…) and its
#   donor_sha256 == the merge job's <donor>.zip.sha256 (Track3 / the manager, pin 1); MECHANICAL gates (pin 2): the timing job (status OK) with
#   3 reps whose D517-line seconds (the registered AFFINE line 62.0 x ms_per_window - 163; conservative) have a MEAN <= 3,300 s and EVERY
#   rep <= 3,400 s (the standing worker rule); K2 (status OK, preds == the pkgcheck's, 0 errors / kept-trained fallbacks); the pkgcheck (status
#   OK, finite, every gate true); the approved name is not taken.
# GO then: a relative symlink <dir>/t2p_fin2_w25all6.zip -> the PREBUILD zip (the same bytes) + README_PROMOTED.txt (the line, the sha, the date).
# K1 / K2 / the package check / the H200 timing are read from their job outputs by the manager and Track3 before GO (listed in README_PROMOTED).
# It never writes lead/UPLOAD_TODAY_*: the manager links the promoted name there after Track3's fresh review and its own re-hash.
set -euo pipefail
dir=${1:?local dir of the fetched W21lp build job}; mode=${2:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W25_APPROVED:-}" == *"T2 W25 approved"* && "${W25_APPROVED:-}" == *"D642"* ]] || { echo "set W25_APPROVED to the owner's typed line (item 167, naming D642)"; exit 3; }
[[ "${PB_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "PB_SHA = the 64-hex sha256 of the gated PREBUILD zip"; exit 2; }
W21_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc   # the SHIPPED W21 zip (lead/UPLOAD_TODAY_2026-11-21)
for v in TIME_DIR KIT_DIR PKG_DIR; do [ -d "${!v:-}" ] && grep -q "status=OK" "${!v}/S1_DONE" 2>/dev/null || { echo "!! $v: a fetched job dir with S1_DONE status=OK"; exit 3; }; done
C=$HOME/workspace/outputs/neural_interface_challenge/t2_d642/setread/bw21/set_combined_all6.json
python3 -c "import json,sys; sys.exit(0 if json.load(open('$C'))['passes'] is True else 1)" || { echo "!! D642's ALL6 combine has not PASSED ($C)"; exit 3; }
[ -d "$dir" ] || { echo "no dir $dir"; exit 2; }
mapfile -t zs < <(cd "$dir" && ls -1 -- *_PREBUILD.zip 2>/dev/null || true)
[ "${#zs[@]}" = 1 ] || { echo "!! expected exactly one *_PREBUILD.zip in $dir, found ${#zs[@]}"; exit 4; }
pz=${zs[0]}; az=${pz%_PREBUILD.zip}.zip
[ "$pz" = t2p_fin2_w25all6_PREBUILD.zip ] || { echo "!! $pz is not a launch_w25.sh PREBUILD zip"; exit 4; }
[ -f "$dir/$pz.sha256" ] || { echo "!! no $pz.sha256 from the build job"; exit 4; }
[ ! -e "$dir/$az" ] && [ ! -L "$dir/$az" ] || { echo "!! $dir/$az already exists"; exit 4; }
h=$(sha256sum "$dir/$pz" | cut -d' ' -f1); s=$(cut -d' ' -f1 "$dir/$pz.sha256")
echo "### $pz: sha256 $h (the build job's .sha256 $s; PB_SHA $PB_SHA)"
[ "$h" = "$PB_SHA" ] && [ "$h" = "$s" ] || { echo "!! sha mismatch: not the gated PREBUILD zip"; exit 4; }
ZC=$dir/zipcheck_w21_vs_w25.json
python3 - "$ZC" "$PB_SHA" "$W21_SHA" <<'PYEOF' || { echo "!! the HARD structural check $ZC has not passed on these bytes (with --donor and --d642-ids)"; exit 4; }
import json, sys
import os
r = json.load(open(sys.argv[1]))
dsha = open(r["donor"] + ".sha256").read().split()[0] if r.get("donor") and os.path.exists(r["donor"] + ".sha256") else ""
ok = (r["passes"] is True and r["cand_sha256"] == sys.argv[2] and r["base_sha256"] == sys.argv[3] and r["donor"] and dsha and r["donor_sha256"] == dsha
      and any(n.startswith("4b.") for n in r["notes"]) and any(n.startswith("5-lite.") for n in r["notes"]))
print(f"### zipcheck: passes {r['passes']}, cand {r['cand_sha256'][:12]} (PB_SHA {sys.argv[2][:12]}), base {r['base_sha256'][:12]} (W21 {sys.argv[3][:12]}), "
      f"donor {r.get('donor_sha256', '')[:12]} (the merge job's .sha256 {dsha[:12]})")
sys.exit(0 if ok else 1)
PYEOF
python3 "$(dirname "$0")/../scripts/t2p_w25_gates.py" "$TIME_DIR" "$KIT_DIR" "$PKG_DIR" > "$dir/gates_w25.txt" \
  || { cat "$dir/gates_w25.txt"; echo "!! a mechanical gate (timing / K2 / pkgcheck) failed"; exit 4; }
cat "$dir/gates_w25.txt"
if [ "$mode" == dry ]; then echo "dry: would link $dir/$az -> $pz and write README_PROMOTED.txt"; exit 0; fi
ln -s "$pz" "$dir/$az"
[ "$(sha256sum "$dir/$az" | cut -d' ' -f1)" = "$PB_SHA" ] || { rm -f "$dir/${az:?}"; echo "!! the link does not read back PB_SHA"; exit 4; }
printf '%s\n' "PROMOTED $(date '+%F %T %Z'): $az -> $pz (sha256 $PB_SHA, the gated PREBUILD bytes; no rebuild)." \
  "The owner's line (W25_APPROVED): $W25_APPROVED" \
  "Gates: D642 ALL6 combine PASS ($C); the structural W21-vs-W25 check PASS ($ZC, on these bytes; base = the shipped W21, donor = the merge job's); the mechanical timing / K2 / pkgcheck gates PASS:" \
  "$(cat "$dir/gates_w25.txt")" \
  "K1 (the build steps) and the gates above are also read by the manager + Track3 before this GO." \
  "Upload only after Track3's fresh review and the manager's re-hash; the manager links it into lead/UPLOAD_TODAY_*." > "$dir/README_PROMOTED.txt"
echo "### PROMOTED: $dir/$az -> $pz (sha256 $PB_SHA); README_PROMOTED.txt written"
