#!/usr/bin/env bash
# T2-D603 PREBUILD -> the approved W22 name: sha-pinned, NO rebuild (Track3's point (d) + the manager, 2026-11-19).
# The ONLY way a launch_w22_d603.sh PREBUILD zip becomes an upload candidate. Local, after the PREBUILD chain is gated and fetched:
#   W22_APPROVED="<the owner's typed line in T2's shell naming D603 (item 164)>" PB_SHA=<the 64-hex sha256 of the gated PREBUILD zip> \
#     bash tracks/t2_bci/sofia/promote_w22_d603_prebuild.sh <local dir of the fetched PREBUILD build job> dry|GO
# Checks (dry and GO): the W22_APPROVED test == launch_w22_d603.sh GO's; the dir holds exactly one <N>_PREBUILD.zip, its .zip.sha256 and
#   README_PREBUILD.txt; sha256(zip) == PB_SHA == the .zip.sha256 written by the build job on sofia; the approved name is not taken.
# GO then: a relative symlink <dir>/<N>.zip -> <N>_PREBUILD.zip (the same bytes) + README_PROMOTED.txt (the line, the sha, the date).
# It never writes lead/UPLOAD_TODAY_*: the manager links the promoted name there after Track3's fresh review and its own re-hash.
set -euo pipefail
dir=${1:?local dir of the fetched PREBUILD build job}; mode=${2:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W22_APPROVED:-}" == *"T2 W22"* && "${W22_APPROVED:-}" == *"D603"* ]] || { echo "set W22_APPROVED to the owner's typed line naming D603"; exit 3; }
[[ "${PB_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "PB_SHA = the 64-hex sha256 of the gated PREBUILD zip"; exit 2; }
[ -d "$dir" ] || { echo "no dir $dir"; exit 2; }
mapfile -t zs < <(cd "$dir" && ls -1 -- *_PREBUILD.zip 2>/dev/null || true)
[ "${#zs[@]}" = 1 ] || { echo "!! expected exactly one *_PREBUILD.zip in $dir, found ${#zs[@]}"; exit 4; }
pz=${zs[0]}; az=${pz%_PREBUILD.zip}.zip
[[ "$pz" =~ ^t2p_fin2_w22[a-z0-9]*d603_PREBUILD\.zip$ ]] || { echo "!! $pz is not a launch_w22_d603.sh PREBUILD zip"; exit 4; }
[ -f "$dir/README_PREBUILD.txt" ] && grep -q "PREBUILD - NOT APPROVED" "$dir/README_PREBUILD.txt" || { echo "!! no README_PREBUILD.txt from the build job"; exit 4; }
[ -f "$dir/$pz.sha256" ] || { echo "!! no $pz.sha256 from the build job"; exit 4; }
[ ! -e "$dir/$az" ] && [ ! -L "$dir/$az" ] || { echo "!! $dir/$az already exists"; exit 4; }
h=$(sha256sum "$dir/$pz" | cut -d' ' -f1); s=$(cut -d' ' -f1 "$dir/$pz.sha256")
echo "### $pz: sha256 $h (the build job's .sha256 $s; PB_SHA $PB_SHA)"
[ "$h" = "$PB_SHA" ] && [ "$h" = "$s" ] || { echo "!! sha mismatch: not the gated PREBUILD zip"; exit 4; }
if [ "$mode" == dry ]; then echo "dry: would link $dir/$az -> $pz and write README_PROMOTED.txt"; exit 0; fi
ln -s "$pz" "$dir/$az"
[ "$(sha256sum "$dir/$az" | cut -d' ' -f1)" = "$PB_SHA" ] || { rm -f "$dir/${az:?}"; echo "!! the link does not read back PB_SHA"; exit 4; }
printf '%s\n' "PROMOTED $(date '+%F %T %Z'): $az -> $pz (sha256 $PB_SHA, the gated PREBUILD bytes; no rebuild)." \
  "The owner's line (W22_APPROVED): $W22_APPROVED" \
  "Upload only after Track3's fresh review and the manager's re-hash; the manager links it into lead/UPLOAD_TODAY_*." > "$dir/README_PROMOTED.txt"
echo "### PROMOTED: $dir/$az -> $pz (sha256 $PB_SHA); README_PROMOTED.txt written"
