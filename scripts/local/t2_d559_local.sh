#!/usr/bin/env bash
# T2-D559 (the refit-schedule study; registered 2026-11-17 before any number; the manager's YES 17:xx, "as a lab screen on the 5090"): ONE
# schedule on ONE draw set = a new root outputs/t2_d559_<tag>_<d0> whose CNN members (cnn_*, route_*, side_*, s10, the DC bag) are symlinks
# to the draw set's chain (825 -> t2_d555, 828 -> t2_d557), with its OWN teacher (scripts/local/t2_d559_teacher.py: post_r<c> at the
# schedule) and its OWN in-stream students (T2-D512 part P1 stage 2 EXACTLY = t2_d536_p1_local.sh sh,ifn <cks> confR with the constants
# moved, as the chains do). Nothing else differs from the draw set's W19 read.
#   mem-guard bash scripts/local/t2_d559_local.sh <tag> 825|828|831 <cks> [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?tag}; D0=${2:?825|828|831}; CKS=${3:?cks}
case $D0 in 825) SRCD=t2_d555 ;; 828) SRCD=t2_d557 ;; 831) SRCD=t2_d558 ;; *) echo "draws 825 | 828 | 831"; exit 2 ;; esac
[[ "$TAG" =~ ^[a-z0-9]+$ && "$CKS" =~ ^[0-9]+(,[0-9]+)+$ ]] || exit 2
FILES="scripts/local/t2_d536_p1_local.sh scripts/local/t2_d512_teacher.py scripts/local/t2_d559_teacher.py scripts/local/t2_d559_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts $FILES)" ] || { echo "commit the D559 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
N=t2_d559_${TAG}_$D0; R=$O/$N
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
D1=$((D0 + 1)); D2=$((D0 + 2))
t=$(bash scripts/local/t2_d536_p1_local.sh sh,ifn "$CKS" confR --dry)
t=${t//scripts\/local\/t2_d536_teacher.py/scripts/local/t2_d559_teacher.py --root $N --draws $D0 --cks $CKS}
t=${t//t2_d536/$N}
t=${t//816,817,818/$D0,$D1,$D2}
t=${t// d536 / d559 }
! grep -qE "t2_d536|816|817|818" <<< "$t" || { echo "a D536 constant survived" >&2; exit 4; }
job=$(mktemp "$O/t2_d559_job.XXXX.sh")
{
  echo "set -uo pipefail; mkdir -p '$R'; echo \"### \$(date +%T) d559 $N start (git \$(git -C '$(pwd)' rev-parse --short HEAD); cks $CKS)\" >> '$R/chain.log'"
  echo "for x in cnn_fbc cnn_ifn cnn_sh route_fbc route_ifn route_sh side_ifn side_sh s10; do [ -d '$O/$SRCD/'\$x ] || { echo \"### d559: $SRCD/\$x missing\" >> '$R/chain.log'; exit 5; }; ln -s '$O/$SRCD/'\$x '$R/'\$x; done"
  echo "for x in '$O/$SRCD'/pred_s*_dc.npz; do ln -s \"\$x\" '$R/'; done"
  echo "( $t ) || { echo \"### d559 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d559 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "echo \"### done d559 $N \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${4:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d559-$TAG-$D0" bash "$job"
