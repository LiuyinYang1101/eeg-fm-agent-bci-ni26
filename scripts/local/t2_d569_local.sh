#!/usr/bin/env bash
# T2-D569 (the label-propagation teacher for the CNN-side refits; registered 2026-11-18 before any number; the manager's seed, local 5090 only):
# ONE P1 root on ONE draw set = outputs/t2_d569_<tag>_<d0> whose CNN members (cnn_*, route_*, side_*, s10, the DC bag) are symlinks to the
# draw set's chain (825 -> t2_d555, 828 -> t2_d557, 831 -> t2_d558), with its OWN teacher_p1 and its OWN in-stream students (T2-D512 part P1
# stage 2 EXACTLY = t2_d536_p1_local.sh sh,ifn 60,100,150 confR with the constants moved, as D559's roots; the teacher build line dropped).
#   tag id = the identity root: the chain's teacher files copied byte for byte (the P1 re-run control = the decision REF)
#   tag lp = the LP teacher (t2_lab.py lpteach, alpha 0.5, k 10; on 831 under NI26_LAB_D557=1 = W20+D557's members' features)
# Step 1 (here, CPU): the teacher files. Step 2: ONE gpu-submit job (symlinks + P1 sh / ifn + the 60-file check).
#   mem-guard bash scripts/local/t2_d569_local.sh id|lp 825|828|831 [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
TAG=${1:?id|lp}; D0=${2:?825|828|831}
case $TAG in id) ALPHA=0 ;; lp) ALPHA=0.5 ;; *) echo "tag id | lp"; exit 2 ;; esac
case $D0 in 825) SRCD=t2_d555; SET=A ;; 828) SRCD=t2_d557; SET=B ;; 831) SRCD=t2_d558; SET=C ;; *) echo "draws 825 | 828 | 831"; exit 2 ;; esac
FILES="scripts/local/t2_d536_p1_local.sh scripts/local/t2_lab.py scripts/local/t2_lab_ideas.py scripts/local/t2_d540a_stack_read.py scripts/local/t2_d569_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts $FILES)" ] || { echo "commit the D569 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
N=t2_d569_${TAG}_$D0; R=$O/$N
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
D1=$((D0 + 1)); D2=$((D0 + 2))
t=$(bash scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR --dry | grep -v "scripts/local/t2_d536_teacher.py")
t=${t//t2_d536/$N}
t=${t//816,817,818/$D0,$D1,$D2}
t=${t// d536 / d569 }
! grep -qE "t2_d536|816|817|818" <<< "$t" || { echo "a D536 constant survived" >&2; exit 4; }
job=$(mktemp "$O/t2_d569_job.XXXX.sh")
{
  echo "set -uo pipefail; echo \"### \$(date +%T) d569 $N start (git \$(git -C '$(pwd)' rev-parse --short HEAD); alpha $ALPHA)\" >> '$R/chain.log'"
  echo "[ \$(ls '$R/teacher_p1' | grep -c '^lc9_.*\.npz\$') -eq 30 ] || { echo \"### d569: teacher_p1 not 30 files\" >> '$R/chain.log'; exit 5; }"
  echo "for x in cnn_fbc cnn_ifn cnn_sh route_fbc route_ifn route_sh side_ifn side_sh s10; do [ -d '$O/$SRCD/'\$x ] || { echo \"### d569: $SRCD/\$x missing\" >> '$R/chain.log'; exit 5; }; ln -s '$O/$SRCD/'\$x '$R/'\$x; done"
  echo "for x in '$O/$SRCD'/pred_s*_dc.npz; do ln -s \"\$x\" '$R/'; done"
  echo "( $t ) || { echo \"### d569 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d569 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "echo \"### done d569 $N \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${3:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
mkdir -p "$R"
if [ "$TAG" = id ]; then
  $PY -u scripts/local/t2_lab.py lpteach --set $SET --out "$R/teacher_p1" --alpha 0 2>&1 | tail -2
  cmp -s <(cd "$O/$SRCD/teacher_p1" && sha256sum lc9_*.npz) <(cd "$R/teacher_p1" && sha256sum lc9_*.npz) || { echo "identity teacher != the chain's"; exit 7; }
else
  if [ "$D0" = 831 ]; then export NI26_LAB_D557=1; fi
  $PY -u scripts/local/t2_lab.py lpteach --set $SET --out "$R/teacher_p1" --alpha $ALPHA --k 10 2>&1 | grep -E "^\[lab\]|Error|assert" | tail -6
fi
(cd "$R/teacher_p1" && sha256sum lc9_*.npz) > "$R/teacher_p1.sha256"
gpu-submit -L "neural_interface_challenge:t2-d569-$TAG-$D0" bash "$job"
