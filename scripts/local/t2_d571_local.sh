#!/usr/bin/env bash
# T2-D571 / D572 / D569-amended (registered 2026-11-18 before any number): ONE P1 root on ONE draw set whose teacher is the arm's REAL teacher,
# = D569's root mechanism (outputs/t2_d571_<tag>_<d0>: the chain's CNN members symlinked; its OWN teacher_p1 and P1 students = t2_d536_p1_local.sh
# sh,ifn 60,100,150 confR with the constants moved, the teacher build line dropped). Tags:
#   fbw20 / fbw20d557 = fm_feedback's sequential teacher (FM rows = the arm's current adapted FM part; W20, or W20+D557 on 831)  [D571, D569]
#   unw20 / unw20d557 = the stable current teacher (FM rows = the current members' UNADAPTED outputs)                          [D572]
#   lpw20 / lpw20d557 = LP(the fb root's teacher) (alpha 0.5, k 10, the members' features)                                     [D569 amended]
#   mem-guard bash scripts/local/t2_d571_local.sh <tag> 825|828|831 [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done   # T2-D571 F7 / F8 (Track3): no inherited NI26_* env (the F4 pin; the members are the tag's)
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
TAG=${1:?tag}; D0=${2:?825|828|831}
D568=0; D582=0; D583=0; D586=0; D588=0; D596=0; D611=0; D618=0; FM45=0
case $TAG in fbw20|unw20|lpw20) D557=0 ;; fbw20d557|unw20d557|lpw20d557) D557=1 ;; fbw20d568|lpw20d568) D557=1; D568=1 ;; fbw20d582|lpw20d582) D557=1; D582=1 ;; fbw20d583|lpw20d583) D557=1; D583=1 ;; fbw20d586|lpw20d586) D557=1; D586=1 ;; fbw20d588|lpw20d588) D557=1; D588=1 ;; fbw20d596|lpw20d596) D557=1; D596=1 ;; fbw20fm45|lpw20fm45) D557=0; FM45=1 ;; fbw20d557fm45|lpw20d557fm45) D557=1; FM45=1 ;;
  fbw20d557d583fm45|lpw20d557d583fm45) D557=1; D583=1; FM45=1 ;; fbw20d557d586fm45|lpw20d557d586fm45) D557=1; D586=1; FM45=1 ;;   # T2-D592 combination roots
  fbw20d557d588fm45|lpw20d557d588fm45) D557=1; D588=1; FM45=1 ;;
  fbw20d611|lpw20d611) D557=1; D611=1 ;;   # T2-D611: its D read (set D only)
  fbw20d618|lpw20d618) D557=1; D618=1 ;;   # T2-D618: its D read (set D only)
  *) echo "tag fbw20 | fbw20d557 | unw20 | unw20d557 | lpw20 | lpw20d557 | fbw20d568 | lpw20d568"; exit 2 ;; esac
ALPHA=$([[ $TAG == lp* ]] && echo 0.5 || echo 0)
case $D0 in 825) SRCD=t2_d555; SET=A ;; 828) SRCD=t2_d557; SET=B ;; 831) SRCD=t2_d558; SET=C ;; 834) SRCD=t2_d578; SET=D ;; 860) SRCD=t2_d615; SET=E ;; *) echo "draws 825 | 828 | 831 | 834 | 860"; exit 2 ;; esac
FILES="scripts/local/t2_d536_p1_local.sh scripts/local/t2_lab.py scripts/local/t2_lab_ideas.py scripts/local/t2_d540a_stack_read.py scripts/local/t2_d571_local.sh scripts/local/t2_shipped_starts_w20d557.json"
[ $D557 = 0 ] || [ $D0 = 831 ] || [ $D0 = 834 ] || [ $D0 = 860 ] || { echo "the W20+D557 teacher exists on 831 / 834 / 860 only"; exit 2; }
{ [ $D0 != 834 ] && [ $D0 != 860 ]; } || [ $D557 = 1 ] || { echo "sets D / E hold W20+D557's members only: use the *d557 tags"; exit 2; }
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts $FILES)" ] || { echo "commit the D571 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
N=t2_d571_${TAG}_$D0; R=$O/$N
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
D1=$((D0 + 1)); D2=$((D0 + 2))
t=$(bash scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR --dry | grep -v "scripts/local/t2_d536_teacher.py")
t=${t//t2_d536/$N}
t=${t//816,817,818/$D0,$D1,$D2}
t=${t// d536 / d571 }
! grep -qE "t2_d536|816|817|818" <<< "$t" || { echo "a D536 constant survived" >&2; exit 4; }
job=$(mktemp "$O/t2_d571_job.XXXX.sh")
{
  echo "set -uo pipefail; echo \"### \$(date +%T) d571 $N start (git \$(git -C '$(pwd)' rev-parse --short HEAD); alpha $ALPHA)\" >> '$R/chain.log'"
  echo "[ \$(ls '$R/teacher_p1' | grep -c '^lc9_.*\.npz\$') -eq 30 ] || { echo \"### d571: teacher_p1 not 30 files\" >> '$R/chain.log'; exit 5; }"
  echo "for x in cnn_fbc cnn_ifn cnn_sh route_fbc route_ifn route_sh side_ifn side_sh s10; do [ -d '$O/$SRCD/'\$x ] || { echo \"### d571: $SRCD/\$x missing\" >> '$R/chain.log'; exit 5; }; ln -s '$O/$SRCD/'\$x '$R/'\$x; done"
  echo "for x in '$O/$SRCD'/pred_s*_dc.npz; do ln -s \"\$x\" '$R/'; done"
  echo "( $t ) || { echo \"### d571 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d571 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "echo \"### done d571 $N \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${3:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
mkdir -p "$R"
if [ $D557 = 1 ] && [ $D0 = 831 ]; then export NI26_LAB_D557=1; fi   # on 834 the base members ARE W20+D557's (no env)
if [ $D568 = 1 ]; then export NI26_LAB_D568=1; fi   # the W21 combination read (D569 x D568): W20+D557 with the D568 members
if [ $D582 = 1 ]; then [ $D0 = 834 ] || { echo "the D582 roots are for set D only"; exit 2; }; export NI26_LAB_D582=1; fi
if [ $D583 = 1 ]; then [ $D0 = 834 ] || { [ $D0 = 831 ] && [ $FM45 = 1 ]; } || { echo "the D583 roots: set D, or set C as a T2-D592 combination with fm45"; exit 2; }; export NI26_LAB_D583=1; fi   # T2-D583: its D read (the arm's own real teacher)
if [ $D586 = 1 ]; then [ $D0 = 834 ] || { [ $D0 = 831 ] && [ $FM45 = 1 ]; } || { echo "the D586 roots: set D, or set C as a T2-D592 combination with fm45"; exit 2; }; export NI26_LAB_D586=1; fi   # T2-D586: its D read
if [ $D596 = 1 ]; then [ $D0 = 834 ] || { echo "the D596 roots are for set D only"; exit 2; }; export NI26_LAB_D596=1; fi   # T2-D596: its D read
if [ $D611 = 1 ]; then [ $D0 = 834 ] || { echo "the D611 roots are for set D only"; exit 2; }; export NI26_LAB_D611=1; fi   # T2-D611: its D read
if [ $D618 = 1 ]; then [ $D0 = 834 ] || { echo "the D618 roots are for set D only"; exit 2; }; export NI26_LAB_D618=1; fi   # T2-D618: its D read
if [ $D588 = 1 ]; then [ $D0 = 834 ] || { [ $D0 = 831 ] && [ $FM45 = 1 ]; } || { echo "the D588 roots: set D, or set C as a T2-D592 combination with fm45"; exit 2; }; export NI26_LAB_D588=1; fi   # T2-D588: its D read   # T2-D582: its D read (the arm's own real teacher)
echo "### $(date +%T) $N: the NI26_* env used for the teacher build: $(compgen -e | grep '^NI26_' | while read v; do printf '%s=%s ' "$v" "${!v}"; done)" | tee -a "$R/chain.log"
case $TAG in
  fb*) TWA=(); [ $FM45 = 1 ] && TWA=(--tw-fm 0.45)   # T2-D591: the teacher's FM share 0.45 (the CNN families scaled; the template's fam_w)
       $PY -u scripts/local/t2_lab.py fbteach --set $SET --out "$R/teacher_p1" --fm-rows own --proto 0.5 "${TWA[@]}" 2>&1 | grep -E "^\[lab\]|Error|assert" | tail -4 ;;
  un*) $PY -u scripts/local/t2_lab.py fbteach --set $SET --out "$R/teacher_p1" --fm-rows base --proto 0.5 2>&1 | grep -E "^\[lab\]|Error|assert" | tail -4 ;;
  lp*) FB=$O/t2_d571_fb${TAG#lp}_$D0/teacher_p1
       [ "$(ls "$FB" 2>/dev/null | grep -c '^lc9_.*\.npz$')" -eq 30 ] || { echo "build t2_d571_fb${TAG#lp}_$D0 first"; rm -rf "$R"; exit 3; }
       $PY -u scripts/local/t2_lab.py lpteach --set $SET --out "$R/teacher_p1" --src "$FB" --alpha $ALPHA --k 10 2>&1 | grep -E "^\[lab\]|Error|assert" | tail -6 ;;
esac
[ "$(ls "$R/teacher_p1" 2>/dev/null | grep -c '^lc9_.*\.npz$')" -eq 30 ] || { echo "teacher build failed"; exit 7; }
(cd "$R/teacher_p1" && sha256sum lc9_*.npz) > "$R/teacher_p1.sha256"
if [ $FM45 = 1 ] && [[ $TAG == fb* ]]; then   # T2-D591: the fb fm45 root only feeds its lp root's LP teacher (no P1 students read): teacher-only
  echo "### $(date +%T) $N: T2-D591 teacher-only fb root (no P1 job); done d571 teacher-only" >> "$R/chain.log"; exit 0
fi
gpu-submit -L "neural_interface_challenge:t2-d571-$TAG-$D0" bash "$job"
