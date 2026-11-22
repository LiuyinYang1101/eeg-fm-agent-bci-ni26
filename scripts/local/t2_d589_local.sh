#!/usr/bin/env bash
# T2-D589 (mixup inside the in-stream P1 students' fine-tune; registered 2026-11-18 before any job): ONE root per draw set = W21's lp root with
# ONLY the P1 students re-trained under r1_pulx_vp2_e180_vr_pmx4 (the same pooled fit, bitwise; the copies' fine-tune with mixup Beta(0.4, 0.4)),
# the SAME teacher (the lp root's teacher_p1, symlinked). One local 5090 job via gpu-submit: t2_d536_p1_local.sh's P1 stage (slots sh,ifn,
# refits 60,100,150, confR) with the constants moved and the teacher build dropped, into outputs/t2_d589/pmx_<d0>; then the root
# outputs/t2_d589_pmx_<d0> = every lp-root entry symlinked EXCEPT p1_sh / p1_ifn, which hold the pmx4 students under the stack's names
# (vr_pmx4 -> vr). Read: t2_lab_pair_read.py, REF = the lp root, ARM = this root, both --idea w20_fb.
#   mem-guard bash scripts/local/t2_d589_local.sh 825|828|831|834 [fm45|fm45d583|fm45d586|fm45d588] [--dry]
# T2-D592 (the combination D589 + D591): 'fm45' = the same P1 students on D591's fm45 lp root (its 0.45 teacher) -> t2_d589_pmx_fm45_<d0>;
# the THREE-part combination (D589 + D591 + an FM recipe, Track3 LOW): 'fm45d58x' = on t2_d571_lpw20d557d58xfm45_831 (set C only)
set -euo pipefail
cd "$(dirname "$0")/../.."
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done
D0=${1:?825|828|831|834}; V=""; DRY=""
for a in "${@:2}"; do case $a in fm45|fm45d583|fm45d586|fm45d588) V=$a ;; --dry) DRY=1 ;; *) echo "unknown arg $a"; exit 2 ;; esac; done
case $D0 in 825) LP=t2_d571_lpw20_825 ;; 828) LP=t2_d571_lpw20_828 ;; 831) LP=t2_d571_lpw20d557_831 ;; 834) LP=t2_d571_lpw20d557_834 ;; *) echo "draws 825 | 828 | 831 | 834"; exit 2 ;; esac
[ -z "$V" ] || LP=${LP/lpw20d557_/lpw20d557fm45_}; [ -z "$V" ] || LP=${LP/lpw20_/lpw20fm45_}   # T2-D592: D591's fm45 lp root
if [ -n "$V" ] && [ "$V" != fm45 ]; then [ $D0 = 831 ] || { echo "$V: set C only"; exit 2; }; LP=t2_d571_lpw20d557${V#fm45}fm45_831; fi   # the three-part root
FILES="src/ni26 tracks/t2_bci/scripts scripts/local/t2_d536_p1_local.sh scripts/local/t2_d589_local.sh scripts/local/t2_d540a_stack_read.py scripts/local/t2_lab.py"
[ -z "$(git status --porcelain -- $FILES)" ] || { echo "commit the D589 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
N=t2_d589_pmx${V:+_$V}_$D0; R=$O/$N; W=$O/t2_d589/pmx${V:+_$V}_$D0
[ ! -e "$R" ] && [ ! -e "$W" ] || { echo "$R or $W exists: never overwrite a read's inputs"; exit 3; }
grep -q "done d571" $O/$LP/chain.log 2>/dev/null || { echo "$LP is not done"; exit 3; }
[ "$(ls $O/$LP/teacher_p1 | grep -c '^lc9_.*\.npz$')" -eq 30 ] || { echo "$LP/teacher_p1 is not 30 files"; exit 3; }
D1=$((D0 + 1)); D2=$((D0 + 2))
t=$(bash scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR --dry | grep -v "scripts/local/t2_d536_teacher.py")
t=${t//t2_d536/t2_d589\/pmx${V:+_$V}_$D0}
t=${t//816,817,818/$D0,$D1,$D2}
t=${t// d536 / d589 }
t=${t//--recipe r1_pulx_vp2_e180_vr /--recipe r1_pulx_vp2_e180_vr_pmx4 }
! grep -qE "t2_d536|816|817|818" <<< "$t" || { echo "a D536 constant survived" >&2; exit 4; }
[ "$(grep -c -- '--recipe r1_pulx_vp2_e180_vr_pmx4 ' <<< "$t")" -ge 2 ] || { echo "the pmx4 recipe did not reach both P1 families" >&2; exit 4; }
job=$(mktemp "$O/t2_d589_job.XXXX.sh")
{
  echo "set -uo pipefail; echo \"### \$(date +%T) d589 $N start (git \$(git -C '$(pwd)' rev-parse --short HEAD))\" >> '$W/chain.log'"
  # the P1 stage reads the chain's members / routes / side nets / DC predictions from its own dir (= t2_d571_local.sh's links), here from the lp root
  echo "for x in cnn_fbc cnn_ifn cnn_sh route_fbc route_ifn route_sh side_ifn side_sh s10; do [ -e '$O/$LP/'\$x ] || { echo \"### d589: $LP/\$x missing\" >> '$W/chain.log'; exit 5; }; ln -s '$O/$LP/'\$x '$W/'\$x; done"
  echo "for x in '$O/$LP'/pred_s*_dc.npz; do ln -s \"\$x\" '$W/'; done"
  echo "( $t ) || { echo \"### d589 p1 FAILED \$(date +%T)\" >> '$W/chain.log'; exit 6; }"
  echo "n=\$(ls '$W'/p1_sh/runs '$W'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d589 p1 short: \$n isS files (need 60)\" >> '$W/chain.log'; exit 6; }"
  echo "mkdir -p '$R'; for x in '$O/$LP'/*; do b=\$(basename \"\$x\"); case \$b in p1_sh|p1_ifn) ;; *) ln -s \"\$x\" '$R/'\$b ;; esac; done"
  echo "for fam in sh ifn; do mkdir -p '$R/'p1_\$fam/runs; for f in '$W/'p1_\$fam/runs/*.npz; do ln -s \"\$f\" '$R/'p1_\$fam/runs/\$(basename \"\$f\" | sed 's/_vr_pmx4/_vr/'); done; done"
  echo "echo \"### done d589 $N \$(date +%T): root \$(ls '$R' | wc -l) entries (the lp root has \$(ls '$O/$LP' | wc -l)); P1 files \$(find '$R/p1_sh' '$R/p1_ifn' -name '*.npz' | wc -l)\" >> '$W/chain.log'"
} > "$job"
if [ -n "$DRY" ]; then cat "$job"; rm -f "$job"; exit 0; fi
mkdir -p "$W"; ln -s "$O/$LP/teacher_p1" "$W/teacher_p1"
gpu-submit -L "neural_interface_challenge:t2-d589-pmx${V:+-$V}-$D0" bash "$job"
