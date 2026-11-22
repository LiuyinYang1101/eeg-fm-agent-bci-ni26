#!/usr/bin/env bash
# T2-D594b (the low-margin MI / WORD set loss for the P1 students; registered 2026-11-18 before any job; its zero-GPU gate passed): ONE root per
# draw set = W21's lp root with ONLY the P1 students re-trained with --instream-setval 0.20 (the same recipe, pooled fit and teacher; the confR
# rows whose LP-teacher top-2 are {MI, WORD} with p_top1 - p_top2 < 0.20 learn -log(p_MI + p_WORD)). One local 5090 job via gpu-submit:
# t2_d536_p1_local.sh's P1 stage with the constants moved, the teacher build dropped and the flag added, into outputs/t2_d594b/sv_<d0>; the root
# outputs/t2_d594b_sv_<d0> = every lp-root entry symlinked EXCEPT p1_sh / p1_ifn (the set-loss students; the same file names).
# Read: t2_lab_pair_read.py, REF = the lp root, ARM = this root, both --idea w20_fb.
#   mem-guard bash scripts/local/t2_d594b_local.sh 825|828|831|834 [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done
D0=${1:?825|828|831|834}; DRY=""
for a in "${@:2}"; do case $a in --dry) DRY=1 ;; *) echo "unknown arg $a"; exit 2 ;; esac; done
case $D0 in 825) LP=t2_d571_lpw20_825 ;; 828) LP=t2_d571_lpw20_828 ;; 831) LP=t2_d571_lpw20d557_831 ;; 834) LP=t2_d571_lpw20d557_834 ;; *) echo "draws 825 | 828 | 831 | 834"; exit 2 ;; esac
FILES="src/ni26 tracks/t2_bci/scripts scripts/local/t2_d536_p1_local.sh scripts/local/t2_d594b_local.sh scripts/local/t2_d540a_stack_read.py scripts/local/t2_lab.py"
[ -z "$(git status --porcelain -- $FILES)" ] || { echo "commit the D589 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
N=t2_d594b_sv_$D0; R=$O/$N; W=$O/t2_d594b/sv_$D0
[ ! -e "$R" ] && [ ! -e "$W" ] || { echo "$R or $W exists: never overwrite a read's inputs"; exit 3; }
grep -q "done d571" $O/$LP/chain.log 2>/dev/null || { echo "$LP is not done"; exit 3; }
[ "$(ls $O/$LP/teacher_p1 | grep -c '^lc9_.*\.npz$')" -eq 30 ] || { echo "$LP/teacher_p1 is not 30 files"; exit 3; }
D1=$((D0 + 1)); D2=$((D0 + 2))
t=$(bash scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR --dry | grep -v "scripts/local/t2_d536_teacher.py")
t=${t//t2_d536/t2_d594b\/sv_$D0}
t=${t//816,817,818/$D0,$D1,$D2}
t=${t// d536 / d594b }
t=${t//--instream-rep-sel confR /--instream-rep-sel confR --instream-setval 0.20 }
! grep -qE "t2_d536|816|817|818" <<< "$t" || { echo "a D536 constant survived" >&2; exit 4; }
[ "$(grep -c -- '--instream-setval 0.20 ' <<< "$t")" -ge 2 ] || { echo "the set flag did not reach both P1 families" >&2; exit 4; }
job=$(mktemp "$O/t2_d594b_job.XXXX.sh")
{
  echo "set -uo pipefail; echo \"### \$(date +%T) d594b $N start (git \$(git -C '$(pwd)' rev-parse --short HEAD))\" >> '$W/chain.log'"
  # the P1 stage reads the chain's members / routes / side nets / DC predictions from its own dir (= t2_d571_local.sh's links), here from the lp root
  echo "for x in cnn_fbc cnn_ifn cnn_sh route_fbc route_ifn route_sh side_ifn side_sh s10; do [ -e '$O/$LP/'\$x ] || { echo \"### d594b: $LP/\$x missing\" >> '$W/chain.log'; exit 5; }; ln -s '$O/$LP/'\$x '$W/'\$x; done"
  echo "for x in '$O/$LP'/pred_s*_dc.npz; do ln -s \"\$x\" '$W/'; done"
  echo "( $t ) || { echo \"### d594b p1 FAILED \$(date +%T)\" >> '$W/chain.log'; exit 6; }"
  echo "n=\$(ls '$W'/p1_sh/runs '$W'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d594b p1 short: \$n isS files (need 60)\" >> '$W/chain.log'; exit 6; }"
  echo "mkdir -p '$R'; for x in '$O/$LP'/*; do b=\$(basename \"\$x\"); case \$b in p1_sh|p1_ifn) ;; *) ln -s \"\$x\" '$R/'\$b ;; esac; done"
  echo "for fam in sh ifn; do mkdir -p '$R/'p1_\$fam/runs; for f in '$W/'p1_\$fam/runs/*.npz; do ln -s \"\$f\" '$R/'p1_\$fam/runs/; done; done"
  echo "echo \"### done d594b $N \$(date +%T): root \$(ls '$R' | wc -l) entries (the lp root has \$(ls '$O/$LP' | wc -l)); P1 files \$(find '$R/p1_sh' '$R/p1_ifn' -name '*.npz' | wc -l)\" >> '$W/chain.log'"
} > "$job"
if [ -n "$DRY" ]; then cat "$job"; rm -f "$job"; exit 0; fi
mkdir -p "$W"; ln -s "$O/$LP/teacher_p1" "$W/teacher_p1"
gpu-submit -L "neural_interface_challenge:t2-d594b-sv-$D0" bash "$job"
