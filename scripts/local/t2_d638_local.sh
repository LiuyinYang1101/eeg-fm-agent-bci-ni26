#!/usr/bin/env bash
# T2-D638 (registered + amended ×2 before any number): ONE gpu-submit job on the local 5090 (the manager's sequencing) for the full-stack 2 × 2.
#   (0) identity (ii): the P1-student path (t2p_bench lc9_0, seed 860, the reference lp teacher) == t2_d571_lpw20d557_860's isSpr (<= 1e-5), else STOP;
#   per cell in bw21 bw23 cw21 cw23 (D630's nets B / C x the W21 / session-1 router), W21's lab chain regenerated:
#   (1) the base root (spliced route_*, the base teacher = t2_d512_teacher's logic) -> (2) the base P1 teachers -> (3) the BASE students from the
#   cell's own pooled nets (t2p_bench ss8 + rt1_P04 / rt1_P10 once per spliced fold, seeds 880-882; C: --user-balance) -> (4) spliced into the
#   base root -> (5) fbteach on the base root -> (6) lpteach -> the lp root -> (7) the final P1 teachers -> (8) the FINAL students -> (9) spliced
#   into the lp root; a cell's chain writes '### done d638 cell <cell>' only when every step passed (the reader refuses partial cells).
#   The identity root (scripts/local/t2_d638_identity.sh, CPU) must have PASSED first: this launcher refuses otherwise.
#   AMENDED (identity (ii) stopped the first run: IFNet GPU nondeterminism): (ii) = ShallowNet bitwise AND IFNet <= 1e-3 with 100 % argmax; then the
#   NOISE cell nz (lc9 students of both generations regenerated on id's no-op routes) before the 4 cells.
#   mem-guard bash scripts/local/t2_d638_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_lab.py scripts/local/t2_lab_ideas.py scripts/local/t2_d540a_stack_read.py scripts/local/t2_d512_teacher.py scripts/local/t2_d638_splice.py scripts/local/t2_d638_routes.py scripts/local/t2_d638_local.sh)" ] || { echo "commit the D638 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d638
grep -q "\[d638 identity\] PASS" "$R/identity.log" 2>/dev/null || { echo "the identity root has not PASSED (scripts/local/t2_d638_identity.sh)"; exit 3; }
CELLS=${CELLS:-}                                    # rerun: CELLS="cw21 cw23" = only those cells (no (ii), no nz; the others stand)
for c in ${CELLS:-nz bw21 bw23 cw21 cw23}; do case $c in bw21|bw23|cw21|cw23|nz) ;; *) echo "bad cell $c"; exit 2 ;; esac; [ ! -e "$O/t2_d638_${c}_860" ] && [ ! -e "$O/t2_d638_lp${c}_860" ] || { echo "a $c root exists: never overwrite a read's inputs"; exit 3; }; done
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
COMMON="--instream-all --instream-sched 60,100,150 --instream-sel true,confR --instream-reps 3 --instream-rep-sel confR"
declare -A M=([sh]="--models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr" [ifn]="--models ifnet --aligns ea --recipe r1_pulx_vp2_e180_vr")
job=$(mktemp "$O/t2_d638_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; L='$R/chain.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "for v in \$(compgen -e | grep '^NI26_'); do unset \"\$v\"; done"
  echo "echo \"### \$(date +%T) d638 start (git \$(git rev-parse --short HEAD))\" >> \"\$L\""
  if [ -z "$CELLS" ]; then
  # (0) identity (ii)
  for fam in sh ifn; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$R/p1/id_ii/$fam/runs' ${M[$fam]} --protocols lc9_0 --seeds 860 $COMMON --instream-pl '$O/t2_d571_lpw20d557_860/teacher_p1/{protocol}__s{seed}.npz' >> '$R/p1_id_ii.log' 2>&1 || { echo '### d638 (ii) bench FAILED' >> \"\$L\"; exit 8; }"
  done
  echo "$PY -u scripts/local/t2_d638_splice.py check-ii >> \"\$L\" 2>&1 || { echo '### d638 STOP: identity (ii)' >> \"\$L\"; exit 9; }"
  # the NOISE cell nz (the manager, after (ii)'s amendment): id's no-op routes, BOTH student generations regenerated on lc9 (seeds 860-862)
  # -> the cell-to-cell floor of IFNet's GPU nondeterminism (nz vs the reference root's mix); also the decisive run-to-run check (Track3)
  echo "cell_nz() {"
  echo "  echo \"### \$(date +%T) d638 cell nz start\" >> \"\$L\""
  echo "  $PY -u scripts/local/t2_d638_splice.py base nz >> \"\$L\" 2>&1 || return 1"
  for gen in base final; do
    [ $gen = final ] && {
      echo "  env NI26_D559_ROOT=t2_d638_nz_860 NI26_D559_CKS=60,100,150 $PY -u scripts/local/t2_lab.py fbteach --set E --out '$O/t2_d638_fbnz_860/teacher_p1' --fm-rows own --proto 0.5 >> '$R/teach_nz.log' 2>&1 || return 1"
      echo "  $PY -u scripts/local/t2_lab.py lpteach --set E --out '$O/t2_d638_lpnz_860/teacher_p1' --src '$O/t2_d638_fbnz_860/teacher_p1' --alpha 0.5 --k 10 >> '$R/teach_nz.log' 2>&1 || return 1"
      echo "  $PY -u scripts/local/t2_d638_splice.py lproot nz >> \"\$L\" 2>&1 || return 1"
    }
    T=$([ $gen = base ] && echo "$O/t2_d638_nz_860/teacher_p1" || echo "$O/t2_d638_lpnz_860/teacher_p1")
    for fam in sh ifn; do
      for i in 0 1 2; do
        echo "  $SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$R/p1/${gen}_nz/$fam/runs' --shard $i --nshards 3 ${M[$fam]} --protocols lc9 --seeds 860,861,862 $COMMON --instream-pl '$T/{protocol}__s{seed}.npz' >> '$R/p1/${gen}_nz.log' 2>&1 &"
      done
      echo "  wait"
    done
    echo "  [ \$(cat '$R/p1/${gen}_nz.log' | grep -c Traceback) -eq 0 ] || return 1"
    echo "  $PY -u scripts/local/t2_d638_splice.py linkp1 nz $gen >> \"\$L\" 2>&1 || return 1"
  done
  echo "  echo \"### done d638 cell nz \$(date +%T)\" >> \"\$L\""
  echo "}"
  echo "mkdir -p '$R/p1'; cell_nz || { rc=7; echo \"### d638 cell nz FAILED \$(date +%T)\" >> \"\$L\"; }"
  fi
  [ -n "$CELLS" ] && echo "echo \"### \$(date +%T) d638 RERUN of cells $CELLS (git \$(git rev-parse --short HEAD))\" >> \"\$L\""
  for cell in ${CELLS:-bw21 bw23 cw21 cw23}; do
    X=""; [ "${cell:0:1}" = c ] && X="--user-balance"   # t2p_bench names its outputs ...__ubw itself (--xtag is t2p_fpcalib's, not the bench's: the first run's C cells failed on it)
    echo "cell_$cell() {"
    echo "  echo \"### \$(date +%T) d638 cell $cell start\" >> \"\$L\""
    echo "  $PY -u scripts/local/t2_d638_splice.py base $cell >> \"\$L\" 2>&1 || return 1"
    for gen in base final; do
      [ $gen = final ] && {
        echo "  env NI26_D559_ROOT=t2_d638_${cell}_860 NI26_D559_CKS=60,100,150 $PY -u scripts/local/t2_lab.py fbteach --set E --out '$O/t2_d638_fb${cell}_860/teacher_p1' --fm-rows own --proto 0.5 >> '$R/teach_$cell.log' 2>&1 || return 1"
        echo "  $PY -u scripts/local/t2_lab.py lpteach --set E --out '$O/t2_d638_lp${cell}_860/teacher_p1' --src '$O/t2_d638_fb${cell}_860/teacher_p1' --alpha 0.5 --k 10 >> '$R/teach_$cell.log' 2>&1 || return 1"
        echo "  $PY -u scripts/local/t2_d638_splice.py lproot $cell >> \"\$L\" 2>&1 || return 1"
      }
      echo "  $PY -u scripts/local/t2_d638_splice.py p1t $cell $gen >> \"\$L\" 2>&1 || return 1"
      T=$R/p1t/${gen}_$cell; W=$R/p1/${gen}_$cell
      for fam in sh ifn; do
        for i in 0 1 2; do
          echo "  $SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$W/ss8/$fam/runs' --shard $i --nshards 3 ${M[$fam]} --protocols ss8 --seeds 880,881,882 $X $COMMON --instream-pl '$T/ss8/{protocol}__s{seed}.npz' >> '$W.log' 2>&1 &"
        done
        echo "  wait"
        n=0
        for f in 0 1 2 4 5 6 7 8; do
          echo "  $SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$W/f$f/$fam/runs' ${M[$fam]} --protocols rt1_P04,rt1_P10 --seeds 880,881,882 $X $COMMON --instream-pl '$T/f$f/{protocol}__s{seed}.npz' >> '$W.log' 2>&1 &"
          n=$((n + 1)); [ $((n % 4)) = 0 ] && echo "  wait"
        done
        echo "  wait"
      done
      echo "  [ \$(cat '$W.log' | grep -c Traceback) -eq 0 ] || return 1"
      echo "  $PY -u scripts/local/t2_d638_splice.py students $cell $gen >> \"\$L\" 2>&1 || return 1"
    done
    echo "  echo \"### done d638 cell $cell \$(date +%T)\" >> \"\$L\""
    echo "}"
    echo "cell_$cell || { rc=7; echo \"### d638 cell $cell FAILED \$(date +%T)\" >> \"\$L\"; }"
  done
  echo "echo \"### done d638 all \$(date +%T) rc=\$rc\" >> \"\$L\"; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d638" bash "$job"
