#!/usr/bin/env bash
# T2-D580 (L2: joint mixup for the CNN side nets; registered 2026-11-18 before any job): ONE side-net root on ONE draw set, local 5090 via
# gpu-submit (ONE job): the chain's side stages EXACTLY (t2_d536_side_local.sh's S5 = seeds d + k * 1000, k 0..4, and D543's S10 half =
# k 5..9; ShallowNet / IFNet, EA, --side dcmean, NI26_SIDE_DUMP=1, 3 shards) with the recipe r1_pulx_vp2_e180_vr -> r1_pulx_vp2_e180_vr_smx4,
# into outputs/t2_d580/mix_<d0>; then the root outputs/t2_d580_mix_<d0> = every entry of W21's lp root symlinked (t2_d571_lpw20d557_831 on C,
# t2_d571_lpw20_828 on B) EXCEPT side_sh / side_ifn / s10, which hold the mixed side nets under the stack's names (vr_smx4 -> vr).
# The reader takes S10 from the root for t2_d580_ roots (scripts/local/t2_d540a_stack_read.py).
#   mem-guard bash scripts/local/t2_d580_local.sh 831|828|834 [--dry]   (834: set D, after set D's W21 root t2_d571_lpw20d557_834 is done)
set -euo pipefail
cd "$(dirname "$0")/../.."
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done
D0=${1:?831|828|834}
case $D0 in 831) LP=t2_d571_lpw20d557_831 ;; 828) LP=t2_d571_lpw20_828 ;; 834) LP=t2_d571_lpw20d557_834 ;; *) echo "draws 831 | 828 | 834"; exit 2 ;; esac   # 834 = its D read (the night plan 2026-11-18)
FILES="src/ni26 tracks/t2_bci/scripts scripts/local/t2_d580_local.sh scripts/local/t2_d540a_stack_read.py scripts/local/t2_lab.py"
[ -z "$(git status --porcelain -- $FILES)" ] || { echo "commit the D580 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
N=t2_d580_mix_$D0; R=$O/$N; W=$O/t2_d580/mix_$D0
[ ! -e "$R" ] && [ ! -e "$W" ] || { echo "$R or $W exists: never overwrite a read's inputs"; exit 3; }
[ -d "$O/$LP" ] || { echo "no $LP"; exit 3; }
REC=r1_pulx_vp2_e180_vr_smx4
S5=""; S10=""; for d in $D0 $((D0 + 1)) $((D0 + 2)); do for k in 0 1 2 3 4; do S5+="$((d + k * 1000)),"; done; for k in 5 6 7 8 9; do S10+="$((d + k * 1000)),"; done; done
S5=${S5%,}; S10=${S10%,}
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$O/t2_d580_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$W'; echo \"### \$(date +%T) d580 $N start (git \$(git rev-parse --short HEAD); recipe $REC)\" >> '$W/arm.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for part in s5 s10; do
    for fam in sh ifn; do
      m=$([ $fam = sh ] && echo shallow || echo ifnet)
      seeds=$([ $part = s5 ] && echo "$S5" || echo "$S10")
      D=$W/$([ $part = s5 ] && echo "side_$fam" || echo "s10/side_$fam")
      line="--protocols lc9 --models $m --aligns ea --recipe $REC --seeds $seeds --side dcmean"
      echo "mkdir -p '$D/runs'; echo \"### \$(date +%T) $part $fam: $line\" >> '$W/arm.log'"
      for i in 0 1 2; do
        echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$E' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
      done
      echo "wait"
      echo "n=\$(ls '$D/runs' | grep -c 'smx4__sidedcmean'); echo \"### done $part $fam \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$W/arm.log'; [ \$n -eq 150 ] || exit 8"
    done
  done
  echo "mkdir -p '$R'; for x in '$O/$LP'/*; do b=\$(basename \"\$x\"); case \$b in side_sh|side_ifn|s10) ;; *) ln -s \"\$x\" '$R/'\$b ;; esac; done"
  echo "for fam in sh ifn; do for sub in side_\$fam s10/side_\$fam; do mkdir -p '$R/'\$sub/runs; for f in '$W/'\$sub/runs/*smx4__sidedcmean__s*.npz; do ln -s \"\$f\" '$R/'\$sub/runs/\$(basename \"\$f\" | sed 's/_vr_smx4__sidedcmean/_vr__sidedcmean/'); done; done; done"
  echo "echo \"### done d580 $N \$(date +%T): root \$(ls '$R' | wc -l) entries (the lp root has \$(ls '$O/$LP' | wc -l)); side files \$(find '$R/side_sh' '$R/side_ifn' '$R/s10' -name '*.npz' | wc -l) (expect 600)\" >> '$W/arm.log'"
} > "$job"
if [ "${2:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d580-mix-$D0" bash "$job"
