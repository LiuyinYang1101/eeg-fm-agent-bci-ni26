#!/usr/bin/env bash
# T2-D194 step 1: what does computing the routing fingerprint every k-th window free? The line package as built (every = 1) and two copies
# whose config sets user_blend.every = 4 / 8 (files symlinked, config edited; the template already honours `every`), each through
# t2p_profile.py on one H200 (B = 1).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_d194_profile.slurm --time=00:40:00 --job-name=ni26-t2p-d194-profile -- <pkg dir>
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
p="$1"; [[ "$p" = /* ]] || p="$S1_ROOT/runs/ni26/$p"
S="$S1_CODE/tracks/t2_bci/scripts"
rc=0
for k in 1 4 8; do
  q="$p"
  if [ $k -gt 1 ]; then
    q="$OUT/pkg_every$k"; mkdir -p "$q"
    for f in "$p"/*; do ln -s "$f" "$q/$(basename "$f")"; done
    rm "$q/config.json"
    python -c "import json,sys; c=json.load(open(sys.argv[1])); c['user_blend']['every']=int(sys.argv[3]); open(sys.argv[2],'w').write(json.dumps(c, indent=1))" \
      "$p/config.json" "$q/config.json" $k || rc=3
  fi
  python -u "$S/t2p_profile.py" --pkg "$q" --export "$D/export_proteus_v1" --out "$OUT/prof_every$k.json" 2>&1 | grep -v "Warn\|warn(\|make_standard\|pos = " | tail -12 || rc=3
done
s1_finish "$rc" "$OUT" 'prof_every*.json' 3
