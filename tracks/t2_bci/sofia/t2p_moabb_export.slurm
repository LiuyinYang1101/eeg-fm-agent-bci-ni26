#!/usr/bin/env bash
# Track 2 CPT pools on sofia (free zen5_vis CPU): public MOABB corpora -> unlabelled contract windows (120 Hz, 4 s, fp16)
# under data/ni26/t2_proteus/cpt_pools/<dataset>_v1. Raw subjects are downloaded (NEMAR / MOABB sources; compute nodes
# have internet) into node-local scratch and deleted after export: only the window stage is kept.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_moabb_export.slurm --partition=zen5_vis --time=24:00:00 \
#       --job-name=ni26-t2p-cptpools -- BNCI2014_001 Cho2017 Lee2019_MI Schirrmeister2017 PhysionetMI Stieger2021
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
P="$S1_ROOT/data/ni26/t2_proteus/cpt_pools"; mkdir -p "$P"
RAW="$S1_JOBTMP/moabb_raw"; mkdir -p "$RAW"
export NUMPY_MADVISE_HUGEPAGE=0
W=$(( ${SLURM_CPUS_PER_TASK:-16} / 4 ))
rc=0
for ds in "$@"; do
  dst="$P/${ds,,}_v1"
  if [ -s "$dst/meta.json" ]; then echo "### $ds: exists ($dst)"; cp "$dst/meta.json" "$OUT/${ds}_meta.json"; continue; fi
  echo "### $ds -> $dst ($(date +%T), $W workers, scratch $(df -h "$RAW" | tail -1 | awk '{print $4}') free)"
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_moabb_export.py" --dataset "$ds" --out "$dst" --raw-root "$RAW" \
      --workers "$W" --threads 4 || { echo "!! $ds failed"; rc=3; }
  [ -s "$dst/meta.json" ] && python -c "import json,sys; m=json.load(open(sys.argv[1])); print('###', m['dataset'], m['n_windows'], 'windows x', m['n_chans'], 'ch,', m['n_subjects'], 'subjects, errors', m.get('errors'))" "$dst/meta.json"
  cp "$dst/meta.json" "$OUT/${ds}_meta.json" 2>/dev/null
  du -sh "$dst" 2>/dev/null
done
s1_finish "$rc" "$OUT" '*_meta.json' "$#"
