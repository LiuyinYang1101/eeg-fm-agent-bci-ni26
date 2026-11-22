#!/usr/bin/env bash
# T2-D367: stage one ni26 cpt_pools dataset (data/ni26/t2_proteus/cpt_pools/<name>) into this job's runs/ dir so `s1 fetch` can pull it
# (s1 fetch reads runs/<tag>/ only); a byte-exact copy (sha256 of every file checked against the source), free CPU.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_stage_pool.slurm --partition=zen5_vis --time=00:30:00 --job-name=ni26-t2p-stage-<name> -- <name>
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
OUT=$(s1_outdir "${RUN_NAME:-}")
name=${1:?pool name}
SRC="$S1_ROOT/data/ni26/t2_proteus/cpt_pools/$name"
[ -d "$SRC" ] || { echo "no pool $SRC"; s1_finish 2 "$OUT" 'none' 1; exit 2; }
mkdir -p "$OUT/$name" && cp -a "$SRC/." "$OUT/$name/" && rc=0 || rc=$?
(cd "$SRC" && sha256sum * ) > "$OUT/src.sha256"; (cd "$OUT/$name" && sha256sum -c "$OUT/src.sha256") > "$OUT/check.txt" 2>&1 || rc=4
cat "$OUT/check.txt"
s1_finish $rc "$OUT" "$name/X.npy" 1
