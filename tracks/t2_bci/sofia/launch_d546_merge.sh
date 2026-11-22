#!/usr/bin/env bash
# T2-D546b build PREPARATION (the manager's OK ≈ 00:2x, Nov 17; not a package; nothing is packaged before the owner's W19 line): the W19 FM donor
# = W18 b2's merged donor (t2p_fin2_w18_donor.zip: SD d2 with the MI member at slot 0) with slots 2 / 4 (seeds 622 / 624) replaced by the
# D546b continued members (job 1626636's donor fm_0 / fm_1), their side heads kept (--keep-side). CPU on zen5_vis (free).
# T2-D549: N = 1 = the b1-draw fallback (W18 b1's merged donor t2p_fin1_w18_donor.zip; SWAP = b1's continuation run dir, given as the 3rd arg).
#   mem-guard bash tracks/t2_bci/sofia/launch_d546_merge.sh dry|GO [2 | 1 <b1 continuation run dir under runs/ni26>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; n=${2:-2}; [[ "$mode" == dry || "$mode" == GO ]] && [[ "$n" == 1 || "$n" == 2 ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_merge_donors.py tracks/t2_bci/sofia/t2p_merge.slurm tracks/t2_bci/sofia/launch_d546_merge.sh tests/test_t2p_merge_donors.py)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "$n" == 2 ]; then
  BASE=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18_donor.zip; SWAP=$R/ni26-t2p-d546b-cont-b2-1626636/t2p_d546b_cont_b2.zip
else
  BASE=$R/ni26-t2p-w18-t2p_fin1_w18-1622522/t2p_fin1_w18_donor.zip; SWAP=$R/${3:?b1 continuation run dir}/t2p_d546b_cont_b1.zip
fi
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_EXPORT="RUN_NAME=ni26-t2p-d546b-merge-b$n,NAME=t2p_w19b${n}_donor" \
  S1_PURPOSE="T2-D546b build PREPARATION (not a package): W18 b$n's merged FM donor with slots 2 / 4 = the continued members ($SWAP) (side heads kept), for the W19 build only if the owner's line" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_merge.slurm --partition=zen5_vis --account=vsc --time=00:45:00 --job-name=ni26-t2p-d546b-merge-b$n -- \
  --base "$BASE" --swap "$SWAP" --slots 2,4 --swap-index 0,1 --keep-side 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
