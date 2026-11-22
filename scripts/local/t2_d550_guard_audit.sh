#!/usr/bin/env bash
# T2-D550 guard AUDIT (the manager's ask 2, zero GPU): every T2 read that fed a decision since D536 is re-run with the oracle guard in place
# (scripts/local/t2_oracle_guard.py inside the readers' need / load / side_prob). An oracle input would be REFUSED (SystemExit naming the file);
# each re-run's numbers are then compared with the recorded originals (backed up first: the readers rewrite their JSONs).
#   mem-guard bash scripts/local/t2_d550_guard_audit.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
O=$HOME/workspace/outputs/neural_interface_challenge
A=$O/t2_d550_guard_audit
mkdir -p "$A/orig"
PY="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $HOME/miniconda3/envs/neural_interface_challenge/bin/python -u"
for f in t2_d536.json t2_d539.json t2_w19_nom_D541.json t2_w19_nom_D543.json t2_w19_nom_D544.json t2_w19_nom_D546a_D546b.json t2_w19_nom_D547.json \
         t2_w19_nom_D538_D541_D542_D543_D544_D545_D546a_D546b_D547.json t2_d540a_check816.json t2_gain_carry_diag.json; do
  [ -f "$O/$f" ] && [ ! -f "$A/orig/$f" ] && cp -p "$O/$f" "$A/orig/$f"
done
cp -p "$O/t2_d540a_stack819.log" "$A/orig/" 2>/dev/null
rc=0
run() { local nm=$1; shift; echo "### $(date +%T) $nm"; $PY "$@" > "$A/$nm.log" 2>&1; local r=$?; grep -h "oracle guard" "$A/$nm.log" | head -3; echo "### $nm rc=$r"; [ $r = 0 ] || rc=1; }
run d536 scripts/local/t2_d536_read.py
run d539 scripts/local/t2_d539_read.py
run nom_all scripts/local/t2_w19_nom_read.py --items D538,D541,D542,D543,D544,D545,D546a,D546b,D547
run stack816 scripts/local/t2_d540a_stack_read.py --draws 816 --items D539b,D543,D545
run stack819 scripts/local/t2_d540a_stack_read.py --draws 819 --items D539b,D543,D545,D546b
run gaincarry scripts/local/t2_gain_carry_diag.py
echo "### done rc=$rc"
exit $rc
