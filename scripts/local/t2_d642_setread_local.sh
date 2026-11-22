#!/usr/bin/env bash
# T2-D642 FULL-CHAIN SET READ, the local chain (design pinned in docs/DECISIONS_T2.md before any number; the splicer
# scripts/local/t2_d642_setsplice.py, Track3-reviewed before any root is read). On D638's chosen cell <cell>:
#   identity <cell>  CPU (run under mem-guard, background): the NO-OP root id (every slot, set E's own FM rows written back) -> fbteach (the
#                    cell's base root; NI26_LAB_FMSTART=dump records the FM starts) -> lpteach -> lp root (the cell's routes + final students)
#                    -> check-id (fb / lp teachers == the cell's BITWISE) -> the chronological mix (dump records the final mix's FM start) ==
#                    D638's mix of the cell BITWISE -> '[d642 identity] PASS' in outputs/t2_d642/setread/identity.log.
#   teach <cell>     CPU (mem-guard, background; refuses unless the identity PASSED and members.json nominates a set): per tag bs cs bp04
#                    cp04 bp10 cp10 (3 at a time): fmspl -> fbteach (p tags: NI26_LAB_FMSTART=freeze, Track3 Q2) -> lpteach -> lp root -> p1t.
#   students <cell> [--dry]  ONE gpu-submit job (the manager sequences the 5090): the final P1 students of each tag's SCORED group only
#                    (root s: ss8 x seeds 880-882; roots p04 / p10: rt1_<X> per spliced fold; D638's commands; C cells --user-balance) ->
#                    the students splice -> '### done d642 tag <tag>'.
#   [READ=first|replicate] mem-guard bash scripts/local/t2_d642_setread_local.sh identity|teach|students <cell> [--dry]   (W26: READ=replicate on the decision cell)
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?identity|teach|students|d654prep|stackprep|d655prep}; cell=${2:?the D638 cell (bw21 or bw23: the manager runs BOTH; each under setread/<cell>/)}
case $cell in bw21|bw23|cw21|bw21fm40|bw21fm55) ;; *) echo "bad cell $cell"; exit 2 ;; esac   # bw21fm55 = T2-D658   # cw21 = T2-D650; bw21fm40 = T2-D655 (the FM share 0.40)
BCELL=${cell%fm40}; BCELL=${BCELL%fm55}; FMX=""; [ "$cell" = bw21fm40 ] && FMX=0.40; [ "$cell" = bw21fm55 ] && FMX=0.55   # T2-D658   # T2-D655: the D638 CNN cell under it, and the teacher's FM share
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_lab.py scripts/local/t2_lab_ideas.py scripts/local/t2_d540a_stack_read.py scripts/local/t2_d642_setsplice.py scripts/local/t2_d642_member_read.py scripts/local/t2_d642_setread_local.sh)" ] || { echo "commit the D642 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
SR0=$O/t2_d642/setread                              # the shared input fmpaths_setE.json
SR=$SR0/$cell                                       # every file of this cell's set read; roots t2_d642_<fb|lp><tag>_<cell>_860
mkdir -p "$SR"
for v in $(compgen -e | grep '^NI26_'); do unset "$v"; done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
[ -f "$SR0/fmpaths_setE.json" ] || { echo "no $SR0/fmpaths_setE.json (t2_lab.py fmpaths --set E --base w20)"; exit 3; }
[ -f "$O/t2_d638/mix_${BCELL}_chrono.npz" ] && grep -q '"b_best"' "$O/t2_d638/result.json" || { echo "D638's read has not run (no mix / result.json)"; exit 3; }
FB="--set E --fm-rows own --proto 0.5"
LP="--set E --alpha 0.5 --k 10"

teach() {   # tag [FMSTART spec] -> fmspl, fbteach, lpteach, lp root (+ p1t unless id); every step logged to $SR/teach_<tag>.log
  local t=$1 st=${2:-} L="$SR/teach_$1.log"
  case "${t:0:1}" in                                   # T2-D654: the transform tags 2 / 3 / 5 / 6 were written by `d654prep` (t2_d654_viewmix fmspl)
    2|3|5|6|o|z) [ -f "$SR/fmspl_$t/manifest.json" ] || { echo "T2-D654 / stack: $t has no viewmix splice (run d654prep / stackprep)" >> "$L"; return 1; } ;;
    *) if [ -n "$FMX" ]; then [ -L "$SR/fmspl_$t" ] && [ -f "$SR/fmspl_$t/manifest.json" ] || { echo "T2-D655: $t needs d655prep's link" >> "$L"; return 1; }
       else $PY -u scripts/local/t2_d642_setsplice.py fmspl "$t" "$cell" >> "$L" 2>&1 || return 1; fi ;;
  esac
  env NI26_LAB_FMSPLICE="$SR/fmspl_$t" $PY -u scripts/local/t2_lab.py fmpaths --set E --base w20 --out "$SR/fmpaths_$t.json" >> "$L" 2>&1 || return 1
  $PY -u scripts/local/t2_d642_setsplice.py check-paths "$t" "$cell" >> "$L" 2>&1 || return 1   # Track3 LOW 2: no silent fallback
  env NI26_LAB_FMSPLICE="$SR/fmspl_$t" ${st:+NI26_LAB_FMSTART=$st} NI26_D559_ROOT="t2_d638_${BCELL}_860" NI26_D559_CKS=60,100,150 \
    $PY -u scripts/local/t2_lab.py fbteach $FB ${FMX:+--tw-fm $FMX} --out "$O/t2_d642_fb${t}_${cell}_860/teacher_p1" >> "$L" 2>&1 || return 1
  env NI26_LAB_FMSPLICE="$SR/fmspl_$t" $PY -u scripts/local/t2_lab.py lpteach $LP --out "$O/t2_d642_lp${t}_${cell}_860/teacher_p1" \
    --src "$O/t2_d642_fb${t}_${cell}_860/teacher_p1" >> "$L" 2>&1 || return 1
  $PY -u scripts/local/t2_d642_setsplice.py lproot "$t" "$cell" >> "$L" 2>&1 || return 1
  [ "$t" = id ] || $PY -u scripts/local/t2_d642_setsplice.py p1t "$t" "$cell" >> "$L" 2>&1 || return 1
  echo "### done d642 teach $t $(date +%T)" >> "$L"
}

if [ "$mode" = identity ]; then
  [ ! -e "$SR/fmspl_id" ] && [ ! -e "$O/t2_d642_fbid_${cell}_860" ] && [ ! -e "$O/t2_d642_lpid_${cell}_860" ] && [ ! -e "$SR/fmstart.npz" ] || { echo "an id root exists: never overwrite"; exit 3; }
  L=$SR/identity.log
  echo "### $(date +%T) d642 identity on cell $cell (git $(git rev-parse --short HEAD))" >> "$L"
  teach id "dump:$SR/fmstart.npz" || { echo "### d642 identity FAILED (teach) $(date +%T)" >> "$L"; exit 7; }
  $PY -u scripts/local/t2_d642_setsplice.py check-id "$cell" >> "$L" 2>&1 || { echo "### d642 identity STOP (teachers)" >> "$L"; exit 9; }
  env NI26_LAB_FMSPLICE="$SR/fmspl_id" NI26_LAB_FMSTART="dump:$SR/fmstart.npz" NI26_LAB_PRE_R=1 NI26_D559_ROOT=t2_d642_lpid_${cell}_860 NI26_D559_CKS=60,100,150 \
    $PY -u scripts/local/t2_lab.py mix --set E --base w20 --idea w20_fb --out "$SR/mix_id_chrono.npz" >> "$SR/mix_id.log" 2>&1 || { echo "### d642 identity FAILED (mix)" >> "$L"; exit 7; }
  $PY - "$SR/mix_id_chrono.npz" "$O/t2_d638/mix_${cell}_chrono.npz" "$SR/fmstart.npz" >> "$L" 2>&1 <<'PYEOF' || { echo "### d642 identity STOP (mix)" >> "$L"; exit 9; }
import sys, numpy as np
a, b, st = (np.load(p) for p in sys.argv[1:4])
ks = sorted(k for k in b.files if k.startswith(("ix_", "mx_")))
assert ks and sorted(k for k in a.files if k.startswith(("ix_", "mx_"))) == ks, "mix keys differ"
bad = [k for k in ks if not np.array_equal(a[k], b[k])]
n_fb, n_mfb, n_mix = (sum(k.startswith(p_) for k in st.files) for p_ in ("fbteach_fb_", "mix_fb_", "mix_final_"))
assert n_fb == n_mfb == n_mix == 30 and len(st.files) == 90, ("FM-start records (fold x draw)", n_fb, n_mfb, n_mix, len(st.files))
assert not bad, f"the id mix != D638's mix of the cell: {bad[:4]}"
print(f"[d642 identity] PASS: the no-op root's chronological mix == D638's mix of the cell BITWISE ({len(ks) // 2} fold x draw arrays); "
      f"FM starts recorded (fold x draw): {n_fb} fbteach_fb + {n_mfb} mix_fb + {n_mix} mix_final", flush=True)
PYEOF
  echo "### done d642 identity $(date +%T)" >> "$L"
  exit 0
fi

grep -q "\[d642 identity\] PASS: the no-op root's chronological mix" "$SR0/$BCELL/identity.log" 2>/dev/null || { echo "the identity has not PASSED"; exit 3; }   # T2-D655: the base cell's
READ=${READ:-first}                                 # W26 (D642 x7): READ=replicate = the second read, T vs C' (tags t* / d*), the decision cell only
case $READ in first) TAGS="bs cs bp04 cp04 bp10 cp10" ;; replicate) TAGS="ts ds tp04 dp04 tp10 dp10" ;; w27) TAGS="gs es gp04 ep04 gp10 ep10" ;;
  all6) TAGS="hs ks hp04 kp04 hp10 kp10" ;; all6rep) TAGS="ms ns mp04 np04 mp10 np10" ;;   # D642 x9 ALL6 (bw21 only)
  d646) TAGS="xs ys xp04 yp04 xp10 yp10" ;;   # T2-D646: the third ALL6 read (bw21 only)
  d648) TAGS="vs ws vp04 wp04 vp10 wp10" ;;   # T2-D648: the fourth ALL6 read (bw21 only)
  d650) TAGS="ks kp04 kp10" ;;   # T2-D650: the ALL6 balanced FM (k) on cell cw21 (vs bw21's k mixes)
  d654) TAGS="2s 3s 2p04 3p04 2p10 3p10" ;;   # T2-D654 read 1: the v1 vs the picked-arm transform of the 1600 base splice (1*)
  d654rep) TAGS="5s 6s 5p04 6p04 5p10 6p10" ;;   # T2-D654 read 2: the same on the 1700 base splice (4*)
  d655) TAGS="ns np04 np10 ys yp04 yp10 ws wp04 wp10" ;;
  w30fm55) TAGS="qs qp04 qp10" ;;   # T2-D658: W30's q tags (V at 950) on the cell bw21fm55   # T2-D655: the three W25 ALL6 roots' balanced tags on the cell bw21fm40 (the FM share 0.40)
  stack) TAGS="os zs op04 zp04 op10 zp10" ;;   # THE STACK READ (the owner's rule 3): C at 1800 (v1) vs the stack arm at 1800 (+ the D654 arm if included)
  d649) TAGS="fs as fp04 ap04 fp10 ap10" ;;   # T2-D649 (W29) read 1: C vs G_picked at 1200 + slot (bw21; both arms new)
  d649rep) TAGS="js is jp04 ip04 jp10 ip10" ;;   # T2-D649 read 2 (the replicate): C vs G_picked at 1300 + slot
  w30) TAGS="qs qp04 qp10" ;;   # T2-D652 (W30) read 1: V at 950 (its base n = all6rep's C', already on bw21)
  w30rep) TAGS="rs rp04 rp10" ;;   # T2-D652 read 2: V at 1100 (its base w = D648's W, already on bw21)
  *) echo "bad READ $READ"; exit 2 ;; esac
[ "$READ" != w27 ] || $PY -c "import json,sys; m=json.load(open('$O/t2_d642/members_w27.json')); sys.exit(0 if m['entering'] else 4)" || { echo "W27: no slot entered"; exit 4; }
[ "$READ" = w27 ] || $PY -c "import json,sys; m=json.load(open('$O/t2_d642/members.json')); sys.exit(0 if m['nominated'] and m['entering'] else 4)" || { echo "no set nominated (members.json)"; exit 4; }

if [ "$mode" = d654prep ]; then                      # T2-D654 stage 2: the base splice (1 / 4) + its v1 and picked-arm transforms (2, 3 / 5, 6)
  case $READ in d654) B=1; V=2; A=3 ;; d654rep) B=4; V=5; A=6 ;; *) echo "d654prep needs READ=d654|d654rep"; exit 2 ;; esac
  [ "$cell" = bw21 ] || { echo "D654 on bw21 only"; exit 2; }
  ARM=$($PY -c "import json; print(json.load(open('$O/t2_d654/members_d654_d654a_1600.json'))['pick'] or '')")
  [ -n "$ARM" ] || { echo "T2-D654: no stage-1 pick (members_d654_d654a_1600.json): D654 closed, no stage 2"; exit 4; }
  for r in s p04 p10; do
    for t in $B$r $V$r $A$r; do [ ! -e "$SR/fmspl_$t" ] || { echo "fmspl_$t exists: never overwrite"; exit 3; }; done
    $PY -u scripts/local/t2_d642_setsplice.py fmspl "$B$r" "$cell" || exit 5
    $PY -u scripts/local/t2_d654_viewmix.py fmspl "$V$r" "$B$r" v1 "$cell" || exit 5
    $PY -u scripts/local/t2_d654_viewmix.py fmspl "$A$r" "$B$r" "$ARM" "$cell" || exit 5
  done
  echo "d654prep ($READ): base $B, v1 $V, arm $A = $ARM: 9 splices written"
  exit 0
fi

if [ "$mode" = stackprep ]; then                     # THE STACK READ: stack_spec.json from the included parts, the base splices 7 / 8, transforms o / z
  [ "$READ" = stack ] && [ "$cell" = bw21 ] || { echo "stackprep needs READ=stack on bw21"; exit 2; }
  $PY - "$SR" "$O" <<'PYEOF' || exit 4
import json, sys
from pathlib import Path
SR, O = Path(sys.argv[1]), Path(sys.argv[2])
inc = {k: json.loads((SR / f"decide_{d}.json").read_text())["included"] if (SR / f"decide_{d}.json").exists() else False
       for k, d in (("W29", "d649"), ("W30", "d652"), ("D654", "d654"))}
fm = {k for k in ("W29", "W30") if inc[k]}
assert sum(inc.values()) >= 2, f"the stack read needs >= 2 included parts: {inc}"
arm = {frozenset(): "C", frozenset({"W29"}): "G", frozenset({"W30"}): "V", frozenset({"W29", "W30"}): "J"}[frozenset(fm)]
view = json.loads((O / "t2_d654/members_d654_d654a_1600.json").read_text())["pick"] if inc["D654"] else "v1"
assert arm != "C" or view != "v1", (arm, view)
(SR / "stack_spec.json").write_text(json.dumps(dict(included=inc, arm=arm, view=view), indent=1))
print(f"stack_spec: parts {sorted(k for k, v in inc.items() if v)} -> member arm {arm}, view arm {view}")
PYEOF
  VIEW=$($PY -c "import json; print(json.load(open('$SR/stack_spec.json'))['view'])")
  for r in s p04 p10; do
    for t in 7$r 8$r o$r z$r; do [ ! -e "$SR/fmspl_$t" ] || { echo "fmspl_$t exists: never overwrite"; exit 3; }; done
    $PY -u scripts/local/t2_d642_setsplice.py fmspl "7$r" "$cell" || exit 5
    $PY -u scripts/local/t2_d642_setsplice.py fmspl "8$r" "$cell" || exit 5
    $PY -u scripts/local/t2_d654_viewmix.py fmspl "o$r" "7$r" v1 "$cell" || exit 5
    $PY -u scripts/local/t2_d654_viewmix.py fmspl "z$r" "8$r" "$VIEW" "$cell" || exit 5
  done
  echo "stackprep: 12 splices written ($(cat $SR/stack_spec.json | tr -d '\n '))"
  exit 0
fi

if [ "$mode" = d655prep ]; then                      # T2-D655: the cell bw21fm40 = bw21's FM splices + FM starts + identity, linked (the same files)
  { [ "$READ" = d655 ] && [ "$cell" = bw21fm40 ]; } || { [ "$READ" = w30fm55 ] && [ "$cell" = bw21fm55 ]; } || { echo "d655prep needs READ=d655 on bw21fm40 (or T2-D658: READ=w30fm55 on bw21fm55)"; exit 2; }
  B=$SR0/bw21
  for t in $TAGS; do [ ! -e "$SR/fmspl_$t" ] || { echo "fmspl_$t exists: never overwrite"; exit 3; }; [ -f "$B/fmspl_$t/manifest.json" ] || { echo "bw21 has no fmspl_$t"; exit 3; }; done
  for f in fmstart.npz identity.log; do [ -e "$SR/$f" ] || ln -s "../bw21/$f" "$SR/$f"; done
  for t in $TAGS; do ln -s "../bw21/fmspl_$t" "$SR/fmspl_$t"; done
  echo "d655prep: $(ls -d $SR/fmspl_* | wc -l) splices linked from bw21 (+ fmstart.npz, identity.log)"
  exit 0
fi

if [ "$mode" = teach ]; then
  for t in $TAGS; do [ ! -e "$O/t2_d642_fb${t}_${cell}_860" ] && [ ! -e "$O/t2_d642_lp${t}_${cell}_860" ] || { echo "a $t root exists: never overwrite"; exit 3; }
    case "${t:0:1}" in 2|3|5|6|o|z) [ -f "$SR/fmspl_$t/manifest.json" ] || { echo "T2-D654 / stack: $t needs d654prep / stackprep first"; exit 3; } ;; *) if [ -n "$FMX" ]; then [ -L "$SR/fmspl_$t" ] || { echo "T2-D655: $t needs d655prep first"; exit 3; }; else [ ! -e "$SR/fmspl_$t" ] || { echo "fmspl_$t exists: never overwrite"; exit 3; }; fi ;; esac; done
  pids=()
  for t in $TAGS; do
    st=""; [ "${t:1:1}" = p ] && st="freeze:$SR/fmstart.npz"
    ( teach "$t" "$st" || echo "### d642 teach $t FAILED $(date +%T)" >> "$SR/teach_$t.log" ) &
    pids+=($!)
    [ ${#pids[@]} -ge 3 ] && { wait "${pids[0]}"; pids=("${pids[@]:1}"); }
  done
  wait
  for t in $TAGS; do grep -q "### done d642 teach $t " "$SR/teach_$t.log" || { echo "teach $t FAILED"; exit 7; }; done
  echo "d642 teach ($READ): all 6 tags done"
  exit 0
fi

[ "$mode" = students ] || { echo "bad mode $mode"; exit 2; }
for t in $TAGS; do grep -q "### done d642 teach $t " "$SR/teach_$t.log" || { echo "teach $t not done"; exit 3; }; [ ! -e "$SR/p1/$t" ] || { echo "$SR/p1/$t exists"; exit 3; }; done
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
COMMON="--instream-all --instream-sched 60,100,150 --instream-sel true,confR --instream-reps 1 --instream-rep-sel confR"   # reps 1: rep 0 (isSpr, the only one read) unchanged, GATED below
declare -A M=([sh]="--models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr" [ifn]="--models ifnet --aligns ea --recipe r1_pulx_vp2_e180_vr")
X=""; [ "${cell:0:1}" = c ] && X="--user-balance"
SSEEDS=880,881,882   # every read, T2-D646 included (Track3 / the manager: fresh seeds would retrain the students' base nets off the routed part's)          # the cell's nets (D638: C = --user-balance, t2p_bench names it __ubw)
job=$(mktemp "$O/t2_d642_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; L='$SR/chain.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "for v in \$(compgen -e | grep '^NI26_'); do unset \"\$v\"; done"
  echo "echo \"### \$(date +%T) d642 students start, cell $cell (git \$(git rev-parse --short HEAD))\" >> \"\$L\""
  echo "mkdir -p '$SR/p1'"                           # the gate's log lives there (job 119 stopped on its absence before any bench ran)
  # the reps-1 GATE (the manager / Track3's optional speed-up, only if rep 0 stays the same): D638's own final student of this cell (rt1_P04,
  # fold 0, seed 880, its teacher) re-run at --instream-reps 1 == D638's reps-3 isSpr: ShallowNet BITWISE, IFNet <= 1e-3 with 100 % argmax
  for fam in sh ifn; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$SR/p1/reps_gate_$READ/$fam/runs' ${M[$fam]} --protocols rt1_P04 --seeds 880 $X $COMMON --instream-pl '$O/t2_d638/p1t/final_$BCELL/f0/{protocol}__s{seed}.npz' >> '$SR/p1/reps_gate_$READ.log' 2>&1 &"
  done
  echo "wait"
  echo "$PY - '$SR/p1/reps_gate_$READ' '$O/t2_d638/p1/final_$BCELL/f0' '$X' >> \"\$L\" 2>&1 <<'PYEOF' || { echo '### d642 STOP: the reps-1 gate' >> \"\$L\"; exit 9; }"
  echo "import sys, numpy as np"
  echo "a, b, x = sys.argv[1], sys.argv[2], '__ubw' if sys.argv[3] else ''"
  echo "r = {}"
  echo "for fam, stem in (('sh', 'shallow__ea__n00__r1_pulx_vp2_e180_vr'), ('ifn', 'ifnet__ea__n00__r1_pulx_vp2_e180_vr')):"
  echo "    za, zb = (np.load(f'{d}/{fam}/runs/rt1_P04__{stem}isSpr{x}__s880.npz') for d in (a, b))"
  echo "    assert np.array_equal(za['test_idx'], zb['test_idx']), fam"
  echo "    pa, pb = za['prob_last'].astype(np.float64), zb['prob_last'].astype(np.float64)"
  echo "    r[fam] = (float(np.abs(pa - pb).max()), float(np.mean(pa.argmax(1) == pb.argmax(1))))"
  echo "print(f'[d642 reps gate] rt1_P04 f0 s880 at reps 1 vs D638 reps 3: {r}', flush=True)"
  echo "assert r['sh'][0] == 0.0 and r['ifn'][0] <= 1e-3 and r['ifn'][1] == 1.0, r"
  echo "print('[d642 reps gate] PASS', flush=True)"
  echo "PYEOF"
  for t in $TAGS; do
    T=$SR/p1t/$t; W=$SR/p1/$t
    echo "tag_$t() {"
    echo "  echo \"### \$(date +%T) d642 tag $t start\" >> \"\$L\""
    for fam in sh ifn; do
      if [ "${t:1:1}" = s ]; then
        for i in 0 1 2; do
          echo "  $SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$W/ss8/$fam/runs' --shard $i --nshards 3 ${M[$fam]} --protocols ss8 --seeds $SSEEDS $X $COMMON --instream-pl '$T/ss8/{protocol}__s{seed}.npz' >> '$W.log' 2>&1 &"
        done
      else
        n=0; xu=${t:1}; xu=${xu^^}
        for f in 0 1 2 4 5 6 7 8; do
          echo "  $SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$W/f$f/$fam/runs' ${M[$fam]} --protocols rt1_$xu --seeds $SSEEDS $X $COMMON --instream-pl '$T/f$f/{protocol}__s{seed}.npz' >> '$W.log' 2>&1 &"
          n=$((n + 1)); [ $((n % 4)) = 0 ] && echo "  wait"
        done
      fi
      echo "  wait"
    done
    echo "  [ \$(cat '$W.log' | grep -c Traceback) -eq 0 ] || return 1"
    echo "  $PY -u scripts/local/t2_d642_setsplice.py students $t $cell >> \"\$L\" 2>&1 || return 1"
    echo "  echo \"### done d642 tag $t \$(date +%T)\" >> \"\$L\""
    echo "}"
    echo "mkdir -p '$W'; tag_$t || { rc=7; echo \"### d642 tag $t FAILED \$(date +%T)\" >> \"\$L\"; }"
  done
  echo "echo \"### done d642 students all \$(date +%T) rc=\$rc\" >> \"\$L\"; exit \$rc"
} > "$job"
if [ "${3:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d642-setread-$cell-$READ" bash "$job"
