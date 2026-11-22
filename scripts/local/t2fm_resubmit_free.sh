#!/usr/bin/env bash
# T2-FM: the resubmissions after the sofia FILE cap of 2026-11-06 ≈ 08:52 (run ONLY on the manager's "free", >= 60k files free; no s1 sync).
# What died: J-CPT A0 1568310 (at its 10k save; ck 2k / 5k exist), the jA fine-tunes 1568313–20 (arm A 1568309 COMPLETED: ck 2k / 5k / 10k / 20k
# exist), the jA0 fine-tunes (cancelled), ST-base 1568346 (at its 25k state write; state-cpt.pth of step 20k + ck 22.5k exist).
#   bash scripts/local/t2fm_resubmit_free.sh a0|ja|ja0|stbase|ar_contingency <AR tag>   (each section prints its job ids)
set -euo pipefail
cd "$(dirname "$0")/../.."
S1=$S1_ROOT
X="--exclude=acc006,acc009,acc010"
JSNAP=$S1/code/ni26/snap_20261105T220423Z_c38bb0da4c68        # the J-CPT CPTs' snapshot (A / A0 / AR4 / AR2 all ran from it)
FSNAP=$S1/code/ni26/snap_20261105T221555Z_95f72a6869bd        # the ST-base smoke's + run's snapshot
CPT_ARGS=(--fm st_large --steps 20000 --batch 64 --accum 8 --lr 2e-5 --schedule const --wd 1e-5 --mask-ratio 0.75 --eval-every 2500 --token-scale)
case "${1:?a0|ja|ja0|stbase|ar_contingency}" in
  a0)      # the CPT, default nice (critical path): 10k / 20k only (run 1's 2k / 5k stay), resumable; 1.69 s/step x 20k ≈ 9.4 h -> 11 h wall
    S1_EXPORT=RUN_NAME=ni26-t2p-jcpt-A0 S1_PURPOSE="T2-D354 J-CPT arm A0 (task-only counterfactual) RERUN for ckpts 10k/20k (run 1 1568310 died at its 10k save: the S1 file cap 2026-11-06 08:52; its 2k/5k stay); ST-L MAE, 20k steps, lr 2e-5 const; feeds the pick: which CPT checkpoint the shipped ST-L members use; T2 GO 730, T2-D353" \
      s1 submit ni26 $JSNAP/tracks/t2_bci/sofia/t2p_jcpt.slurm --time=11:00:00 $X --job-name=ni26-t2p-jcpt-A0 -- \
      tracks/t2_bci/sofia/jcpt/jcpt_A0_spec.json "${CPT_ARGS[@]}" --save-at 10000,20000 --save-tag jA0 --state-every 2500 2>&1 \
      | grep -E 'SUBMITTED|refus|error|!!' | cut -c1-110 ;;
  ja)      # arm A's 8 fine-tunes, no dependency (its checkpoints exist), nice 300
    XOPTS="$X --nice=300" mem-guard bash tracks/t2_bci/sofia/launch_jcpt_ft.sh GO - ni26-t2p-jcpt-A jA ;;
  ja0)     # A0's 2k / 5k fine-tunes now; 10k / 20k afterok on the A0 rerun (job id = $2)
    a0=${2:?the A0 rerun job id}
    STEPS="02000 05000" XOPTS="$X --nice=300" mem-guard bash tracks/t2_bci/sofia/launch_jcpt_ft.sh GO - ni26-t2p-jcpt-A0 jA0
    STEPS="10000 20000" XOPTS="$X --nice=300" mem-guard bash tracks/t2_bci/sofia/launch_jcpt_ft.sh GO "$a0" ni26-t2p-jcpt-A0 jA0 ;;
  stbase)  # ST-base resumes at step 20,000 (state-cpt.pth); 70k steps x 0.64 s ≈ 12.4 h + start-up -> 14 h wall; same recipe (asserted)
    S1_SMOKE=1566601 S1_EXPORT=NI26_CODE=$FSNAP,INHOUSE_SRC=$S1/code/fm/snap_20261105T221250Z_457c01898e64/src,FM_SPLIT=$S1/data/fm/ni26fm/ni26fm_split.parquet,FM_SPLIT_SHA256=2bfcea8769dfc752c91a27d856b101e95ba79be09d55500c939768e5a9d42fb7,RUN_NAME=fm-t2-stbase \
      S1_PURPOSE="T2-FM-D2 / T2-D356 ST-EEGFormer-base FROM SCRATCH, RESUME at step 20k (1568346 died at its 25k state write: the S1 file cap 2026-11-06 08:52), to 90k; tag fm, never the other project's allocation; feeds the pick of an FM member trained from scratch (T2 GO 730)" \
      s1 submit fm $FSNAP/tracks/t2_bci/sofia/t2p_stbase_fm.slurm --time=14:00:00 $X --nice=300 --job-name=fm-t2-stbase -- \
      --fm st_base --random-init --lr 1.5e-4 --wd 0.05 --mask-ratio 0.75 --betas 0.9,0.999 --steps 90000 --batch 64 --accum 8 --warmup-frac 0.05 \
      --token-scale --eval-every 5000 --save-at 22500,45000,67500,90000 --save-tag stb --state-every 5000 --store-workers 20 --store-index-workers 16 \
      --resume 2>&1 | grep -E 'SUBMITTED|refus|error|!!' | cut -c1-110 ;;
  ar_contingency)   # ONLY if an AR CPT died at its 20k save: its 2k / 5k / 10k fine-tunes without a dependency (the 20k one needs a rerun)
    t=${2:?jAR4|jAR2}
    STEPS="02000 05000 10000" XOPTS="$X --nice=300" mem-guard bash tracks/t2_bci/sofia/launch_jcpt_ft.sh GO - ni26-t2p-jcpt-${t#j} "$t" ;;
esac
