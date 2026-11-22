#!/usr/bin/env bash
# Dreyer2023 (Codabench warm-up study) chain, CPU part — run under mem-guard after the kit download finished:
#   1. export the official warm-up windows (27 EEG, what Codabench feeds predict())
#   2. export the same windows keeping EOG (3) + forearm EMG (2): the channel-role study (sealed X has EMG/EOG)
#   3. classical ladder: kit split (= warm-up leaderboard protocol, pooled BA) and the sealed mirror
#      (runs 1-2 = calibration, runs 3-6 = "later", 5 subject groups)
#   4. channel-role ladder on the aux export (EEG / EOG / EMG / EEG+EOG / EEG+EMG / all) with fbts_lr
# GPU follow-ups (EEGNet) are queued separately through gpu-submit once these caches exist.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
export BENCHOPT_DATA_HOME=~/workspace/datasets/neural_interface_challenge/benchopt NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
W=~/workspace/datasets/neural_interface_challenge/t2_bci/windows
O=~/workspace/outputs/neural_interface_challenge/t2/baselines
mkdir -p "$O"
[ -s "$W/dreyer2023/meta.json" ]     || $PY scripts/t2_export_windows.py --overlay dreyer2023 --out "$W/dreyer2023"
[ -s "$W/dreyer2023_aux/meta.json" ] || $PY scripts/t2_export_windows.py --overlay dreyer2023 --picks eeg eog emg --out "$W/dreyer2023_aux"
$PY scripts/t2_baselines.py --cache "$W/dreyer2023" --protocol kit --models logvar_lda logbp_lda ts_lr fbts_lr --out "$O/dreyer_kit_classical.json"
$PY scripts/t2_baselines.py --cache "$W/dreyer2023" --protocol xsession --unit run --n-calib 2 --n-folds 5 \
    --models logvar_lda ts_lr fbts_lr --out "$O/dreyer_xsession_classical.json"
# channel groups from the aux export's own channel list (names never assumed)
read -r EEG EOG EMG < <($PY - "$W/dreyer2023_aux/meta.json" <<'PY'
import json, sys
ch = json.load(open(sys.argv[1]))["ch_names"]
eog = [c for c in ch if c.upper().startswith("EOG")]; emg = [c for c in ch if c.upper().startswith("EMG")]
eeg = [c for c in ch if c not in eog + emg]
print(",".join(eeg), ",".join(eog), ",".join(emg))
PY
)
echo "groups: EEG=$EEG | EOG=$EOG | EMG=$EMG"
for g in EEG EOG EMG EEG+EOG EEG+EMG EEG+EOG+EMG; do
  chans=$(echo "$g" | sed "s/EEG/$EEG/; s/EOG/$EOG/; s/EMG/$EMG/" | tr '+,' '  ')
  $PY scripts/t2_baselines.py --cache "$W/dreyer2023_aux" --protocol xsession --unit run --n-calib 2 --n-folds 5 \
      --models logbp_lda fbts_lr --channels $chans --out "$O/dreyer_aux_${g//+/_}_xsession.json"
done
echo "chain done $(date -Is)"
