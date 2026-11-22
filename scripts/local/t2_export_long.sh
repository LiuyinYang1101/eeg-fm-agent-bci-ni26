#!/usr/bin/env bash
# Long epochs (marker-relative) for the timing scan and the TSF-style spectral analysis — through the kit pipeline.
#   Scherer2015: [-1, 10] s around the TRIAL marker (fixation 0-3 s, cue at +3 s, imagery 3-10 s)
#   Dreyer2023 : [-4, 8] s around the Stimulus marker, keeping EOG + forearm EMG (which event the marker is
#                on is what the timing scan measures)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
export BENCHOPT_DATA_HOME=~/workspace/datasets/neural_interface_challenge/benchopt NUMPY_MADVISE_HUGEPAGE=0
W=~/workspace/datasets/neural_interface_challenge/t2_bci/windows_long
mkdir -p "$W"
[ -s "$W/scherer2015_m1p10/meta.json" ] || \
  $PY scripts/t2_export_windows.py --study Scherer2015Individually --no-download --start -1 --duration 11 --out "$W/scherer2015_m1p10"
[ -s "$W/dreyer2023_aux_m4p8/meta.json" ] || \
  $PY scripts/t2_export_windows.py --overlay dreyer2023 --no-download --picks eeg eog emg --start -4 --duration 12 --out "$W/dreyer2023_aux_m4p8"
echo "long exports done $(date -Is)"
