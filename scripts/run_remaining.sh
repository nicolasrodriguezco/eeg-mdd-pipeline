#!/usr/bin/env bash
# Cadena de CPU para lo que falta (se reanuda: omite los modelos ya calculados).
# El deep learning corre aparte en la GPU (scripts/08_deep.py).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/nicog/Anaconda3/envs/eeg-mdd/python.exe}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
$PY -W ignore scripts/07_models.py --n-jobs 28 --only families selection riemann control
$PY -W ignore scripts/07_models.py --n-jobs 28 --all-subjects --only primary
$PY -W ignore scripts/09_confounds.py --n-jobs 28
$PY -W ignore scripts/10_interpret.py --n-jobs 28
echo "CPU CHAIN DONE"
