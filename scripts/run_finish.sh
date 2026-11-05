#!/usr/bin/env bash
# Pasos finales de CPU (se reanuda). Espera a que termine un 07_models.py ya en curso.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/nicog/Anaconda3/envs/eeg-mdd/python.exe}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore EEG_RESERVE_GB=${EEG_RESERVE_GB:-7}
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '07_models.py' }) { exit 0 } else { exit 1 }"; do sleep 30; done
echo "07_models previo terminado: $(date)"
$PY -W ignore scripts/07_models.py --n-jobs 28 --only riemann control
$PY -W ignore scripts/07_models.py --n-jobs 28 --all-subjects --only primary
$PY -W ignore scripts/09_confounds.py --n-jobs 28
$PY -W ignore scripts/10_interpret.py --n-jobs 28
echo "CPU CHAIN DONE $(date)"
