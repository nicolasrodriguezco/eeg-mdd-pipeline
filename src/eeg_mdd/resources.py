"""Uso seguro de CPU y RAM: elige el número de procesos según la memoria realmente libre.

Motivo: con 26 procesos y copias de los datos por proceso, el sistema se quedó sin
memoria (2026-11-04). Estas funciones limitan el paralelismo para dejar siempre un
margen al sistema operativo, y comparten los arreglos grandes por memmap en disco.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import psutil

RESERVE_GB = float(os.environ.get("EEG_RESERVE_GB", 4.0))   # margen para Windows y otros programas


def free_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def safe_n_jobs(per_job_gb: float, max_jobs: int | None = None, reserve_gb: float = RESERVE_GB) -> int:
    """Máximo de procesos que caben en la RAM libre menos el margen (al menos 1)."""
    cpus = max_jobs or max(1, (os.cpu_count() or 2) - 2)
    by_mem = int(max(0.0, free_gb() - reserve_gb) // max(per_job_gb, 1e-3))
    return max(1, min(cpus, by_mem))


def limit_threads() -> None:
    """Un hilo de BLAS por proceso (evita sobresuscripción cuando hay muchos procesos)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def as_memmap(arr: np.ndarray, path: str | Path, dtype=np.float32) -> np.ndarray:
    """Guarda ``arr`` en disco (si no existe o cambió de forma) y lo devuelve como memmap de solo lectura.

    Los procesos de joblib reciben la referencia al archivo, no una copia en RAM.
    """
    path = Path(path)
    if path.exists():
        mm = np.load(path, mmap_mode="r")
        if mm.shape == arr.shape and mm.dtype == np.dtype(dtype):
            return mm
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=dtype))
    return np.load(path, mmap_mode="r")
