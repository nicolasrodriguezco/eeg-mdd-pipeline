"""Búsqueda y lectura de archivos de EEG."""

from __future__ import annotations

import re
from pathlib import Path

import mne


def find_eeg_files(raw_dir: str | Path, extensions: list[str]) -> list[Path]:
    """Lista los archivos de EEG bajo ``raw_dir``.

    Los registros ``.mff`` de EGI son carpetas, así que también se aceptan
    directorios con esa extensión.
    """
    raw_dir = Path(raw_dir)
    exts = {e.lower() for e in extensions}
    found: list[Path] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        if p.is_dir() and p.suffix.lower() != ".mff":
            continue
        # Evitar archivos dentro de una carpeta .mff ya contada.
        if any(parent.suffix.lower() == ".mff" for parent in p.parents):
            continue
        found.append(p)
    return found


def extract_subject_id(path: str | Path, pattern: str) -> str | None:
    """Saca el ID del sujeto del nombre del archivo con una expresión regular.

    Devuelve ``None`` si no hay coincidencia: esos archivos se revisan a mano,
    nunca se adivina el ID.
    """
    match = re.search(pattern, Path(path).name)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def load_raw(path: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    """Lee un registro según su extensión."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".mff", ".raw"):
        return mne.io.read_raw_egi(path, preload=preload, verbose="error")
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="error")
    if ext in (".edf", ".bdf"):
        reader = mne.io.read_raw_edf if ext == ".edf" else mne.io.read_raw_bdf
        return reader(path, preload=preload, verbose="error")
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="error")
    if ext == ".mat":
        return _read_mat(path)
    raise ValueError(f"Extensión no soportada: {ext} ({path})")


def _read_mat(path: Path) -> mne.io.BaseRaw:
    """Lector para archivos .mat.

    La estructura de un .mat depende de cómo se exportó, así que este lector
    se escribe después de inspeccionar un archivo real, por ejemplo con
    ``scipy.io.loadmat(path)`` o ``h5py`` si es MATLAB v7.3.
    """
    raise NotImplementedError(
        f"Falta implementar la lectura de .mat para {path.name}. "
        "Inspecciona un archivo con scipy.io.loadmat y construye un mne.io.RawArray "
        "con nombres de canal, frecuencia de muestreo y eventos."
    )
