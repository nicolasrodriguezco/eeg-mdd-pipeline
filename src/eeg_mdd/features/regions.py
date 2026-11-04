"""Mapa de regiones del montaje HydroCel GSN 129 y equivalentes 10-20.

Las regiones se definen con reglas fijas sobre las coordenadas del montaje
(posiciones normalizadas a la esfera unidad, x = derecha, y = anterior,
z = arriba). Así el mapa es reproducible y se documenta con una figura.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

MONTAGE = "GSN-HydroCel-129"
REGIONS = [f"{lobe}_{side}" for lobe in ("frontal", "central", "temporal", "parietal", "occipital")
           for side in ("L", "R")]

# Equivalentes 10-20 = electrodo HydroCel más cercano (ángulo mínimo en la esfera).
# Verificado contra el montaje de MNE en la Fase A.
EQUIV_1020 = {"F3": "E24", "F4": "E124", "F7": "E33", "F8": "E122", "Fz": "E11", "Cz": "Cz",
              "Pz": "E62", "P3": "E60", "P4": "E85", "P7": "E58", "P8": "E96", "O1": "E70",
              "O2": "E83", "CPz": "E55", "POz": "E72"}


def _sphere_center(p: np.ndarray) -> np.ndarray:
    """Centro de la esfera que mejor ajusta los puntos (mínimos cuadrados lineales)."""
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


def unit_positions(montage: str = MONTAGE) -> pd.DataFrame:
    m = mne.channels.make_standard_montage(montage)
    pos = m.get_positions()["ch_pos"]
    names = list(pos)
    p = np.array([pos[n] for n in names])
    p = p - _sphere_center(p)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return pd.DataFrame(p, index=names, columns=["x", "y", "z"])


# Anclas 10-10 por región (hemisferio izquierdo; el derecho es el espejo con número par).
_ANCHORS_L = {
    "frontal": ["Fp1", "AF3", "AF7", "F1", "F3", "F5", "F7", "F9"],
    "central": ["FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3"],
    "temporal": ["FT7", "T7", "TP7", "FT9", "T9", "TP9"],
    "parietal": ["CP5", "P1", "P3", "P5"],
    "occipital": ["P7", "PO3", "PO7", "O1", "P9", "PO9"],
}
MAX_POLAR_DEG = 115.0   # T9/T10 del 10-10 están a ~108° de Cz
_MIDLINE = ["Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "Iz"]
_EXCLUDED = ["Nz", "LPA", "RPA"]


def _mirror(name: str) -> str:
    """Fp1 -> Fp2, AF7 -> AF8, ... (número impar -> par siguiente)."""
    head = name.rstrip("0123456789")
    return f"{head}{int(name[len(head):]) + 1}"


def _ref_positions() -> pd.DataFrame:
    for name in ("colin27_1020", "standard_1020"):
        try:
            return unit_positions(name)
        except ValueError:
            continue
    raise RuntimeError("No hay montaje 10-20 de referencia en esta versión de MNE")


def anchors() -> dict[str, str]:
    """Nombre 10-10 -> región (o 'midline'/'excluded')."""
    out = {}
    for lobe, names in _ANCHORS_L.items():
        for n in names:
            out[n] = f"{lobe}_L"
            out[_mirror(n)] = f"{lobe}_R"
    out.update({n: "midline" for n in _MIDLINE})
    out.update({n: "excluded" for n in _EXCLUDED})
    return out


def assign_regions(montage: str = MONTAGE) -> pd.Series:
    """Región de cada electrodo = región del ancla 10-10 más cercana (en ángulo)."""
    egi = unit_positions(montage)
    ref = _ref_positions()
    anc = {k: v for k, v in anchors().items() if k in ref.index}
    A = ref.loc[list(anc)].to_numpy()
    labels = np.array(list(anc.values()))
    nearest = np.argmax(egi.to_numpy() @ A.T, axis=1)
    reg = pd.Series(labels[nearest], index=egi.index)
    # Anillo inferior (mejillas, cuello): más de MAX_POLAR_DEG desde el vértice (Cz).
    polar = np.degrees(np.arccos(np.clip(egi.to_numpy() @ egi.loc["Cz"].to_numpy(), -1, 1)))
    reg[polar > MAX_POLAR_DEG] = "excluded"
    return reg


def region_map(montage: str = MONTAGE) -> dict[str, list[str]]:
    reg = assign_regions(montage)
    return {r: reg.index[reg == r].tolist() for r in REGIONS}


def region_table(montage: str = MONTAGE) -> pd.DataFrame:
    pos = unit_positions(montage)
    pos["region"] = assign_regions(montage)
    return pos


def neighbors_of(center: str, k: int, montage: str = MONTAGE) -> list[str]:
    """``center`` y sus k vecinos más cercanos (para clústeres de ERP)."""
    pos = unit_positions(montage)
    d = np.arccos(np.clip(pos.to_numpy() @ pos.loc[center].to_numpy(), -1, 1))
    return [pos.index[i] for i in np.argsort(d)[: k + 1]]
