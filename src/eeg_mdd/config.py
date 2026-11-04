"""Carga de la configuración y resolución de rutas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Lee el YAML y convierte las rutas de ``paths`` y ``clinical.file`` en absolutas."""
    path = Path(path) if path else DEFAULT_CONFIG
    root = Path(root) if root else REPO_ROOT
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: _abs(v, root) for k, v in cfg["paths"].items()}
    clinical = cfg["dataset"]["clinical"]
    if clinical.get("file"):
        clinical["file"] = _abs(clinical["file"], root)
    return cfg


def _abs(value: str | Path, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Crea las carpetas de salida (nunca toca ``raw_dir``)."""
    for key in ("interim_dir", "processed_dir", "reports_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
