"""Deep learning por ensayo (braindecode) con validación por sujeto.

Por pliegue externo:
  1. Los sujetos de entrenamiento se dividen (estratificado, por sujeto) en
     entrenamiento interno y validación (20 %) para early stopping.
  2. Escalado por canal con estadísticos SOLO del entrenamiento interno.
  3. Aumentación simple: desplazamiento temporal, escala de amplitud y ruido.
  4. Las probabilidades por ensayo de los sujetos de prueba se promedian por sujeto.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..evaluation.cv import assert_disjoint


@dataclass
class DLConfig:
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-3
    val_frac_splits: int = 5          # 1/5 de los sujetos de entrenamiento -> validación
    shift_max: int = 10               # muestras (40 ms a 250 Hz)
    noise_sd: float = 0.1
    scale_range: tuple = (0.9, 1.1)
    seed: int = 0


def build_model(name: str, n_chans: int, n_times: int, sfreq: float) -> torch.nn.Module:
    from braindecode.models import BDTCN, EEGNet, ShallowFBCSPNet

    if name == "EEGNet":
        return EEGNet(n_chans=n_chans, n_outputs=2, n_times=n_times)
    if name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=2, n_times=n_times, final_conv_length="auto")
    if name == "TCN":
        return BDTCN(n_chans=n_chans, n_outputs=2, n_times=n_times, sfreq=sfreq, n_blocks=3, n_filters=32,
                     kernel_size=5, drop_prob=0.3)
    raise ValueError(name)


def _forward(model, xb):
    out = model(xb)
    if out.ndim == 3:            # salidas densas en el tiempo -> promedio
        out = out.mean(dim=-1)
    return out


def _augment(xb: torch.Tensor, cfg: DLConfig, gen: torch.Generator) -> torch.Tensor:
    n = xb.shape[0]
    shifts = torch.randint(-cfg.shift_max, cfg.shift_max + 1, (n,), generator=gen)
    xb = torch.stack([torch.roll(x, int(s), dims=-1) for x, s in zip(xb, shifts)])
    lo, hi = cfg.scale_range
    scale = lo + (hi - lo) * torch.rand(n, 1, 1, generator=gen)
    noise = cfg.noise_sd * torch.randn(xb.shape, generator=gen)
    return xb * scale + noise


def _channel_stats(X: np.ndarray, idx: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Media y DE por canal sobre las filas ``idx``, por bloques (sin copiar todo el arreglo)."""
    n, s1, s2 = 0, 0.0, 0.0
    for i in range(0, len(idx), chunk):
        b = np.asarray(X[np.sort(idx[i:i + chunk])], dtype=np.float64)
        n += b.shape[0] * b.shape[2]
        s1 = s1 + b.sum(axis=(0, 2))
        s2 = s2 + (b ** 2).sum(axis=(0, 2))
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-12
    return mu[None, :, None].astype(np.float32), sd[None, :, None].astype(np.float32)


def _batch(X, rows, mu, sd) -> torch.Tensor:
    order = np.argsort(rows)
    b = np.asarray(X[rows[order]], dtype=np.float32)
    b = b[np.argsort(order)]                      # vuelve al orden pedido
    return torch.from_numpy((b - mu) / sd)


def _predict(model, X: np.ndarray, rows: np.ndarray, mu, sd, device, batch=512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            xb = _batch(X, rows[i:i + batch], mu, sd).to(device)
            out.append(torch.softmax(_forward(model, xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _subject_auc(prob, y, g) -> float:
    import pandas as pd

    d = pd.DataFrame({"g": g, "y": y, "p": prob}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return roc_auc_score(d["y"], d["p"]) if d["y"].nunique() == 2 else np.nan


def train_fold(name: str, X: np.ndarray, y: np.ndarray, g: np.ndarray, tr: np.ndarray, te: np.ndarray,
               sfreq: float, cfg: DLConfig, device: str = "cuda") -> dict:
    """Entrena en ``tr`` (con validación interna por sujeto) y predice ``te``."""
    assert_disjoint(g, tr, te)
    rng_seed = cfg.seed
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    gen = torch.Generator().manual_seed(rng_seed)

    inner = StratifiedGroupKFold(n_splits=cfg.val_frac_splits, shuffle=True, random_state=rng_seed)
    itr_rel, iva_rel = next(inner.split(np.zeros(len(tr)), y[tr], g[tr]))
    itr, iva = tr[itr_rel], tr[iva_rel]
    assert_disjoint(g, itr, iva)
    assert_disjoint(g, iva, te)

    # Escalado por canal con estadísticos del entrenamiento interno (por bloques, sin copias completas).
    mu, sd = _channel_stats(X, itr)

    model = build_model(name, X.shape[1], X.shape[2], sfreq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    w = torch.tensor(len(itr) / (2 * np.bincount(y[itr], minlength=2)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    ytr = torch.from_numpy(y[itr]).long()

    best_state, best_val, wait, history = None, -np.inf, 0, []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = _augment(_batch(X, itr[idx.numpy()], mu, sd), cfg, gen).to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(_forward(model, xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        val_auc = _subject_auc(_predict(model, X, iva, mu, sd, device), y[iva], g[iva])
        history.append(val_auc)
        if val_auc > best_val + 1e-4:
            best_val, wait = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return {"prob": _predict(model, X, te, mu, sd, device), "best_val_auc": float(best_val), "epochs": len(history),
            "seconds": time.time() - t0}
