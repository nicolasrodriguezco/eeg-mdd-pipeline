import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("braindecode")

from eeg_mdd.evaluation.cv import SubjectLeakError, outer_splits  # noqa: E402
from eeg_mdd.models.deep import DLConfig, build_model, train_fold  # noqa: E402


def test_models_build_and_output_two_classes():
    for name in ("EEGNet", "ShallowFBCSPNet", "TCN"):
        net = build_model(name, 8, 201, 250.0)
        out = net(torch.randn(3, 8, 201))
        out = out.mean(-1) if out.ndim == 3 else out
        assert out.shape == (3, 2)


def test_train_fold_runs_subject_disjoint_and_predicts_test_rows():
    rng = np.random.default_rng(0)
    n_subj, n_tr = 10, 12
    g = np.repeat([f"s{i}" for i in range(n_subj)], n_tr)
    y = np.repeat([1] * 5 + [0] * 5, n_tr)
    X = rng.normal(size=(len(y), 8, 201)).astype(np.float32)
    X[y == 1, :, 50:80] += 1.0
    _, _, tr, te = next(outer_splits(y, g, 5, 1))
    out = train_fold("EEGNet", X, y, g, tr, te, 250.0, DLConfig(max_epochs=2, patience=2, batch_size=32,
                                                                 val_frac_splits=4), device="cpu")
    assert out["prob"].shape == (len(te),)
    assert np.all((out["prob"] >= 0) & (out["prob"] <= 1))
    with pytest.raises(SubjectLeakError):
        train_fold("EEGNet", X, y, g, tr, np.r_[te, tr[:1]], 250.0, DLConfig(max_epochs=1), device="cpu")
