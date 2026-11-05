"""La evaluación nunca mezcla sujetos y detecta (o no) señal donde corresponde."""

import numpy as np
import pandas as pd
import pytest

from eeg_mdd.evaluation import (aggregate_to_subject, bootstrap_ci, nested_cv, outer_splits, permutation_test,
                                subject_metrics)
from eeg_mdd.evaluation.cv import SubjectLeakError, assert_disjoint
from eeg_mdd.evaluation.nested import corrected_ttest
from eeg_mdd.models import model_selection_grid, riemann_models, tabular_models


def _trial_data(n_subj=20, n_trials=15, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y_s = np.array([1] * (n_subj // 2) + [0] * (n_subj - n_subj // 2))
    subj_offset = rng.normal(0, 2, (n_subj, 4))           # identidad del sujeto: muy fuerte
    X, y, g = [], [], []
    for s in range(n_subj):
        x = rng.normal(0, 1, (n_trials, 4)) + subj_offset[s]
        x[:, 0] += effect * y_s[s]
        X.append(x); y += [y_s[s]] * n_trials; g += [f"s{s:02d}"] * n_trials
    return np.vstack(X), np.array(y), np.array(g)


def test_every_outer_fold_is_subject_disjoint_and_covers_all_subjects():
    X, y, g = _trial_data()
    seen = {}
    for r, k, tr, te in outer_splits(y, g, n_splits=5, n_repeats=3):
        assert not set(g[tr]) & set(g[te])
        for s in set(g[te]):
            seen.setdefault(r, []).append(s)
    for r, subs in seen.items():
        assert sorted(subs) == sorted(set(g))                 # cada sujeto se prueba una vez por repetición


def test_assert_disjoint_raises_on_leak():
    g = np.array(["a", "a", "b", "c"])
    with pytest.raises(SubjectLeakError):
        assert_disjoint(g, np.array([0, 2]), np.array([1, 3]))


def test_aggregation_and_metrics():
    agg = aggregate_to_subject(np.array([0.9, 0.7, 0.2, 0.4]), np.array([1, 1, 0, 0]), np.array(["a", "a", "b", "b"]))
    assert agg.set_index("subject_id")["prob"].to_dict() == pytest.approx({"a": 0.8, "b": 0.3})
    m = subject_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.1]))
    assert m["auc"] == 1 and m["balanced_accuracy"] == 1 and m["sensitivity"] == 1 and m["specificity"] == 1
    with pytest.raises(ValueError):
        aggregate_to_subject(np.array([0.5, 0.5]), np.array([1, 0]), np.array(["a", "a"]))


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 40), 0, 1)
    ci = bootstrap_ci(y, p, n_boot=300)
    assert (ci["ci_low"] <= ci["estimate"]).all() and (ci["estimate"] <= ci["ci_high"]).all()


def test_no_signal_gives_chance_even_with_strong_subject_identity():
    """Con identidad de sujeto fuerte y SIN efecto de grupo, la CV por sujeto debe dar azar."""
    X, y, g = _trial_data(effect=0.0, seed=1)
    pipe, grid = tabular_models()["LR-L2"]
    res = nested_cv("null", pipe, {"clf__C": [0.1, 1.0]}, X, y, g, n_splits=5, n_repeats=3, n_inner=3)
    assert 0.2 < res.summary()["auc"] < 0.8
    assert set(res.oof["subject_id"]) == set(g)


def test_real_signal_is_detected():
    X, y, g = _trial_data(effect=8.0, seed=2)
    pipe, _ = tabular_models()["LR-L2"]
    res = nested_cv("signal", pipe, {"clf__C": [0.1, 1.0]}, X, y, g, n_splits=5, n_repeats=2, n_inner=3)
    assert res.summary()["auc"] > 0.9
    ci = res.ci(n_boot=200)
    assert ci.loc["auc", "ci_low"] > 0.6


def test_permutation_test_p_value_small_with_signal():
    X, y, g = _trial_data(n_subj=16, n_trials=5, effect=5.0, seed=3)
    pipe, _ = tabular_models()["LR-L2"]
    obs = nested_cv("s", pipe, {}, X, y, g, n_splits=4, n_repeats=1).repeat_scores["auc"].iloc[0]
    res = permutation_test(pipe, {}, X, y, g, obs, n_perm=19, n_splits=4)
    assert res["p_value"] == pytest.approx(0.05)
    assert res["null_mean"] < obs


def test_model_selection_grid_and_riemann_pipelines_run():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    y = np.r_[np.ones(15), np.zeros(15)].astype(int)
    X.iloc[:, 0] += 4 * y
    g = np.array([f"s{i}" for i in range(30)])
    pipe, grids = model_selection_grid()
    small = [{k: (v if k == "clf" else list(v)[-1:]) for k, v in d.items()} for d in grids]
    res = nested_cv("sel", pipe, small, X, y, g, n_splits=3, n_repeats=1, n_inner=3)
    assert res.summary()["auc"] > 0.7
    Xt = rng.normal(size=(40, 3, 50))
    yt = np.r_[np.ones(20), np.zeros(20)].astype(int)
    gt = np.repeat([f"s{i}" for i in range(8)], 5)
    yt = np.repeat(np.r_[np.ones(4), np.zeros(4)], 5).astype(int)
    for name, (p, grid) in riemann_models().items():
        r = nested_cv(name, p, {}, Xt, yt, gt, n_splits=2, n_repeats=1)
        assert len(r.oof) == 8


def test_corrected_ttest():
    out = corrected_ttest(np.array([0.1, 0.12, 0.08, 0.11, 0.09]), n_train=40, n_test=10)
    assert out["mean_diff"] == pytest.approx(0.1) and out["p_value"] < 0.05


def test_same_subject_partitions_for_subject_and_trial_level_data():
    X, y, g = _trial_data(n_subj=20, n_trials=7)
    subj = pd.DataFrame({"g": g, "y": y}).drop_duplicates("g")
    a = [(r, k, frozenset(subj["g"].to_numpy()[te])) for r, k, tr, te in
         outer_splits(subj["y"].to_numpy(), subj["g"].to_numpy(), 5, 2)]
    b = [(r, k, frozenset(g[te])) for r, k, tr, te in outer_splits(y, g, 5, 2)]
    assert a == b


def test_precomputed_cov_equivalent():
    """TS-LR/MDM sobre covarianzas precalculadas == mismo modelo con la covarianza dentro del Pipeline."""
    from pyriemann.estimation import Covariances

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4, 50))
    y = np.repeat([0, 1], 60)
    X[y == 1, 0] *= 1.5
    C = Covariances(estimator="oas").transform(X)
    raw, pre = riemann_models(), riemann_models(precomputed_cov=True)
    for name in ("TS-LR", "MDM"):
        a = raw[name][0].fit(X[:80], y[:80]).predict_proba(X[80:])
        b = pre[name][0].fit(C[:80], y[:80]).predict_proba(C[80:])
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
