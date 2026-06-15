"""Smoke test: roda baselines em um dataset pequeno (breast_cancer) e valida saidas.

O objetivo deste teste e detectar problemas de instalacao e incompatibilidade
de dependencias antes que o aluno comece o experimento completo. Nao testa
acuracia ou correcao do modelo, apenas que cada estimador retorna predicoes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from src.models.baselines import BASELINE_FACTORIES
from src.pipeline.evaluate import fit_predict_evaluate
from src.pipeline.split import stratified_split


@pytest.fixture(scope="module")
def small_dataset():
    """breast_cancer (n=569, 30 features, binario): roda em poucos segundos."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.to_numpy()
    return X, y


@pytest.mark.parametrize("model_name", sorted(BASELINE_FACTORIES.keys()))
def test_baseline_runs(small_dataset, model_name):
    X, y = small_dataset
    X_train, X_test, y_train, y_test = stratified_split(X, y, seed=42)
    estimator = BASELINE_FACTORIES[model_name](seed=42)
    metrics = fit_predict_evaluate(estimator, X_train, y_train, X_test, y_test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.auc_ovo <= 1.0
    assert metrics.fit_time_s >= 0.0
    assert metrics.predict_time_s >= 0.0


def test_split_is_stratified(small_dataset):
    X, y = small_dataset
    X_train, X_test, y_train, y_test = stratified_split(X, y, seed=42)
    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_split_seed_reproducibility(small_dataset):
    X, y = small_dataset
    a1, a2, _, _ = stratified_split(X, y, seed=42)
    b1, b2, _, _ = stratified_split(X, y, seed=42)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)


def test_evaluation_metrics_present(small_dataset):
    X, y = small_dataset
    X_train, X_test, y_train, y_test = stratified_split(X, y, seed=42)
    estimator = BASELINE_FACTORIES["lightgbm"](seed=42)
    metrics = fit_predict_evaluate(estimator, X_train, y_train, X_test, y_test)
    keys = set(metrics.to_dict().keys())
    expected = {
        "auc_ovo",
        "accuracy",
        "g_mean",
        "cross_entropy",
        "fit_time_s",
        "predict_time_s",
        "total_time_s",
    }
    assert expected.issubset(keys)


def test_g_mean_against_known():
    """Sanity-check do g_mean: classes equilibradas com 100% recall = 1.0."""
    from src.pipeline.evaluate import g_mean_score

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert g_mean_score(y_true, y_pred) == pytest.approx(1.0)
