"""Calculo das metricas exigidas: AUC OVO, ACC, G-Mean, Cross-Entropy e tempo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    auc_ovo: float
    accuracy: float
    g_mean: float
    cross_entropy: float
    fit_time_s: float
    predict_time_s: float
    train_auc_ovo: float = float("nan")
    train_accuracy: float = float("nan")
    train_g_mean: float = float("nan")
    train_cross_entropy: float = float("nan")

    def to_dict(self) -> dict[str, float]:
        return {
            "auc_ovo": self.auc_ovo,
            "accuracy": self.accuracy,
            "g_mean": self.g_mean,
            "cross_entropy": self.cross_entropy,
            "fit_time_s": self.fit_time_s,
            "predict_time_s": self.predict_time_s,
            "total_time_s": self.fit_time_s + self.predict_time_s,
            "train_auc_ovo": self.train_auc_ovo,
            "train_accuracy": self.train_accuracy,
            "train_g_mean": self.train_g_mean,
            "train_cross_entropy": self.train_cross_entropy,
        }


def g_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """G-Mean: media geometrica do recall por classe."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    if not recalls:
        return 0.0
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0)))))


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
) -> tuple[float, float, float, float]:
    """Retorna (auc_ovo, accuracy, g_mean, cross_entropy) para um conjunto."""
    if y_proba is None:
        auc = float("nan")
        ce = float("nan")
    else:
        if classes.size == 2:
            auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovo", labels=classes))
        ce = float(log_loss(y_true, y_proba, labels=classes))
    return auc, float(accuracy_score(y_true, y_pred)), g_mean_score(y_true, y_pred), ce


def fit_predict_evaluate(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> EvaluationResult:
    """Treina e avalia um estimador no treino e no teste, medindo tempos."""
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0

    classes = np.unique(np.concatenate([y_train, y_test]))

    t0 = time.perf_counter()
    y_pred_test = estimator.predict(X_test)
    y_proba_test = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    predict_time_s = time.perf_counter() - t0

    auc, acc, gm, ce = _compute_metrics(y_test, y_pred_test, y_proba_test, classes)

    y_pred_train = estimator.predict(X_train)
    y_proba_train = estimator.predict_proba(X_train) if hasattr(estimator, "predict_proba") else None
    tr_auc, tr_acc, tr_gm, tr_ce = _compute_metrics(y_train, y_pred_train, y_proba_train, classes)

    return EvaluationResult(
        auc_ovo=auc,
        accuracy=acc,
        g_mean=gm,
        cross_entropy=ce,
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
        train_auc_ovo=tr_auc,
        train_accuracy=tr_acc,
        train_g_mean=tr_gm,
        train_cross_entropy=tr_ce,
    )
