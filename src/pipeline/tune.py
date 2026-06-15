"""Busca de hiperparametros com Optuna em validacao cruzada (CV) no treino.

Implementacao minimalista que serve como ponto de partida. Cada grupo deve
adaptar o `search_space` para o seu modelo. Para baselines do `pytabkit` que
ja vem com defaults meta-tunados, o tuning pode ser opcional.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_N_TRIALS = 50
DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = "roc_auc_ovo"


def tune(
    estimator_factory: Callable[[dict[str, Any]], Any],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
) -> tuple[dict[str, Any], float]:
    """Roda Optuna no espaco de busca passado e retorna (melhor_params, melhor_score).

    Args:
        estimator_factory: funcao que recebe um dict de hiperparametros e
            retorna um estimador sklearn-compativel.
        search_space: funcao que recebe um trial e retorna o dict de
            hiperparametros amostrados.
        X, y: dados de treino (a CV roda dentro deste conjunto).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        estimator = estimator_factory(params)
        scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv, n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)
