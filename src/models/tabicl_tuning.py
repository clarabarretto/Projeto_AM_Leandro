"""Espaço de busca Optuna para TabICL v2.

Define o search_space e a factory para uso com src.pipeline.tune.tune().

Hiperparâmetros tunáveis do TabICL v2:
- n_estimators: número de membros do ensemble (mais = melhor, porém mais lento).
- softmax_temperature: controla a confiança das predições (menor = mais confiante).
- outlier_threshold: limiar de clipping de outliers nas features.
- average_logits: média em logits (True) ou em probabilidades (False).
- norm_methods: método(s) de normalização das features.

Referência: https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

from typing import Any

import optuna

from tabicl import TabICLClassifier


NORM_METHOD_OPTIONS = [
    "quantile",
    "robust",
    "power",
    "none",
]


def tabicl_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Amostra hiperparâmetros do TabICL v2 para um trial Optuna."""
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 4, 32, step=4),
        "softmax_temperature": trial.suggest_float(
            "softmax_temperature", 0.3, 1.5, step=0.1,
        ),
        "outlier_threshold": trial.suggest_float(
            "outlier_threshold", 2.0, 6.0, step=0.5,
        ),
        "average_logits": trial.suggest_categorical(
            "average_logits", [True, False],
        ),
    }

    # normalização: uma ou duas combinadas
    n_norms = trial.suggest_int("n_norm_methods", 1, 2)
    if n_norms == 1:
        params["norm_methods"] = trial.suggest_categorical(
            "norm_method_single", NORM_METHOD_OPTIONS,
        )
    else:
        m1 = trial.suggest_categorical("norm_method_1", NORM_METHOD_OPTIONS)
        m2 = trial.suggest_categorical("norm_method_2", NORM_METHOD_OPTIONS)
        params["norm_methods"] = [m1, m2]

    return params


def tabicl_factory(params: dict[str, Any], seed: int = 42) -> TabICLClassifier:
    """Cria um TabICLClassifier a partir de um dict de hiperparâmetros."""
    return TabICLClassifier(random_state=seed, verbose=False, **params)


def build_tabicl_factory(seed: int = 42):
    """Retorna uma factory compatível com tune() que recebe apenas params."""
    def factory(params: dict[str, Any]) -> TabICLClassifier:
        return tabicl_factory(params, seed=seed)
    return factory
