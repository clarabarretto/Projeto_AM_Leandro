"""Baselines obrigatorios: LightGBM, XGBoost e CatBoost.

Por padrao usamos as variantes meta-tunadas do `pytabkit` (sufixo `_TD`), que
trazem hiperparametros default fortes obtidos por meta-aprendizado em centenas
de datasets (Holzmuller et al., NeurIPS 2024).
"""

from __future__ import annotations

from typing import Callable

from pytabkit import (
    LGBM_TD_Classifier,
    XGB_TD_Classifier,
    CatBoost_TD_Classifier,
)


def build_lightgbm(seed: int = 42) -> LGBM_TD_Classifier:
    """LightGBM com defaults meta-tunados (TD) do pytabkit."""
    return LGBM_TD_Classifier(random_state=seed)


def build_xgboost(seed: int = 42) -> XGB_TD_Classifier:
    """XGBoost com defaults meta-tunados (TD) do pytabkit."""
    return XGB_TD_Classifier(random_state=seed)


def build_catboost(seed: int = 42) -> CatBoost_TD_Classifier:
    """CatBoost com defaults meta-tunados (TD) do pytabkit."""
    return CatBoost_TD_Classifier(random_state=seed)


BASELINE_FACTORIES: dict[str, Callable[[int], object]] = {
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
}
