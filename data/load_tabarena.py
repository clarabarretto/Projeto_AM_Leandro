"""Carregamento padronizado dos 30 datasets do TabArena-v0.1.

Os 30 datasets sao selecionados a partir dos 51 datasets curados do TabArena-v0.1
(NeurIPS 2025), estratificados por regime de tamanho. Cada dataset e carregado
via OpenML, com cache local para evitar download repetido.

Para a lista oficial de task IDs do TabArena, consulte:
    https://tabarena.ai
    https://github.com/autogluon/tabarena

A constante RECOMMENDED_TASK_IDS abaixo contem 30 IDs estratificados por tamanho.
Caso a lista oficial seja atualizada, basta substituir os IDs aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import openml
import pandas as pd

CACHE_DIR = Path(os.environ.get("TABARENA_CACHE", "./cache/tabarena"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
openml.config.set_root_cache_directory(str(CACHE_DIR))

REGIME_THRESHOLDS = {"small": 1_000, "medium": 10_000}


RECOMMENDED_TASK_IDS: list[int] = [
    37, 359955, 15, 49, 3484,
    2, 11, 41, 3022, 3512,
    3892, 168757, 359956, 359967, 359968,
    45, 233214, 233215, 359935, 360932,
    3688, 3945, 168868, 359979, 360113,
    211986, 233211, 233212, 6, 26,
]
"""Lista placeholder de 30 task IDs do OpenML, cobrindo varios regimes de tamanho.

Substitua pela lista oficial do TabArena-v0.1 ao iniciar o experimento.
"""


@dataclass(frozen=True)
class TabularDataset:
    task_id: int
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical_indicator: list[bool]
    attribute_names: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def n_categorical(self) -> int:
        return int(sum(self.categorical_indicator))

    @property
    def has_missing(self) -> bool:
        return bool(self.X.isna().any().any())

    @property
    def regime(self) -> str:
        return classify_regime(self.n_samples)


def classify_regime(n_samples: int) -> str:
    """Retorna 'small', 'medium' ou 'large' conforme o numero de amostras."""
    if n_samples < REGIME_THRESHOLDS["small"]:
        return "small"
    if n_samples < REGIME_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def load_task(task_id: int) -> TabularDataset:
    """Carrega um dataset do OpenML pelo task_id, com cache."""
    task = openml.tasks.get_task(task_id, download_data=True)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    return TabularDataset(
        task_id=task_id,
        name=dataset.name,
        X=X,
        y=np.asarray(y),
        categorical_indicator=list(categorical_indicator),
        attribute_names=list(attribute_names),
    )


def iter_datasets(task_ids: list[int] | None = None) -> Iterator[TabularDataset]:
    """Itera sobre todos os datasets configurados."""
    ids = task_ids if task_ids is not None else RECOMMENDED_TASK_IDS
    for task_id in ids:
        yield load_task(task_id)


def summarize(task_ids: list[int] | None = None) -> pd.DataFrame:
    """Tabela-resumo (n_samples, n_features, n_classes, regime, missing)."""
    rows = []
    for ds in iter_datasets(task_ids):
        rows.append(
            {
                "task_id": ds.task_id,
                "name": ds.name,
                "n_samples": ds.n_samples,
                "n_features": ds.n_features,
                "n_classes": ds.n_classes,
                "n_categorical": ds.n_categorical,
                "has_missing": ds.has_missing,
                "regime": ds.regime,
            }
        )
    return pd.DataFrame(rows)
