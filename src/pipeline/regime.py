"""Quebra dos resultados por regime de dataset.

Regimes considerados:
    - tamanho: small (n menor que 1k), medium (1k a 10k), large (acima de 10k)
    - num_classes: binary vs. multiclass
    - cat_share: low (menos de 25% das features sao categoricas), high (acima de 25%)
    - missing: yes vs. no
"""

from __future__ import annotations

import pandas as pd

from data.load_tabarena import classify_regime


def assign_regimes(metadata: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de regime a uma tabela de metadados de datasets.

    Args:
        metadata: DataFrame com colunas n_samples, n_features, n_classes,
            n_categorical, has_missing.

    Returns:
        DataFrame com colunas adicionais regime_size, regime_classes,
        regime_cat_share, regime_missing.
    """
    df = metadata.copy()
    df["regime_size"] = df["n_samples"].apply(classify_regime)
    df["regime_classes"] = df["n_classes"].apply(
        lambda k: "binary" if k == 2 else "multiclass"
    )
    df["regime_cat_share"] = (df["n_categorical"] / df["n_features"]).apply(
        lambda r: "high" if r >= 0.25 else "low"
    )
    df["regime_missing"] = df["has_missing"].map({True: "yes", False: "no"})
    return df


def aggregate_by_regime(
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    regime_col: str,
    metric_col: str = "auc_ovo",
) -> pd.DataFrame:
    """Agrega a metrica por modelo e por valor do regime.

    Args:
        results: DataFrame com colunas dataset_id, model, <metric_col>.
        metadata: DataFrame com regimes por dataset_id.
        regime_col: coluna do regime (ex.: 'regime_size').
        metric_col: coluna da metrica a agregar.

    Returns:
        DataFrame indexado por (regime, model) com media e desvio da metrica.
    """
    if "task_id" in metadata.columns and "task_id" not in results.columns:
        results = results.rename(columns={"dataset_id": "task_id"})
    merged = results.merge(metadata[["task_id", regime_col]], on="task_id")
    return (
        merged.groupby([regime_col, "model"])[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
