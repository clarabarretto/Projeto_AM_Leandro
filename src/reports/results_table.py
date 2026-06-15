"""Geracao de tabelas-resumo a partir do CSV de resultados brutos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_METRICS = ["auc_ovo", "accuracy", "g_mean", "cross_entropy", "total_time_s"]


def summary_by_model(
    raw_results: pd.DataFrame,
    metrics: list[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Tabela com media e desvio de cada metrica, agrupada por modelo."""
    agg = raw_results.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index().sort_values("auc_ovo_mean", ascending=False)


def pivot_for_stats(
    raw_results: pd.DataFrame,
    metric: str = "auc_ovo",
) -> pd.DataFrame:
    """Pivota para `dataset x modelo` no formato esperado por autorank/baycomp."""
    return raw_results.pivot(index="task_id", columns="model", values=metric)


def export_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    """Exporta a tabela-resumo em Markdown para inclusao no relatorio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.to_markdown(index=False, floatfmt=".4f"))
