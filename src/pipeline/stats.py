"""Analise estatistica: Demsar (autorank) e Bayesiana com ROPE (baycomp).

Friedman + post-hoc + diagrama de diferenca critica via `autorank.autorank`.
Bayesian signed-rank test com ROPE via `baycomp.SignedRankTest`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from baycomp import SignedRankTest

DEFAULT_ROPE = 0.01


def demsar_analysis(
    results: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Roda Friedman + post-hoc + CD diagram via autorank.

    Args:
        results: DataFrame com formato `dataset x modelo`, valores sao a metrica
            (ex.: AUC OVO) na qual queremos comparar (maior = melhor).
        output_dir: se informado, salva CD diagram em PNG nesse diretorio.

    Returns:
        dict com 'ranking' (Series) e 'autorank_result' (objeto autorank).
    """
    result = autorank(results, alpha=0.05, verbose=False, order="descending")
    ranking = pd.Series(result.rankdf["meanrank"], name="mean_rank").sort_values()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_stats(result, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "cd_diagram.png", dpi=150)
        plt.close(fig)

    return {"ranking": ranking, "autorank_result": result}


def bayesian_pairwise(
    results: pd.DataFrame,
    rope: float = DEFAULT_ROPE,
) -> pd.DataFrame:
    """Roda Bayesian signed-rank test par a par com regiao de equivalencia ROPE.

    Args:
        results: DataFrame `dataset x modelo` com a metrica.
        rope: tamanho da regiao de equivalencia pratica (default: 0.01 em AUC).

    Returns:
        DataFrame com colunas (modelo_a, modelo_b, p_left, p_rope, p_right).
        p_left = prob. modelo_a < modelo_b - rope
        p_rope = prob. |modelo_a - modelo_b| <= rope
        p_right = prob. modelo_a > modelo_b + rope
    """
    rows = []
    models = list(results.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            scores_a = results[a].to_numpy()
            scores_b = results[b].to_numpy()
            test = SignedRankTest(x=scores_a, y=scores_b, rope=rope)
            p_left, p_rope, p_right = test.probs()
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "p_a_worse": float(p_left),
                    "p_equivalent": float(p_rope),
                    "p_a_better": float(p_right),
                }
            )
    return pd.DataFrame(rows)
